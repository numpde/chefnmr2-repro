"""Resolve with uv, select with the SDK tool, and assemble images offline."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
from fetch_sdk import artifact_names, release_url
from work import temporary_run_directory

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


class Build:
    def __init__(self, host, pool, proxy=None):
        if os.getuid() == 0:
            raise ValueError('Run dependency and image operations as the non-root checkout owner so cache files retain that ownership')

        if not host:
            raise ValueError('Select a local Docker host socket explicitly for image and dependency operations')

        if not pool:
            raise ValueError('Select WHEELHOUSE explicitly; build inputs come from that shared pool')

        self.pool = Path(pool).resolve(strict=True)

        if not self.pool.is_dir() or (self.pool == ROOT) or (ROOT in self.pool.parents) or (self.pool in ROOT.parents):
            raise ValueError(f'WHEELHOUSE={str(self.pool)!r} should be a directory outside the checkout and its ancestors')

        self.environment = {key: value for (key, value) in os.environ.items() if not key.startswith(('COMPOSE_', 'BUILDX_')) and (key not in ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH'))}

        (self.host, self.proxy) = (host, proxy)
        self.docker = ['docker', '--host', host]
        self.sdk = re.search(r'^provider-core==([^\s]+)$', (ROOT / 'builds/sdk.constraints.txt').read_text(), re.M).group(1)
        self.manifest_hash = re.search(r'^# release-manifest-sha256 = ([0-9a-f]{64})$', (ROOT / 'builds/sdk.constraints.txt').read_text(), re.M).group(1)
        self.release = release_url(self.sdk)
        self.tool = self.pool / f'wheelhouse-tool-{self.sdk}.pyz'
        self.hashes = self.pool / f'provider_core-{self.sdk}-sha256.json'

    def run(self, arguments, **kwargs):
        try:
            return subprocess.run(arguments, cwd=ROOT, env=self.environment, check=True, **kwargs)
        except subprocess.CalledProcessError as error:
            if error.stderr:
                print(error.stderr, file=sys.stderr, end='')

            raise

    def bake(self, target):
        document = json.loads(self.run(self.docker + ['buildx', 'bake', '-f', 'compose/images.yaml', '--print', target], capture_output=True, text=True).stdout)

        return document['target'][target]

    def require_image(self, image):
        self.run(self.docker + ['image', 'inspect', image], stdout=subprocess.DEVNULL)

    def sdk_artifacts(self, acquire=False):
        log.info('Checking SDK %s artifacts in %s', self.sdk, self.pool)

        if acquire:
            image = self.bake('provider')['args']['PYTHON_BASE']

            self.require_image(image)
            log.info('Acquiring missing SDK artifacts')

            self.run(
                self.acquisition_command(
                    image,
                    [(self.pool, '/pool'), (ROOT / 'make/fetch_sdk.py', '/fetch_sdk.py')],
                    ['-B', '/fetch_sdk.py', self.sdk, self.manifest_hash, '/pool'],
                    writable_pool=True
                )
            )

        if digest(self.hashes) != self.manifest_hash:
            raise ValueError('Cannot use the SDK checksum manifest: SHA-256 differs from builds/sdk.constraints.txt')

        hashes = json.loads(self.hashes.read_text())

        for name in artifact_names(self.sdk):
            path = self.pool / name

            if digest(path) != hashes[name]:
                raise ValueError(f'Cannot use {path.name}: its SHA-256 differs from the selected SDK release')

        log.info('SDK %s verified and ready', self.sdk)

    def container_options(self, mounts, writable_pool=False, scratch=None):
        command = ['--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--cpus', '2', '--memory', '4g', '--tmpfs', '/tmp:rw,nosuid,nodev,size=3g', '-e', 'HOME=/tmp', '-e', 'PROVIDER_CORE_IN_CONTAINER=1']

        if scratch is not None:
            mounts = [*mounts, (scratch, '/scratch')]
            command += ['-e', 'UV_CACHE_DIR=/scratch']

        for (source, target) in mounts:
            if ',' in str(source):
                raise ValueError(f'Cannot mount {source}: Docker bind paths cannot contain commas')

            command += ['--mount', f'type=bind,src={source},dst={target}' + ('' if (writable_pool and (target == '/pool')) or (target == '/scratch') else ',readonly')]

        return command

    def container(self, image, mounts, writable_pool=False, scratch=None):
        user = '65532:65532'

        if writable_pool or (scratch is not None):
            options = json.loads(self.run(self.docker + ['info', '--format', '{{json .SecurityOptions}}'], capture_output=True, text=True).stdout)

            if 'name=rootless' not in options:
                raise ValueError('Resolver scratch requires a rootless engine so host files retain the invoking user')

            user = '0:0'

        return self.docker + ['run', '--rm', '--pull=never', '--network', 'none', '--user', user] + self.container_options(mounts, writable_pool, scratch) + ['--entrypoint', 'python', image]

    def acquisition_command(self, image, mounts, arguments, writable_pool=False):
        if self.proxy is None:
            raise ValueError('Acquisition requires explicit PROXY_URL=<url> or PROXY_URL=none')

        return [
            'bash',
            str(ROOT / 'make/acquisition.sh'),
            self.host,
            image,
            self.proxy,
            *self.container_options(mounts, writable_pool),
            '--entrypoint',
            'python',
            '--',
            *arguments
        ]

    def sdk_tool(self, image, stage, *arguments, online=False):
        mounts = [(self.pool, '/pool'), (stage, '/stage')]
        application = ['-B', '/pool/' + self.tool.name, *arguments]

        if online:
            command = self.acquisition_command(image, mounts, application, writable_pool=arguments[0] == 'acquire')
        else:
            command = self.container(image, mounts) + application

        return self.run(command, stdout=subprocess.PIPE, text=True).stdout

    def prepare_lock_image(self, target):
        log.info('Preparing the dependency resolver image')
        base = target['args']['PYTHON_BASE']
        lock = self.bake('lock')

        self.require_image(base)
        self.require_image(lock['contexts']['uv'].removeprefix('docker-image://'))
        self.run(self.docker + ['buildx', 'bake', '-f', 'compose/images.yaml', 'lock', '--set', 'lock.args.PYTHON_BASE=' + base, '--load'])

        return lock['tags'][0]

    def input_identity(self, profile, target):
        directory = ROOT / 'builds' / profile
        inputs = {str(path.relative_to(ROOT)): digest(path) for path in (ROOT / 'builds/sdk.constraints.txt', directory / 'requirements.in')}

        for constraint in sorted(directory.glob('*.constraints.txt')):
            inputs[str(constraint.relative_to(ROOT))] = digest(constraint)

        release_lock = f'provider_core-{self.sdk}-install.lock'
        inputs['sdk-release-lock'] = digest(self.pool / release_lock)
        inputs['sdk-tool'] = digest(self.tool)
        inputs['target'] = {'args': target['args'], 'platforms': target['platforms']}
        inputs['uv'] = self.bake('lock')['contexts']['uv']

        return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

    def lock(self, profile):
        self.sdk_artifacts()
        target = self.bake(profile)
        image = self.prepare_lock_image(target)
        directory = ROOT / 'builds' / profile
        release_lock = f'provider_core-{self.sdk}-install.lock'

        with temporary_run_directory(ROOT, 'lock') as stage:
            scratch = stage / 'scratch'

            scratch.mkdir(mode=0o700)
            command = self.container(image, [(ROOT / 'builds', '/input'), (self.pool, '/pool')], scratch=scratch)
            command[command.index('--entrypoint') + 1] = 'uv'
            command += ['pip', 'compile', f'/input/{profile}/requirements.in', '-c', '/input/sdk.constraints.txt', '-c', '/pool/' + release_lock, '--no-python-downloads', '--python-version', target['args']['PY_VERSION'], '--generate-hashes', '--no-header', '--no-annotate', '--find-links', '/pool']

            for constraint in sorted(directory.glob('*.constraints.txt')):
                command += ['-c', f'/input/{profile}/{constraint.name}']

            command += ['--offline', '--no-index']

            log.info('Resolving %s dependencies from cached packages', profile)
            resolved = self.run(command, stdout=subprocess.PIPE, text=True).stdout

            # OmegaConf's sole source-only dependency is built in its own image
            # stage. Add its original archive identity to the resolver's hashes.
            if 'antlr4-python3-runtime==' in resolved:
                version = re.search(r'^antlr4-python3-runtime==([^\s]+)', resolved, re.M).group(1)
                archive = self.pool / f'antlr4-python3-runtime-{version}.tar.gz'

                for wheel in self.pool.glob(f'antlr4_python3_runtime-{version}-*.whl'):
                    resolved = resolved.replace(digest(wheel), digest(archive))

            (stage / 'resolved.txt').write_text(resolved)
            arguments = ['locate', '--lock', '/stage/resolved.txt', '--pool', '/pool', '--index-url', 'https://pypi.org/simple']

            if 'provider-core==' in resolved:
                sdk_hash = json.loads(self.hashes.read_text())[f'provider_core-{self.sdk}-py3-none-any.whl']
                arguments += ['--source', self.release + f'provider_core-{self.sdk}-py3-none-any.whl#sha256={sdk_hash}']

            if 'torch==' in resolved:
                version = re.search(r'^torch==([^\s]+)', resolved, re.M).group(1)
                wheel = next(self.pool.glob(f'torch-{version}-*.whl'))
                url = target['args']['TORCH_INDEX'] + '/' + urllib.parse.quote(wheel.name)
                arguments += ['--source', url + '#sha256=' + digest(wheel)]

            if 'antlr4-python3-runtime==' in resolved:
                arguments += ['--require-source', 'antlr4-python3-runtime']

            log.info('Locating published artifact URLs for %s', profile)
            projected = self.sdk_tool(image, stage, *arguments, online=True)

            # The SDK's PEP 751 projection is the complete artifact lock.
            # The .lock filename is copied to a pylock name when invoking the tool.
            output = projected + '\n[tool.chefnmr2]\ninputs-sha256 = ' + json.dumps(self.input_identity(profile, target)) + '\nresolution = ' + json.dumps(resolved) + '\n'

            with tempfile.NamedTemporaryFile(mode='w', dir=directory, prefix='.requirements-', delete=False) as generated:
                temporary_lock = Path(generated.name)

                try:
                    generated.write(output)
                    generated.flush()
                    os.fsync(generated.fileno())
                    temporary_lock.chmod(0o644)
                    os.replace(temporary_lock, directory / 'requirements.lock')
                finally:
                    temporary_lock.unlink(missing_ok=True)

        log.info('Saved dependency lock: %s', directory / 'requirements.lock')

    @contextmanager
    def selected(self, profile, acquire=False):
        self.sdk_artifacts()
        log.info('Checking %s build inputs and cached base image', profile)
        target = self.bake(profile)
        base = target['args']['PYTHON_BASE']

        self.require_image(base)

        with temporary_run_directory(ROOT, 'build') as stage:
            shutil.copyfile(ROOT / 'builds' / profile / 'requirements.lock', stage / 'pylock.profile.toml')
            document = tomllib.loads((stage / 'pylock.profile.toml').read_text())

            if document['tool']['chefnmr2']['inputs-sha256'] != self.input_identity(profile, target):
                raise ValueError(f'Cannot use the {profile} lock: build inputs changed; regenerate it with make lock')

            (stage / 'resolution.lock').write_text(document['tool']['chefnmr2']['resolution'])
            source_options = []

            for package in document['packages']:
                if 'sdist' in package:
                    if package['name'] != 'antlr4-python3-runtime':
                        raise ValueError(f"Cannot select source package {package['name']!r}: only antlr has an explicit offline build")

                    source_options += ['--require-sdist', package['name']]

            if acquire:
                log.info('Acquiring locked %s dependencies into %s', profile, self.pool)
                self.sdk_tool(base, stage, 'acquire', '--lock', '/stage/pylock.profile.toml', '--source-lock', '/stage/resolution.lock', '--pool', '/pool', *source_options, online=True)

            log.info('Selecting locked %s artifacts', profile)
            selected = json.loads(self.sdk_tool(base, stage, 'select', '--lock', '/stage/pylock.profile.toml', '--source-lock', '/stage/resolution.lock', *source_options))['artifacts']

            log.info('Selected %d locked artifacts for %s', len(selected), profile)
            yield (target, stage, selected)

    def acquire(self, profile):
        with self.selected(profile, acquire=True):
            pass

        log.info('Dependencies ready for %s', profile)

    def build(self, profile):
        with self.selected(profile) as (target, stage, artifacts):
            (install, sources, build) = ([], [], [])

            for artifact in artifacts:
                (filename, checksum) = (artifact['filename'], artifact['sha256'])
                requirement = f'/wheels/{filename} --hash=sha256:{checksum}\n'

                if filename.endswith('.whl'):
                    install.append(requirement)

                    if artifact['package'] == 'setuptools':
                        build.append(requirement)
                elif artifact['package'] == 'antlr4-python3-runtime':
                    sources.append(requirement)
                else:
                    raise ValueError(f'Cannot build {filename}: only the declared antlr source dependency has an offline wheel build')

            for (name, lines) in (('install.txt', install), ('sources.txt', sources), ('build.txt', build)):
                (stage / name).write_text(''.join(lines))

            log.info('Building %s offline', profile)

            self.run(
                self.docker + [
                    'buildx',
                    'bake',
                    '-f',
                    'compose/images.yaml',
                    profile,
                    '--allow',
                    'fs.read=' + str(self.pool),
                    '--allow',
                    'fs.read=' + str(stage),
                    '--set',
                    profile + '.contexts.wheels=' + str(self.pool),
                    '--set',
                    profile + '.contexts.installation=' + str(stage),
                    '--load'
                ]
            )

            image = json.loads(self.run(self.docker + ['image', 'inspect', target['tags'][0]], capture_output=True, text=True).stdout)[0]['Id']

        print(json.dumps({'profile': profile, 'image': image}))
        log.info('Build complete: %s', profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('action', choices=('sdk/acquire', 'lock', 'acquire', 'build'))
    parser.add_argument('--profile', default='')
    parser.add_argument('--host', required=True, help='Resolved local Unix socket for the selected Docker context')
    parser.add_argument('--pool', required=True)
    parser.add_argument('--proxy')
    args = parser.parse_args()

    log.info('Starting %s%s', args.action, f' for {args.profile}' if args.profile else '')
    build = Build(args.host, args.pool, args.proxy)

    if args.action == 'sdk/acquire':
        build.sdk_artifacts(acquire=True)
    elif args.action == 'lock':
        build.lock(args.profile)
    elif args.action == 'acquire':
        build.acquire(args.profile)
    else:
        build.build(args.profile)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
