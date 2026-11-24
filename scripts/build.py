"""Resolve with uv, select with the SDK tool, and assemble images offline."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


class Build:
    def __init__(self, context, pool):
        if os.getuid() == 0:
            raise ValueError('Run dependency and image operations as the non-root checkout owner so cache files retain that ownership')
        if not context:
            raise ValueError('Select DOCKER_CONTEXT explicitly for image and dependency operations')
        if not pool:
            raise ValueError('Select WHEELHOUSE explicitly; build inputs come from that shared pool')
        self.pool = Path(pool).resolve(strict=True)
        if not self.pool.is_dir() or self.pool == ROOT or ROOT in self.pool.parents or self.pool in ROOT.parents:
            raise ValueError(f'WHEELHOUSE={str(self.pool)!r} should be a directory outside the checkout and its ancestors')
        self.environment = {key: value for key, value in os.environ.items() if not key.startswith(('COMPOSE_', 'BUILDX_')) and key not in ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH')}
        self.docker = ['docker', '--context', context]
        self.sdk = re.search(r'^provider-core==([^\s]+)$', (ROOT / 'builds/sdk.constraints.txt').read_text(), re.M).group(1)
        self.manifest_hash = re.search(r'^# release-manifest-sha256 = ([0-9a-f]{64})$', (ROOT / 'builds/sdk.constraints.txt').read_text(), re.M).group(1)
        self.release = f'https://github.com/numpde/nmr-provider-tools/releases/download/v{self.sdk}/'
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
        document = json.loads(self.run(self.docker + ['buildx', 'bake', '-f', 'docker-bake.hcl', '--print', target], capture_output=True, text=True).stdout)
        return document['target'][target]

    def require_image(self, image):
        self.run(self.docker + ['image', 'inspect', image], stdout=subprocess.DEVNULL)

    def sdk_artifacts(self, acquire=False):
        if acquire:
            self.fetch(self.hashes.name, self.manifest_hash)
        if digest(self.hashes) != self.manifest_hash:
            raise ValueError('Cannot use the SDK checksum manifest: SHA-256 differs from builds/sdk.constraints.txt')
        hashes = json.loads(self.hashes.read_text())
        names = [self.tool.name, f'provider_core-{self.sdk}-native-install.lock', f'provider_core-{self.sdk}-install.lock', f'provider_core-{self.sdk}-py3-none-any.whl']
        for name in names:
            path = self.pool / name
            if acquire and not path.exists():
                self.fetch(name, hashes[name])
            if digest(path) != hashes[name]:
                raise ValueError(f'Cannot use {path.name}: its SHA-256 differs from the selected SDK release')

    def fetch(self, name, expected=None):
        with tempfile.NamedTemporaryFile(dir=self.pool, prefix='.sdk-', delete=False) as output:
            temporary = Path(output.name)
            try:
                with urllib.request.urlopen(self.release + name, timeout=120) as response:
                    total = 0
                    while block := response.read(65536):
                        total += len(block)
                        if total > 8 * 1024 * 1024:
                            raise ValueError(f'Cannot acquire {name}: SDK support artifacts should fit within the 8 MiB download limit')
                        output.write(block)
                output.close()
                if expected is not None and digest(temporary) != expected:
                    raise ValueError(f'Cannot acquire {name}: downloaded bytes differ from the SDK release SHA-256')
                temporary.chmod(0o644)
                os.replace(temporary, self.pool / name)
            finally:
                temporary.unlink(missing_ok=True)

    def container(self, image, mounts, network='none', writable_pool=False, scratch=None):
        command = self.docker + ['run', '--rm', '--pull=never', '--network', network, '--read-only', '--user', '65532:65532', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--cpus', '2', '--memory', '4g', '--tmpfs', '/tmp:rw,nosuid,nodev,size=3g', '-e', 'HOME=/tmp', '-e', 'PROVIDER_CORE_IN_CONTAINER=1']
        if writable_pool or scratch is not None:
            options = json.loads(self.run(self.docker + ['info', '--format', '{{json .SecurityOptions}}'], capture_output=True, text=True).stdout)
            if 'name=rootless' not in options:
                raise ValueError('Acquisition and resolver scratch require a rootless engine so host files retain the invoking user')
            command[command.index('--user') + 1] = '0:0'
        if scratch is not None:
            mounts = [*mounts, (scratch, '/scratch')]
            command += ['-e', 'UV_CACHE_DIR=/scratch']
        for source, target in mounts:
            if ',' in str(source):
                raise ValueError(f'Cannot mount {source}: Docker bind paths cannot contain commas')
            command += ['--mount', f'type=bind,src={source},dst={target}' + ('' if (writable_pool and target == '/pool') or target == '/scratch' else ',readonly')]
        return command + ['--entrypoint', 'python', image]

    def sdk_tool(self, image, stage, *arguments, network='none'):
        return self.run(self.container(image, [(self.pool, '/pool'), (stage, '/stage')], network, writable_pool=arguments[0] == 'acquire') + ['-B', '/pool/' + self.tool.name, *arguments], capture_output=True, text=True).stdout

    def prepare_lock_image(self, target):
        base = target['args']['PYTHON_BASE']
        lock = self.bake('lock')
        self.require_image(base)
        self.require_image(lock['contexts']['uv'].removeprefix('docker-image://'))
        self.run(self.docker + ['buildx', 'bake', '-f', 'docker-bake.hcl', 'lock', '--set', 'lock.args.PYTHON_BASE=' + base, '--load'])
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
        with tempfile.TemporaryDirectory(prefix='chefnmr2-lock-') as temporary:
            stage = Path(temporary)
            stage.chmod(0o755)
            scratch = stage / 'scratch'
            scratch.mkdir(mode=0o700)
            command = self.container(image, [(ROOT / 'builds', '/input'), (self.pool, '/pool')], scratch=scratch)
            command[command.index('--entrypoint') + 1] = 'uv'
            command += ['pip', 'compile', f'/input/{profile}/requirements.in', '-c', '/input/sdk.constraints.txt', '-c', '/pool/' + release_lock, '--no-python-downloads', '--python-version', target['args']['PY_VERSION'], '--generate-hashes', '--no-header', '--no-annotate', '--find-links', '/pool']
            for constraint in sorted(directory.glob('*.constraints.txt')):
                command += ['-c', f'/input/{profile}/{constraint.name}']
            command += ['--offline', '--no-index']
            resolved = self.run(command, capture_output=True, text=True).stdout
            # OmegaConf's sole source-only dependency is built in its own image
            # stage. Add its original archive identity to the resolver's hashes.
            if 'antlr4-python3-runtime==' in resolved:
                version = re.search(r'^antlr4-python3-runtime==([^\s]+)', resolved, re.M).group(1)
                archive = self.pool / f'antlr4-python3-runtime-{version}.tar.gz'
                for wheel in self.pool.glob(f'antlr4_python3_runtime-{version}-*.whl'):
                    resolved = resolved.replace(digest(wheel), digest(archive))
            (stage / 'resolved.txt').write_text(resolved)
            sdk_hash = json.loads(self.hashes.read_text())[f'provider_core-{self.sdk}-py3-none-any.whl']
            arguments = ['locate', '--lock', '/stage/resolved.txt', '--pool', '/pool', '--index-url', 'https://pypi.org/simple', '--source', self.release + f'provider_core-{self.sdk}-py3-none-any.whl#sha256={sdk_hash}']
            if 'torch==' in resolved:
                version = re.search(r'^torch==([^\s]+)', resolved, re.M).group(1)
                wheel = next(self.pool.glob(f'torch-{version}-*.whl'))
                url = target['args']['TORCH_INDEX'] + '/' + urllib.parse.quote(wheel.name)
                arguments += ['--source', url + '#sha256=' + digest(wheel), '--require-source', 'antlr4-python3-runtime']
            projected = self.sdk_tool(image, stage, *arguments, network='bridge')
            # The SDK's PEP 751 projection is the complete artifact lock. The
            # .lock filename is copied to a pylock name when invoking the tool.
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

    @contextmanager
    def selected(self, profile, acquire=False):
        self.sdk_artifacts()
        target = self.bake(profile)
        base = target['args']['PYTHON_BASE']
        self.require_image(base)
        with tempfile.TemporaryDirectory(prefix='chefnmr2-inputs-') as temporary:
            stage = Path(temporary)
            stage.chmod(0o755)
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
            self.sdk_tool(base, stage, 'acquire' if acquire else 'check', '--lock', '/stage/pylock.profile.toml', '--source-lock', '/stage/resolution.lock', '--pool', '/pool', *source_options, network='bridge' if acquire else 'none')
            selected = json.loads(self.sdk_tool(base, stage, 'select', '--lock', '/stage/pylock.profile.toml', '--source-lock', '/stage/resolution.lock', *source_options))['artifacts']
            yield target, stage, selected

    def acquire(self, profile):
        with self.selected(profile, acquire=True):
            pass

    def build(self, profile):
        with self.selected(profile) as (target, stage, artifacts):
            wheels = stage / 'wheels'
            wheels.mkdir()
            install, sources, build = [], [], []
            descriptor = os.open(self.pool, os.O_RDONLY | os.O_DIRECTORY)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH)
                for artifact in artifacts:
                    filename, checksum = artifact['filename'], artifact['sha256']
                    destination = wheels / filename
                    shutil.copyfile(self.pool / filename, destination)
                    if digest(destination) != checksum:
                        raise ValueError(f'Cannot stage {filename}: SHA-256 differs from the selected profile lock')
                    requirement = f'/wheels/{filename} --hash=sha256:{checksum}\n'
                    if filename.endswith('.whl'):
                        install.append(requirement)
                        if artifact['package'] == 'setuptools':
                            build.append(requirement)
                    elif artifact['package'] == 'antlr4-python3-runtime':
                        sources.append(requirement)
                    else:
                        raise ValueError(f'Cannot build {filename}: only the declared antlr source dependency has an offline wheel build')
            finally:
                os.close(descriptor)
            (wheels / 'install.txt').write_text(''.join(install))
            (wheels / 'sources.txt').write_text(''.join(sources))
            (wheels / 'build.txt').write_text(''.join(build))
            self.run(self.docker + ['buildx', 'bake', '--allow', 'fs.read=' + str(wheels), '-f', 'docker-bake.hcl', profile, '--set', profile + '.contexts.wheels=' + str(wheels), '--load'])
            print(json.dumps({'profile': profile, 'image': json.loads(self.run(self.docker + ['image', 'inspect', target['tags'][0]], capture_output=True, text=True).stdout)[0]['Id']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('sdk/acquire', 'lock', 'acquire', 'build'))
    parser.add_argument('--profile', default='')
    parser.add_argument('--context', required=True)
    parser.add_argument('--pool', required=True)
    args = parser.parse_args()
    build = Build(args.context, args.pool)
    if args.action == 'sdk/acquire':
        build.sdk_artifacts(acquire=True)
    elif args.action == 'lock':
        build.lock(args.profile)
    elif args.action == 'acquire':
        build.acquire(args.profile)
    else:
        build.build(args.profile)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
