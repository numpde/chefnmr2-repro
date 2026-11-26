"""Resolve complete role wheel locks and assemble their checked inputs offline."""

import argparse
from contextlib import contextmanager
import errno
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
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

    def wheelhouse_artifacts(self, acquire=False):
        pin = json.loads((ROOT / 'builds/wheelhouse-release.json').read_text())
        tool = self.pool / pin['name']

        if acquire:
            image = self.bake('worker-cpu')['args']['PYTHON_BASE']

            self.require_image(image)

            self.run(
                self.acquisition_command(
                    image,
                    [
                        (self.pool, '/pool'),
                        (ROOT / 'builds/wheelhouse-release.json', '/wheelhouse-release.json'),
                        (ROOT / 'make/fetch_artifact.py', '/fetch_artifact.py'),
                    ],
                    [
                        '-B',  # Keep bytecode out of captured source mounts.
                        '/fetch_artifact.py',  # Use the shared exact artifact fetch owner.
                        '/wheelhouse-release.json',  # Read the exact tool URL and digest.
                        '/pool',  # Retain verified bytes in the shared pool.
                    ],
                    writable_pool=True,
                )
            )

        if not tool.is_file():
            raise FileNotFoundError(f'Cannot use independent Wheelhouse: {tool=!s} is missing; run make wheelhouse/acquire')

        if digest(tool) != pin['sha256']:
            raise ValueError(f'Cannot use independent Wheelhouse: {tool=!s} has a different SHA-256 from builds/wheelhouse-release.json')

        return (tool, pin['sha256'])

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

    def wheelhouse_tool(self, image, stage, *arguments, online=False, scratch=None):
        """Run the captured tool for the selected target while streaming progress to stderr."""
        mounts = [(self.pool, '/pool'), (stage, '/stage')]

        application = [
            '-B',  # Keep bytecode out of captured source mounts.
            '/stage/wheelhouse.pyz',  # Run the captured independently pinned tool.
            *arguments,  # Pass the caller's complete lock operation unchanged.
        ]

        if online:
            if scratch is not None:
                mounts.append((scratch, '/scratch'))

            command = self.acquisition_command(image, mounts, application, writable_pool=(arguments[0] == 'acquire'))
        else:
            command = self.container(image, mounts, scratch=scratch) + application

        return self.run(command, stdout=subprocess.PIPE, text=True).stdout

    def prepare_lock_image(self, target):
        log.info('Preparing the dependency resolver image')
        base = target['args']['PYTHON_BASE']
        lock = self.bake('lock')

        self.require_image(base)
        self.require_image(lock['contexts']['uv'].removeprefix('docker-image://'))
        self.run(self.docker + ['buildx', 'bake', '-f', 'compose/images.yaml', 'lock', '--set', 'lock.args.PYTHON_BASE=' + base, '--load'])

        return lock['tags'][0]

    def captured_tool(self, stage):
        """Capture the independent tool and verify the bytes that this operation will execute."""
        (tool, checksum) = self.wheelhouse_artifacts()
        destination = stage / 'wheelhouse.pyz'

        shutil.copyfile(tool, destination)

        if digest(destination) != checksum:
            raise ValueError(f'Cannot use captured Wheelhouse {destination=!s}: its SHA-256 differs from the independent tool pin')

    def prepare(self, profile):
        """Acquire exact declared wheel inputs before metadata-only role locking."""
        image = self.bake(profile)['args']['PYTHON_BASE']

        self.require_image(image)

        with temporary_run_directory(ROOT, 'prepare') as stage:
            self.captured_tool(stage)
            shutil.copyfile(ROOT / 'builds' / profile / 'pyproject.toml', stage / 'pyproject.toml')

            project = tomllib.loads((stage / 'pyproject.toml').read_text())

            for (package, source) in project['tool']['wheelhouse'].get('sources', {}).items():
                if (not isinstance(source, dict)) or ('sha256' not in source):
                    continue

                self.wheelhouse_tool(
                    image,
                    stage,
                    'acquire',  # Prepare the declared exact wheel without dependency resolution.
                    '--project',  # Use the same captured source declaration as role locking.
                    '/stage/pyproject.toml',  # Let Wheelhouse verify wheel identity and real metadata.
                    '--package',  # Select one exact declaration before resolving the role.
                    package,  # Do not infer URLs or rewrite package-specific origins.
                    '--pool',  # Publish only verified immutable wheel bytes.
                    '/pool',  # Retain already verified unchanged wheels across later updates.
                    online=True,
                )

    def lock(self, profile):
        """Resolve one complete role and replace its generated lock only after success."""
        target = self.bake(profile)
        image = self.prepare_lock_image(target)
        directory = ROOT / 'builds' / profile

        with temporary_run_directory(ROOT, 'lock') as stage:
            self.captured_tool(stage)
            shutil.copyfile(directory / 'pyproject.toml', stage / 'pyproject.toml')

            scratch = stage / 'scratch'

            scratch.mkdir(mode=0o700)
            log.info('Resolving complete %s wheel selection', profile)

            self.wheelhouse_tool(
                image,
                stage,
                'lock',  # Resolve this role's complete installation environment.
                '--project',  # Read only the captured dependency declaration.
                '/stage/pyproject.toml',  # Keep input identity stable during resolution.
                '--lock',  # Let Wheelhouse atomically render its authoritative selection.
                '/scratch/pylock.toml',  # Keep resolver output in writable private scratch.
                '--pool',  # Prefer eligible already verified cached wheels.
                '/pool',  # Keep the pool read-only during metadata-only resolution.
                online=True,
                scratch=scratch,
            )

            resolved = json.loads((scratch / 'pylock.toml').read_text())
            resolved['project-sha256'] = digest(stage / 'pyproject.toml')

            with tempfile.NamedTemporaryFile(dir=directory, prefix='.pylock-', delete=False) as generated:
                temporary_lock = Path(generated.name)

                try:
                    generated.write((json.dumps(resolved, indent=2, sort_keys=True) + '\n').encode())

                    generated.flush()
                    os.fsync(generated.fileno())
                    temporary_lock.chmod(0o644)
                    os.replace(temporary_lock, directory / 'pylock.toml')
                finally:
                    temporary_lock.unlink(missing_ok=True)

        log.info('Saved complete dependency lock: %s', directory / 'pylock.toml')

    @contextmanager
    def selected(self, profile, acquire=False):
        """Capture one complete role lock, optionally acquire it, then verify it offline."""
        directory = ROOT / 'builds' / profile
        lock = directory / 'pylock.toml'

        if not lock.is_file():
            raise FileNotFoundError(f'Cannot use {profile=}: complete wheel {lock=!s} is missing; prepare the declared prerequisites and run make lock')

        with temporary_run_directory(ROOT, 'build') as stage:
            shutil.copyfile(directory / 'pyproject.toml', stage / 'pyproject.toml')
            shutil.copyfile(lock, stage / 'pylock.toml')
            selected = json.loads((stage / 'pylock.toml').read_text())

            if selected.get('project-sha256') != digest(stage / 'pyproject.toml'):
                raise ValueError(f'Cannot use {profile=}: {lock=!s} was not resolved from the current pyproject.toml; run make lock for this profile')

            target = self.bake(profile)
            base = target['args']['PYTHON_BASE']

            self.require_image(base)
            self.captured_tool(stage)

            if acquire:
                log.info('Acquiring complete locked %s wheel selection', profile)

                self.wheelhouse_tool(
                    base,
                    stage,
                    'acquire',  # Follow the complete locked selection without resolution.
                    '--lock',  # Read the single captured role lock.
                    '/stage/pylock.toml',  # Keep input identity stable through acquisition.
                    '--pool',  # Publish verified wheels to the selected shared pool.
                    '/pool',  # Preserve existing conflicting bytes instead of replacing them.
                    online=True,
                )

            log.info('Checking complete %s wheel selection offline', profile)

            checked = json.loads(
                self.wheelhouse_tool(
                    base,
                    stage,
                    'check',  # Verify every locked wheel for the actual target interpreter.
                    '--lock',  # Read the same captured role lock used for acquisition.
                    '/stage/pylock.toml',  # Never select a post-resolution package subset.
                    '--pool',  # Resolve only the selected cached wheel bytes.
                    '/pool',  # Keep the warm pool read-only.
                )
            )

            yield (stage, base, checked['artifact_entries'])

    def acquire(self, profile):
        """Acquire the exact complete role selection without dependency resolution."""

        with self.selected(profile, acquire=True):
            pass

        log.info('Dependencies ready for %s', profile)

    def build(self, profile):
        """Build offline from the complete checked wheel selection and its hashed projection."""

        with self.selected(profile) as (stage, base, artifacts):
            wheels = stage / 'wheels'

            wheels.mkdir()

            for artifact in artifacts:
                source = Path(artifact['source_path'])

                if source != (Path('/pool') / artifact['filename']):
                    raise ValueError(f'Cannot assemble the Bake wheel context: {source=!s} is outside the selected pool mount')

                destination = wheels / artifact['filename']
                source = self.pool / artifact['filename']

                try:
                    os.link(source, destination)
                except OSError as error:
                    if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.EROFS):
                        raise

                    # Read-only pools and different filesystems still export only selected bytes.
                    with source.open('rb') as reader, destination.open('xb') as writer:
                        shutil.copyfileobj(reader, writer, length=65536)

                if digest(destination) != artifact['sha256']:
                    raise ValueError(f'Cannot assemble the Bake wheel context: captured {destination=!s} differs from the selected SHA-256')

            destinations = {
                'artifacts': [
                    {
                        'source_path': artifact['source_path'],
                        'sha256': artifact['sha256'],
                        'destination': '/wheels/' + artifact['filename'],
                    }
                    for artifact in artifacts
                ],
            }

            (stage / 'destinations.json').write_text(json.dumps(destinations))

            scratch = stage / 'scratch'

            scratch.mkdir(mode=0o700)

            self.wheelhouse_tool(
                base,
                stage,
                'check',  # Verify the same full selection before publishing installation input.
                '--lock',  # Read the single captured complete role lock.
                '/stage/pylock.toml',  # Prevent dependency selection drift during this build.
                '--pool',  # Verify the bytes supplied through Bake's wheel context.
                '/pool',  # Keep pooled artifacts read-only.
                '--requirements',  # Let Wheelhouse render pip's hashed installation input.
                '/scratch/install.txt',  # Publish only after complete verification succeeds.
                '--destination-map',  # Map every exact source identity and digest.
                '/stage/destinations.json',  # Retain each wheel filename at its container path.
                scratch=scratch,
            )

            log.info('Building %s offline', profile)

            self.run(
                self.docker + [
                    'buildx',  # Build using the selected local engine's builder.
                    'bake',  # Preserve Chef's existing image composition owner.
                    '-f',  # Read the repository's declared image build graph.
                    'compose/images.yaml',  # Keep runtime source contexts unchanged.
                    profile,  # Build one complete installation role.
                    '--allow',  # Grant read access to the selected wheel context.
                    'fs.read=' + str(wheels),  # Export only this role's complete selected wheel set.
                    '--allow',  # Grant read access to this operation's installation output.
                    'fs.read=' + str(scratch),  # Keep disposable installation input outside the pool.
                    '--set',  # Supply the verified wheel pool as a named build context.
                    profile + '.contexts.wheels=' + str(wheels),  # Let pip reverify captured bytes at use.
                    '--set',  # Supply the atomically rendered hashed installation input.
                    profile + '.contexts.installation=' + str(scratch),  # Exclude host paths from pip input.
                    '--load',  # Publish a local image only after the build succeeds.
                ]
            )

        log.info('Build complete: %s', profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('action', choices=('wheelhouse/acquire', 'antlr/acquire', 'antlr/prepare', 'prepare', 'lock', 'acquire', 'build'))
    parser.add_argument('--profile', default='')
    parser.add_argument('--host', required=True, help='Resolved local Unix socket for the selected Docker context')
    parser.add_argument('--pool', required=True)
    parser.add_argument('--proxy')
    args = parser.parse_args()

    log.info('Starting %s%s', args.action, f' for {args.profile}' if args.profile else '')
    build = Build(args.host, args.pool, args.proxy)

    if args.action == 'wheelhouse/acquire':
        build.wheelhouse_artifacts(acquire=True)
    elif args.action == 'antlr/acquire':
        from prepare_antlr import acquire

        (tool, checksum) = build.wheelhouse_artifacts(acquire=True)

        acquire(build, tool, checksum)
    elif args.action == 'antlr/prepare':
        from prepare_antlr import prepare

        (tool, checksum) = build.wheelhouse_artifacts()

        prepare(build, tool, checksum)
    elif args.action == 'prepare':
        build.prepare(args.profile)
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
