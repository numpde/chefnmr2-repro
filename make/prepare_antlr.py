"""Prepare ANTLR once offline and retain a private wheel, metadata and inputs."""

import errno
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess

from build import ROOT, digest
from work import temporary_run_directory

log = logging.getLogger(__name__)


def captured(source, destination, checksum):
    with source.open('rb') as reader, destination.open('xb') as writer:
        shutil.copyfileobj(reader, writer, length=65536)

    if digest(destination) != checksum:
        raise ValueError(f'Cannot prepare ANTLR: {source=!s} does not match {checksum=!s}')


def acquire(build, tool, tool_sha256):
    image = build.bake('worker-cpu')['args']['PYTHON_BASE']

    build.require_image(image)

    build.run(
        build.acquisition_command(
            image,
            [
                (build.pool, '/pool'),
                (ROOT / 'builds/antlr/source.json', '/source.json'),
                (ROOT / 'make/fetch_artifact.py', '/fetch_artifact.py'),
            ],
            [
                '-B',  # Keep bytecode out of the captured source mounts.
                '/fetch_artifact.py',  # Use the shared exact artifact fetch owner.
                '/source.json',  # Read the source archive URL, name and digest.
                '/pool',  # Publish verified archive bytes in the shared pool.
            ],
            writable_pool=True,
        )
    )

    with temporary_run_directory(ROOT, 'antlr-acquire') as stage:
        captured(tool, stage / 'wheelhouse.pyz', tool_sha256)
        captured(ROOT / 'builds/antlr/pylock.toml', stage / 'pylock.toml', digest(ROOT / 'builds/antlr/pylock.toml'))

        build.run(
            build.acquisition_command(
                image,
                [(build.pool, '/pool'), (stage, '/stage')],
                [
                    '-B',  # Keep bytecode out of the captured tool mount.
                    '/stage/wheelhouse.pyz',  # Run the independently verified tool.
                    'acquire',  # Acquire the complete locked backend closure.
                    '--lock',  # Follow the committed backend wheel selection.
                    '/stage/pylock.toml',  # Resolve no packages during acquisition.
                    '--pool',  # Publish verified backend wheels in the shared pool.
                    '/pool',  # Use the same pool as the source archive acquisition.
                ],
                writable_pool=True,
            )
        )


def verify_retained(destination, identity, build_inputs):
    """Reuse only a complete result whose recorded inputs and retained bytes agree."""
    receipt = json.loads((destination / 'build.json').read_text())

    if (receipt.get('build_identity') != identity) or (receipt.get('build_inputs') != build_inputs):
        raise ValueError(f'Cannot reuse prepared ANTLR: {destination=!s} records different build inputs')

    for record in (receipt['wheel'], receipt['metadata']):
        name = record['filename']

        if Path(name).name != name:
            raise ValueError(f'Cannot reuse prepared ANTLR: {name=} is not a retained artifact filename')

        if digest(destination / name) != record['sha256']:
            raise ValueError(f'Cannot reuse prepared ANTLR: retained {name=} has a different SHA-256')

    for (name, checksum) in receipt['inputs'].items():
        if Path(name).name != name:
            raise ValueError(f'Cannot reuse prepared ANTLR: {name=} is not a retained input filename')

        if digest(destination / 'inputs' / name) != checksum:
            raise ValueError(f'Cannot reuse prepared ANTLR: retained input {name=} has a different SHA-256')

    return receipt


def prepare(build, tool, tool_sha256):
    """The caller verifies the independent tool pin before invoking this lane."""
    directory = ROOT / 'builds/antlr'
    image = build.bake('worker-cpu')['args']['PYTHON_BASE']

    build.require_image(image)

    source = json.loads((directory / 'source.json').read_text())
    parent = ROOT / 'tmp/prepared/antlr'

    parent.mkdir(parents=True, exist_ok=True)

    with temporary_run_directory(ROOT, 'antlr') as stage:
        inputs = stage / 'inputs'

        inputs.mkdir()
        captured(build.pool / source['name'], inputs / source['name'], source['sha256'])

        for path in (
            directory / 'source.json',
            directory / 'pyproject.toml',
            directory / 'pylock.toml',
            ROOT / 'containers/antlr/prepare.py',
        ):
            captured(path, inputs / path.name, digest(path))

        captured(tool, inputs / 'wheelhouse.pyz', tool_sha256)

        build_inputs = {
            'source_sha256': source['sha256'],
            'source_declaration_sha256': digest(inputs / 'source.json'),
            'backend_lock_sha256': digest(inputs / 'pylock.toml'),
            'backend_project_sha256': digest(inputs / 'pyproject.toml'),
            'recipe_sha256': digest(inputs / 'prepare.py'),
            'wheelhouse_sha256': tool_sha256,
            'image': image,
        }

        identity = hashlib.sha256(json.dumps(build_inputs, sort_keys=True).encode()).hexdigest()
        destination = parent / ('4.9.3-' + identity)

        if destination.exists():
            verify_retained(destination, identity, build_inputs)
            log.info('Reusing verified prepared ANTLR: %s', destination)

            return destination

        scratch = stage / 'scratch'

        scratch.mkdir()

        mounts = [(build.pool, '/pool'), (inputs, '/inputs')]

        def check(*arguments):
            result = build.run(
                build.container(image, mounts, scratch=scratch) + [
                    '-B',  # Keep bytecode out of the captured source mounts.
                    '/inputs/wheelhouse.pyz',  # Run the independently verified tool.
                    'check',  # Verify the complete backend wheel closure offline.
                    '--lock',  # Select the captured build dependency lock.
                    '/inputs/pylock.toml',  # Resolve no packages during preparation.
                    '--pool',  # Read backend wheels from the shared pool.
                    '/pool',  # Keep the pool read-only throughout preparation.
                    *arguments,  # Add the requested disposable installation projection.
                ],
                stdout=subprocess.PIPE,
                text=True,
            )

            return json.loads(result.stdout)

        entries = check()['artifact_entries']

        destinations = {
            'artifacts': [
                {
                    'source_path': entry['source_path'],
                    'sha256': entry['sha256'],
                    'destination': '/inputs/backend/' + entry['filename'],
                }
                for entry in entries
            ],
        }

        (inputs / 'destinations.json').write_text(json.dumps(destinations))

        # /inputs is read-only; check publishes its projection through scratch.
        check(
            '--requirements',  # Let Wheelhouse render pip's hashed installation input.
            '/scratch/build-install.txt',  # Publish only in this run's writable scratch.
            '--destination-map',  # Map every selected wheel into the captured backend.
            '/inputs/destinations.json',  # Keep installation paths independent of the host.
        )

        shutil.copyfile(scratch / 'build-install.txt', inputs / 'build-install.txt')

        backend = inputs / 'backend'

        backend.mkdir()

        for entry in entries:
            origin = Path(entry['source_path'])

            if not origin.is_relative_to('/pool'):
                raise ValueError(f'Cannot prepare ANTLR: its build lock selects a private wheel outside the shared pool: {origin=!s}')

            captured(build.pool / origin.relative_to('/pool'), backend / entry['filename'], entry['sha256'])

        log.info('Building ANTLR offline with %s', image)

        build.run(
            build.container(image, [(inputs, '/inputs')], scratch=scratch) + [
                '-B',  # Keep bytecode out of the captured recipe mount.
                '/inputs/prepare.py',  # Build the pinned source without networking.
            ]
        )

        output = scratch / 'prepared'
        receipt = json.loads((output / 'build.json').read_text())

        receipt.update(
            image=image,
            source=source,
            wheelhouse_sha256=tool_sha256,
            build_identity=identity,
            build_inputs=build_inputs,
        )

        retained_names = (
            'source.json',
            'pyproject.toml',
            'pylock.toml',
            'prepare.py',
            source['name'],
            'build-install.txt',
        )

        receipt['inputs'] = {name: digest(inputs / name) for name in retained_names}

        (output / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')

        retained = output / 'inputs'

        retained.mkdir()

        for name in retained_names:
            shutil.copyfile(inputs / name, retained / name)

        # Publish a complete build identity once; never replace existing prepared bytes.
        try:
            output.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise

            existing = verify_retained(destination, identity, build_inputs)

            if existing['wheel']['sha256'] != receipt['wheel']['sha256']:
                raise ValueError(f'Cannot retain prepared ANTLR: {destination=!s} already contains different wheel bytes for these build inputs') from error

        log.info('Prepared ANTLR wheel, metadata and receipt: %s', destination)

        return destination
