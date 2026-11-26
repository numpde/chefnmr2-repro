"""Prepare ANTLR once offline and retain a private wheel, metadata and inputs."""

import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile

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
                (ROOT / 'make/fetch_sdk.py', '/fetch_sdk.py'),
                (ROOT / 'make/fetch_antlr.py', '/fetch_antlr.py'),
            ],
            [
                '-B',  # Keep bytecode out of the captured source mounts.
                '/fetch_antlr.py',  # Run Chef's exact source acquisition recipe.
                '/pool',  # Publish verified archive bytes in the shared pool.
                '/source.json',  # Read the committed archive identity.
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

        receipt.update(image=image, source=source, wheelhouse_sha256=digest(inputs / 'wheelhouse.pyz'))

        receipt['inputs'] = {
            path.name: digest(path)
            for path in inputs.iterdir()
            if path.is_file() and (path.name != 'wheelhouse.pyz')
        }

        (output / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')

        retained = output / 'inputs'

        retained.mkdir()

        for name in (
            'source.json',
            'pyproject.toml',
            'pylock.toml',
            'prepare.py',
            source['name'],
            'build-install.txt',
        ):
            shutil.copyfile(inputs / name, retained / name)

        # Keep successful outputs private; publication and pool admission are separate operations.
        destination = Path(tempfile.mkdtemp(dir=parent, prefix='candidate-'))

        destination.rmdir()
        output.rename(destination)
        log.info('Prepared ANTLR wheel, metadata and receipt: %s', destination)

        return destination
