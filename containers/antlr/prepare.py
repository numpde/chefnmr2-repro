"""Build the pinned ANTLR source with an already checked offline backend."""

from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv
import zipfile


def run(*arguments):
    subprocess.run(arguments, check=True)


def main():
    os.environ.update(SOURCE_DATE_EPOCH='315532800', PYTHONHASHSEED='0')

    environment = Path('/tmp/antlr-build')

    venv.create(environment, with_pip=True, symlinks=True)

    python = str(environment / 'bin/python')

    pip = [
        python,  # Use the private environment's pinned interpreter.
        '-m',  # Invoke pip as a module under that interpreter.
        'pip',  # Use pip supplied by the pinned base's ensurepip bundle.
        '--disable-pip-version-check',  # Avoid pip's online update check.
        '--no-cache-dir',  # Keep preparation independent of pip's cache.
    ]

    run(
        *pip,  # Reuse the private installer and its offline preparation settings.
        'install',  # Install the complete checked backend wheel selection.
        '--no-index',  # Refuse package index access.
        '--no-deps',  # Use only the lock's complete dependency selection.
        '--require-hashes',  # Reverify the selected wheel bytes at installation.
        '-r',  # Read Wheelhouse's hashed installation projection.
        '/inputs/build-install.txt',  # Use paths into the captured backend wheels.
    )

    output = Path('/scratch/prepared')

    output.mkdir()

    command = [
        *pip,  # Reuse the private installer and its offline preparation settings.
        'wheel',  # Prepare the separately retained ANTLR runtime wheel.
        '--no-index',  # Refuse package index access during source preparation.
        '--no-deps',  # Build only the explicitly pinned source archive.
        '--no-build-isolation',  # Use the installed locked backend without downloads.
        '/inputs/antlr4-python3-runtime-4.9.3.tar.gz',  # Read the verified source snapshot.
        '--wheel-dir',  # Retain the output wheel in this run's private scratch.
        str(output),  # Let the host publish the complete result after success.
    ]

    run(*command)

    wheels = list(output.glob('*.whl'))

    if len(wheels) != 1:
        raise ValueError(f'Cannot retain ANTLR preparation: {len(wheels)=} instead of one output wheel')

    wheel = wheels[0]

    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]

        if len(members) != 1:
            raise ValueError(f'Cannot retain {wheel.name=}: its wheel contains {len(members)=} METADATA records')

        metadata = archive.read(members[0])

    parsed = BytesParser().parsebytes(metadata)

    if (parsed['Name'], parsed['Version']) != ('antlr4-python3-runtime', '4.9.3'):
        raise ValueError(f'Cannot retain {wheel.name=}: its package identity is {(parsed["Name"], parsed["Version"])=!r}')

    wheel.with_name(wheel.name + '.metadata').write_bytes(metadata)

    details = {
        'python': platform.python_version(),
        'pip': (
            subprocess
            .check_output(
                [
                    *pip,  # Query the same installer used to produce this wheel.
                    '--version',  # Record the installer version in the build receipt.
                ],
                text=True,
            )
            .strip()
        ),
        'command': command,
        'environment': {
            'SOURCE_DATE_EPOCH': os.environ['SOURCE_DATE_EPOCH'],
            'PYTHONHASHSEED': os.environ['PYTHONHASHSEED'],
        },
        'wheel': {
            'filename': wheel.name,
            'sha256': hashlib.file_digest(wheel.open('rb'), 'sha256').hexdigest(),
        },
        'metadata': {
            'filename': wheel.name + '.metadata',
            'sha256': hashlib.sha256(metadata).hexdigest(),
        },
    }

    (output / 'build.json').write_text(json.dumps(details, indent=2) + '\n')


if __name__ == '__main__':
    main()
