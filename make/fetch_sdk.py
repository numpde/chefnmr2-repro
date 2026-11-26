"""Acquire the selected SDK's small bootstrap artifacts inside the shared worker."""

import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen

log = logging.getLogger(__name__)


def artifact_names(version):
    return (
        f'wheelhouse-tool-{version}.pyz',
        f'provider_core-{version}-native-install.lock',
        f'provider_core-{version}-install.lock',
        f'provider_core-{version}-py3-none-any.whl',
    )


def release_url(version):
    return f'https://github.com/numpde/nmr-provider-tools/releases/download/v{version}/'


def fetch(pool, release, name, checksum, *, url=None):
    destination = pool / name

    if destination.exists():
        log.info('Checking cached %s', name)

        if hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Cannot use cached artifact: {name=} has a different SHA-256")

        log.info('Verified cached %s', name)

        return

    with tempfile.NamedTemporaryFile(dir=pool, prefix='.artifact-', delete=False) as output:
        temporary = Path(output.name)

        try:
            log.info('Downloading %s', name)

            with urlopen(release + name if url is None else url, timeout=120) as response:
                total = 0

                while block := response.read(65536):
                    total += len(block)

                    if total > (8 * 1024 * 1024):
                        raise ValueError(f"Cannot acquire artifact: {name=} exceeds 8 MiB")

                    output.write(block)

            output.close()

            if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Cannot publish downloaded artifact: {name=} has a different SHA-256")

            temporary.chmod(0o644)
            temporary.replace(destination)
            log.info('Downloaded and verified %s (%d bytes)', name, total)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    if sys.argv[1:2] == ['--artifact']:
        (_, declaration, directory) = sys.argv[1:]
        artifact = json.loads(Path(declaration).read_text())

        fetch(Path(directory), '', artifact['name'], artifact['sha256'], url=artifact['url'])

        return

    (version, manifest_hash, directory) = sys.argv[1:]
    pool = Path(directory)
    release = release_url(version)
    manifest = f'provider_core-{version}-sha256.json'

    fetch(pool, release, manifest, manifest_hash)
    hashes = json.loads((pool / manifest).read_text())

    for name in artifact_names(version):
        fetch(pool, release, name, hashes[name])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
