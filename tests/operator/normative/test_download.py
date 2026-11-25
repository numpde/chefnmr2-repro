"""Exercise the shared checkpoint downloader against a local curl substitute."""

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[3] / 'containers/checkpoint-download/download.sh'

class DownloadTests(unittest.TestCase):
    def test_flat_and_bundled_checkpoints(self):
        for paths, receipt in ((['model.pth'], 'model.pth.sanitized.json'), (['epoch/config.yaml', 'epoch/model.pt'], 'model.pt.sanitized.json')):
            with self.subTest(paths=paths), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                volume, source, bin_dir = (root / part for part in ('volume', 'source', 'bin'))
                for directory in (volume, source, bin_dir):
                    directory.mkdir()
                curl = bin_dir / 'curl'
                curl.write_text('''#!/bin/sh
set -eu
while [ "$1" != --output ]; do shift; done
cp "$SOURCE/${3#https://example.invalid/}" "$2"
''')
                curl.chmod(0o755)
                manifest = root / 'manifest'
                rows = []
                for path in paths:
                    original = source / path
                    original.parent.mkdir(parents=True, exist_ok=True)
                    original.write_bytes(path.encode())
                    local = volume / path
                    local.parent.mkdir(parents=True, exist_ok=True)
                    local.write_bytes(b'outdated')
                    rows.append(f'{hashlib.sha256(original.read_bytes()).hexdigest()} 100 {path}\n')
                marker = (volume / paths[0]).parent / receipt
                marker.write_text('old receipt')
                manifest.write_text(''.join(rows))
                environment = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', SOURCE=str(source), HTTPS_PROXY='http://example.invalid:8080', CHECKPOINT_RELEASE_URL='https://example.invalid', CHECKPOINT_RECEIPT_FILE=receipt)
                def run():
                    return subprocess.run(['sh', '-eu', str(SCRIPT), str(manifest)], cwd=volume, env=environment, capture_output=True, text=True)
                try:
                    result = run()
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertFalse(marker.exists())
                    for path in paths:
                        local = volume / path
                        self.assertEqual(local.read_bytes(), (source / path).read_bytes())
                        self.assertEqual(local.stat().st_mode & 0o777, 0o444)
                    marker.parent.chmod(0o755)
                    marker.write_text('current receipt')
                    marker.parent.chmod(0o555)
                    result = run()
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(marker.read_text(), 'current receipt')
                    self.assertNotIn('Downloading', result.stdout)

                    # A corrupt transfer must not replace a verified file or invalidate its receipt.
                    manifest.write_text('0' * 64 + ' 100 ' + paths[0] + '\n')
                    result = run()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual((volume / paths[0]).read_bytes(), (source / paths[0]).read_bytes())
                    self.assertEqual(marker.read_text(), 'current receipt')
                finally:
                    volume.chmod(0o755)
                    for directory in volume.rglob('*'):
                        if directory.is_dir():
                            directory.chmod(0o755)

