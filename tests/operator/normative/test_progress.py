"""Progress reaches the terminal before completion without contaminating JSON."""

import hashlib
import io
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fetch_sdk import fetch


class ProgressTests(unittest.TestCase):
    def test_tool_stderr_streams_while_stdout_is_captured(self):
        program = '''
import os
from pathlib import Path
import sys
from build import Build
build = Build.__new__(Build)
build.pool = Path('/unused')
build.tool = Path('tool.pyz')
build.environment = os.environ.copy()
build.container = lambda *args: [sys.executable, '-u', '-c',
    'import json, sys; print("tool working", file=sys.stderr); sys.stdin.readline(); print(json.dumps(dict(ready=True)))']
sys.stdout.write(build.sdk_tool('unused-image', Path('/unused-stage'), 'select'))
'''
        with subprocess.Popen([sys.executable, '-B', '-c', program],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) as process:
            try:
                with selectors.DefaultSelector() as ready:
                    ready.register(process.stderr, selectors.EVENT_READ)
                    self.assertTrue(ready.select(timeout=5), 'Progress was hidden until tool completion')
                self.assertEqual(process.stderr.readline(), 'tool working\n')
            finally:
                stdout, stderr = process.communicate('\n', timeout=5)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertEqual(stdout, '{"ready": true}\n')

    def test_sdk_reports_verified_downloads_and_cached_files_only(self):
        payload = b'example SDK artifact'
        checksum = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory)
            with patch('fetch_sdk.urlopen', return_value=io.BytesIO(payload)), self.assertLogs('fetch_sdk') as messages:
                fetch(pool, 'https://example.invalid/', 'tool.pyz', checksum)
            self.assertIn('Downloading tool.pyz', '\n'.join(messages.output))
            self.assertIn('Downloaded and verified tool.pyz', '\n'.join(messages.output))
            with patch('fetch_sdk.urlopen', side_effect=AssertionError('Cached files need no download')), self.assertLogs('fetch_sdk') as messages:
                fetch(pool, 'https://example.invalid/', 'tool.pyz', checksum)
            self.assertIn('Verified cached tool.pyz', '\n'.join(messages.output))
            with patch('fetch_sdk.urlopen', return_value=io.BytesIO(payload)), self.assertLogs('fetch_sdk') as messages:
                with self.assertRaises(ValueError):
                    fetch(pool, 'https://example.invalid/', 'bad.pyz', '0' * 64)
            self.assertNotIn('Downloaded and verified', '\n'.join(messages.output))
            self.assertFalse((pool / 'bad.pyz').exists())
