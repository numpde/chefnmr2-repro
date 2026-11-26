"""Preparation retains one verified result and retires failed private work."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prepare_antlr


class Builder:
    """Substitute external tool/build results while exercising real host lifecycles."""

    def __init__(self, pool, failure=None):
        self.pool = pool
        self.failure = failure
        self.build_calls = 0

    def bake(self, profile):
        return {'args': {'PYTHON_BASE': 'pinned-test-image'}}

    def require_image(self, image):
        pass

    def container(self, image, mounts, scratch=None):
        self.scratch = scratch

        return ['container']

    def run(self, arguments, **options):
        if 'check' in arguments:
            if '--requirements' in arguments:
                (self.scratch / 'build-install.txt').write_text('')

            return SimpleNamespace(stdout=json.dumps({'artifact_entries': []}))

        self.build_calls += 1
        output = self.scratch / 'prepared'

        output.mkdir()
        (output / 'result.whl').write_bytes(b'verified wheel fixture')

        if self.failure is not None:
            raise self.failure

        (output / 'result.whl.metadata').write_bytes(b'verified metadata fixture')

        receipt = {
            key: {
                'filename': name,
                'sha256': hashlib.sha256((output / name).read_bytes()).hexdigest(),
            }
            for (key, name) in (
                ('wheel', 'result.whl'),
                ('metadata', 'result.whl.metadata'),
            )
        }

        (output / 'build.json').write_text(json.dumps(receipt))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()

        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pool = self.root / 'pool'

        self.pool.mkdir()

        directory = self.root / 'builds/antlr'

        directory.mkdir(parents=True)
        (self.pool / 'source.tar.gz').write_bytes(b'pinned source fixture')

        source = {
            'name': 'source.tar.gz',
            'url': 'https://example.invalid/source.tar.gz',
            'sha256': hashlib.sha256(b'pinned source fixture').hexdigest(),
        }

        (directory / 'source.json').write_text(json.dumps(source))
        (directory / 'pyproject.toml').write_text('backend declaration fixture')
        (directory / 'pylock.toml').write_text('complete backend lock fixture')
        (self.root / 'containers/antlr').mkdir(parents=True)
        (self.root / 'containers/antlr/prepare.py').write_text('recipe fixture')

        self.tool = self.root / 'wheelhouse.pyz'

        self.tool.write_bytes(b'verified tool fixture')

        self.checksum = hashlib.sha256(self.tool.read_bytes()).hexdigest()
        self.root_patch = patch.object(prepare_antlr, 'ROOT', self.root)

        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_repeat_reuses_one_verified_result(self):
        build = Builder(self.pool)
        first = prepare_antlr.prepare(build, self.tool, self.checksum)
        second = prepare_antlr.prepare(build, self.tool, self.checksum)

        self.assertEqual(second, first)
        self.assertEqual(build.build_calls, 1)
        self.assertEqual(list((self.root / 'tmp/prepared/antlr').iterdir()), [first])
        self.assertEqual(list((self.root / 'tmp/antlr').iterdir()), [])

    def test_changed_tool_prepares_a_distinct_result(self):
        build = Builder(self.pool)
        first = prepare_antlr.prepare(build, self.tool, self.checksum)

        self.tool.write_bytes(b'updated verified tool fixture')

        checksum = hashlib.sha256(self.tool.read_bytes()).hexdigest()
        second = prepare_antlr.prepare(build, self.tool, checksum)

        self.assertNotEqual(second, first)
        self.assertEqual(build.build_calls, 2)
        self.assertEqual(json.loads((second / 'build.json').read_text())['wheelhouse_sha256'], checksum)
        self.assertEqual(list((self.root / 'tmp/antlr').iterdir()), [])

    def test_corruption_refuses_replacement(self):
        build = Builder(self.pool)
        result = prepare_antlr.prepare(build, self.tool, self.checksum)

        (result / 'result.whl').write_bytes(b'changed retained wheel')

        with self.assertRaisesRegex(ValueError, 'different SHA-256'):
            prepare_antlr.prepare(build, self.tool, self.checksum)

        self.assertEqual(build.build_calls, 1)
        self.assertEqual((result / 'result.whl').read_bytes(), b'changed retained wheel')
        self.assertEqual(list((self.root / 'tmp/antlr').iterdir()), [])

    def test_failure_and_handled_interruption_retire_partial_work(self):
        for failure in (RuntimeError('builder failed'), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                build = Builder(self.pool, failure)

                with self.assertRaises(type(failure)):
                    prepare_antlr.prepare(build, self.tool, self.checksum)

                self.assertEqual(list((self.root / 'tmp/prepared/antlr').iterdir()), [])
                self.assertEqual(list((self.root / 'tmp/antlr').iterdir()), [])
