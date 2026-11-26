"""Role locks retain the declaration that selected their complete wheel set."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import build


class RoleLockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()

        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / 'builds/provider'

        self.directory.mkdir(parents=True)

        self.project = self.directory / 'pyproject.toml'

        self.project.write_text('[project]\ndependencies = ["provider-core==0.7.7"]\n')

        self.lock = self.directory / 'pylock.toml'
        self.checksum = build.digest(self.project)

        self.lock.write_text(json.dumps({'project-sha256': self.checksum}))

        self.builder = build.Build.__new__(build.Build)
        self.builder.bake = Mock(return_value={'args': {'PYTHON_BASE': 'pinned-base'}})
        self.builder.require_image = Mock()
        self.builder.captured_tool = Mock()
        self.builder.wheelhouse_tool = Mock(return_value='{"artifact_entries": []}')
        self.root_patch = patch.object(build, 'ROOT', self.root)

        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_changed_dependency_or_source_pin_blocks_acquisition_and_build(self):
        for change in ('dependencies = ["provider-core==0.7.8"]', '[tool.wheelhouse.sources]\nprovider-core = {sha256 = "changed"}'):
            for operation in (self.builder.acquire, self.builder.build):
                with self.subTest(change=change, operation=operation.__name__):
                    self.project.write_text(change + '\n')

                    with self.assertRaisesRegex(ValueError, 'current pyproject.toml; run make lock'):
                        operation('provider')

                    self.builder.bake.assert_not_called()
                    self.builder.require_image.assert_not_called()
                    self.builder.captured_tool.assert_not_called()
                    self.builder.wheelhouse_tool.assert_not_called()

    def test_unchanged_declaration_allows_acquisition_and_offline_check(self):
        self.builder.acquire('provider')

        self.assertEqual([call.args[2] for call in self.builder.wheelhouse_tool.call_args_list], ['acquire', 'check'])
        self.assertTrue(self.builder.wheelhouse_tool.call_args_list[0].kwargs['online'])

    def test_lock_publication_identifies_captured_producer_declaration(self):
        self.builder.prepare_lock_image = Mock(return_value='resolver-image')

        def resolve(image, stage, *arguments, **options):
            self.project.write_text('declaration changed during resolution\n')
            (options['scratch'] / 'pylock.toml').write_text('{"wheelhouse-lock": 1}')

        self.builder.wheelhouse_tool.side_effect = resolve
        self.builder.lock('provider')
        published = json.loads(self.lock.read_text())

        self.assertEqual(published['project-sha256'], self.checksum)
        self.assertEqual(published['wheelhouse-lock'], 1)

        with self.assertRaisesRegex(ValueError, 'current pyproject.toml'):
            self.builder.acquire('provider')
