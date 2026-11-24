"""Scratch cleanup preserves other active runs and retained recovery records."""

from pathlib import Path
import tempfile
import unittest

from work import create_run_directory, temporary_run_directory


class ScratchLifetimeTests(unittest.TestCase):
    def test_success_removes_only_the_finished_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = create_run_directory(root, 'rollback')

            (record / 'provider.toml').write_text('retained configuration')

            with temporary_run_directory(root, 'build') as first:
                (first / 'install.txt').write_text('first build')

                with temporary_run_directory(root, 'build') as second:
                    (second / 'install.txt').write_text('second build')

                self.assertFalse(second.exists())
                self.assertEqual((first / 'install.txt').read_text(), 'first build')

            self.assertFalse(first.exists())
            self.assertEqual((record / 'provider.toml').read_text(), 'retained configuration')

    def test_exception_and_keyboard_interrupt_remove_nested_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            for exception in (RuntimeError, KeyboardInterrupt):
                with self.subTest(exception=exception):
                    failure = exception('operation interrupted')

                    with self.assertRaises(exception) as caught:
                        with temporary_run_directory(Path(temporary), 'lock') as stage:
                            (stage / 'scratch').mkdir()
                            (stage / 'scratch' / 'resolver.txt').write_text('unfinished')
                            raise failure

                    self.assertIs(caught.exception, failure)
                    self.assertFalse(stage.exists())
