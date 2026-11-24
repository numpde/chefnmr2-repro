"""Normative Make workflow behavior with no real Docker engine."""

import shlex
import subprocess

from operator_fixtures import OperatorFixture, ROOT


class MakefileNormativeTests(OperatorFixture):
    def test_image_pull_uses_native_compose_without_building(self):
        result = self.run_make('images/pull', 'DOCKER_CONTEXT=local', cwd='/tmp')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[-1], [
            '--host', 'unix:///tmp/fake-docker.sock', 'compose', '--project-directory', str(ROOT),
            '--env-file', '/dev/null', '-f', str(ROOT / 'compose/images.yaml'),
            'pull', '--ignore-buildable',
        ])
        self.assertFalse(self.python_log.exists())

    def test_context_menu_selects_a_discovered_context_by_number(self):
        command = shlex.join(['make', '-f', str(ROOT / 'Makefile'), 'check'])
        result = subprocess.run(
            ['script', '--quiet', '--return', '--command', command, '/dev/null'],
            input='2\n', cwd=self.directory, env=self.environment(),
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('1) default', result.stdout)
        self.assertIn('2) science', result.stdout)
        self.assertEqual(self.calls(), [
            ['context', 'ls', '--format', '{{.Name}}'],
            ['context', 'inspect', 'science', '--format', '{{.Endpoints.docker.Host}}'],
        ])
        self.assertEqual(self.python_arguments()[2:5], ['check', '--host', 'unix:///tmp/fake-docker.sock'])

    def test_help_and_dry_run_need_no_selection_or_engine(self):
        help_result = self.run_make('help')
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn('make weights/sanitize', help_result.stdout)
        self.assertIn('Sanitize known checkpoint files', help_result.stdout)
        dry_run = self.run_make('-n', 'weights/verify', 'DOCKER_CONTEXT=local')
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertFalse(self.docker_log.exists())
        self.assertFalse(self.python_log.exists())

    def test_selected_local_socket_reaches_runtime_from_another_directory(self):
        result = self.run_make('check', 'DOCKER_CONTEXT=local', cwd='/tmp')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [[
            'context', 'inspect', 'local', '--format', '{{.Endpoints.docker.Host}}',
        ]])
        self.assertEqual(self.docker_environment_log.read_text().strip(), 'unset|unset|unset|unset')
        arguments = self.python_arguments()
        self.assertEqual(arguments[1], str(ROOT / 'make/runtime.py'))
        self.assertEqual(arguments[2:5], ['check', '--host', 'unix:///tmp/fake-docker.sock'])
        self.assertEqual(arguments[5:7], ['--category', 'all'])

    def test_weights_selects_acquisition_route_and_keeps_preparation_offline(self):
        result = self.run_make('weights/download', 'PROXY_URL=none', 'DOCKER_CONTEXT=local')
        self.assertEqual(result.returncode, 0, result.stderr)
        compose = [call for call in self.calls() if call[0] == '--host']
        self.assertEqual(len(compose), 1)
        self.assertEqual(compose[0][-1], 'volume-init')
        for call in compose:
            self.assertEqual(call[call.index('--pull') + 1], 'missing')
        self.assertIn('--env-file', compose[0])
        arguments = self.python_arguments()
        self.assertEqual(arguments[1], str(ROOT / 'make/weights.py'))
        self.assertEqual(arguments[-2:], ['--proxy', 'none'])

        self.docker_log.unlink()
        sanitize = self.run_make('weights/sanitize', 'DOCKER_CONTEXT=local')
        self.assertEqual(sanitize.returncode, 0, sanitize.stderr)
        compose = [call for call in self.calls() if call[0] == '--host']
        self.assertEqual(len(compose), 1)
        self.assertEqual(compose[0][-1], 'sanitize')
        self.assertIn('chefnmr2-weights-writer', compose[0])
        self.assertEqual(compose[0][compose[0].index('--pull') + 1], 'never')

        self.docker_log.unlink()
        verify = self.run_make('weights/verify', 'DOCKER_CONTEXT=local')
        self.assertEqual(verify.returncode, 0, verify.stderr)
        compose = [call for call in self.calls() if call[0] == '--host']
        self.assertEqual(len(compose), 1)
        self.assertEqual(compose[0][-1], 'verify')
        self.assertEqual(compose[0][compose[0].index('--pull') + 1], 'never')
