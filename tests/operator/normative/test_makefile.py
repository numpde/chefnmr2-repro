"""Normative Make workflow behavior with no real Docker engine."""

from operator_fixtures import OperatorFixture, ROOT


class MakefileNormativeTests(OperatorFixture):
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

    def test_weights_workflows_use_expected_services_and_exclude_parallel_writers(self):
        result = self.run_make('weights/download', 'PROXY_URL=none', 'DOCKER_CONTEXT=local')
        self.assertEqual(result.returncode, 0, result.stderr)
        compose = [call for call in self.calls() if call[0] == '--host']
        self.assertEqual(len(compose), 2)
        self.assertEqual([call[-1] for call in compose], ['volume-init', 'download'])
        self.assertIn('--env-file', compose[0])
        self.assertEqual(self.proxy_log.read_text().splitlines(), ['unset', 'none', 'none'])

        self.docker_log.unlink()
        sanitize = self.run_make('weights/sanitize', 'DOCKER_CONTEXT=local')
        self.assertEqual(sanitize.returncode, 0, sanitize.stderr)
        compose = [call for call in self.calls() if call[0] == '--host']
        self.assertEqual(len(compose), 1)
        self.assertEqual(compose[0][-1], 'sanitize')
        self.assertIn('chefnmr2-weights-writer', compose[0])
