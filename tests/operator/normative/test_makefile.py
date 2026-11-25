"""Normative Make workflow behavior with no real Docker engine."""

import os
import shlex
import subprocess

from operator_fixtures import OperatorFixture, ROOT


class MakefileNormativeTests(OperatorFixture):
    def test_prompt_history_recalls_literal_values_and_accepts_new_ones(self):
        literal = '/cache with spaces/$(touch UNEXPECTED)'

        for value in ('/cache/old', literal, '/cache/third', '/cache/fourth', '/cache/fifth', '/cache/sixth'):
            result = self.run_choose('WHEELHOUSE', value)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        result = self.run_choose('WHEELHOUSE', '5')

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('5) ' + literal, result.stdout)
        self.assertEqual((self.directory / 'selected').read_text(), literal + '\n')
        history = self.directory / 'repo/tmp/inputs/WHEELHOUSE'

        self.assertEqual(history.read_text().splitlines(), [literal, '/cache/sixth', '/cache/fifth', '/cache/fourth', '/cache/third'])
        self.assertEqual(history.stat().st_mode & 0o777, 0o600)
        self.assertFalse((self.directory / 'UNEXPECTED').exists())

    def test_image_pull_uses_native_compose_without_building(self):
        result = self.run_make('images/pull', 'DOCKER_CONTEXT=local', 'IMAGE_NAMESPACE=foreign', cwd='/tmp', IMAGE_NAMESPACE='ambient')

        self.assertEqual(result.returncode, 0, result.stderr)

        self.assertEqual(
            self.calls()[-1],
            [
                '--host',
                'unix:///tmp/fake-docker.sock',
                'compose',
                '--project-directory',
                str(ROOT),
                '--env-file',
                '/dev/null',
                '-f',
                str(ROOT / 'compose/images.yaml'),
                'pull',
                '--ignore-buildable',
            ]
        )

        self.assertFalse(self.python_log.exists())
        self.assertEqual(self.docker_environment_log.read_text().splitlines()[-1].split('|')[-1], os.environ['IMAGE_NAMESPACE'])

    def test_unit_suites_use_compose_on_the_selected_engine_from_another_directory(self):
        for category in ('normative', 'adversarial'):
            with self.subTest(category=category):
                self.docker_log.unlink(missing_ok=True)
                result = self.run_make('test/unit/' + category, 'DOCKER_CONTEXT=science', cwd='/tmp')

                self.assertEqual(result.returncode, 0, result.stderr)
                calls = self.calls()
                selections = [call for call in calls if call[:2] == ['context', 'inspect']]
                self.assertEqual(len(selections), 1)
                self.assertEqual(selections[0][2], 'science')
                self.assertEqual(len(calls), 3)

                for command, service, pattern in zip(
                    calls[1:], ('checkpoints', 'binding'), ('test_checkpoints.py', 'test_binding.py'), strict=True
                ):
                    self.assertEqual(command[:3], ['--host', 'unix:///tmp/fake-docker.sock', 'compose'])
                    self.assertEqual(command[command.index('--project-directory') + 1], str(ROOT))
                    self.assertEqual(command[command.index('--env-file') + 1], '/dev/null')
                    self.assertEqual(command[command.index('-f') + 1], str(ROOT / 'compose/test-unit.yaml'))
                    self.assertIn('run', command)
                    self.assertEqual(command[command.index('--pull') + 1], 'never')
                    for option in ('--rm', '--no-deps', '--interactive=false', '-T'):
                        self.assertIn(option, command)
                    self.assertEqual(
                        command[command.index(service) + 1:],
                        ['-B', '-m', 'unittest', 'discover', '-s', '/tests/unit/' + category, '-p', pattern, '-v'],
                    )

    def test_context_menu_selects_a_discovered_context_by_number(self):
        command = shlex.join(['make', '-f', str(ROOT / 'Makefile'), 'check'])

        result = subprocess.run(
            ['script', '--quiet', '--return', '--command', command, '/dev/null'],
            input='2\n',
            cwd=self.directory,
            env=self.environment(),
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('1) default', result.stdout)
        self.assertIn('2) science', result.stdout)

        self.assertEqual(
            self.calls(),
            [
                ['context', 'ls', '--format', '{{.Name}}'],
                ['context', 'inspect', 'science', '--format', '{{.Endpoints.docker.Host}}'],
            ]
        )

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

        self.assertEqual(
            self.calls(),
            [
                [
                    'context',
                    'inspect',
                    'local',
                    '--format',
                    '{{.Endpoints.docker.Host}}',
                ]
            ]
        )

        self.assertEqual(self.docker_environment_log.read_text().strip(), 'unset|unset|unset|unset|' + os.environ['IMAGE_NAMESPACE'])
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
        self.assertIn(str(ROOT / 'model/weights.yaml'), compose[0])

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
        self.assertIn(str(ROOT / 'model/weights.yaml'), compose[0])
        self.assertIn(os.environ['IMAGE_NAMESPACE'] + '-weights-writer', compose[0])
        self.assertEqual(compose[0][compose[0].index('--pull') + 1], 'never')

        self.docker_log.unlink()
        verify = self.run_make('weights/verify', 'DOCKER_CONTEXT=local')

        self.assertEqual(verify.returncode, 0, verify.stderr)
        compose = [call for call in self.calls() if call[0] == '--host']

        self.assertEqual(len(compose), 1)
        self.assertEqual(compose[0][-1], 'verify')
        self.assertIn(str(ROOT / 'model/weights.yaml'), compose[0])
        self.assertEqual(compose[0][compose[0].index('--pull') + 1], 'never')
