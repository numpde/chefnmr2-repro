"""Failed retirement blocks replacement; failed tests still clean up their project."""

import json
import subprocess
import unittest
from unittest.mock import patch

from runtime import Runtime


def phase(command):
    if 'down' in command:
        return 'cleanup'
    if '--batch-results' in command:
        return 'validation'
    if command[-1] == 'manual':
        return 'manual'
    if 'up' in command:
        return 'startup'
    return 'inference'


class RuntimeAdversarialTests(unittest.TestCase):
    def test_up_rejects_unconfirmed_provider_retirement(self):
        for state in ({'Running': True, 'ExitCode': 0}, {'Running': False, 'ExitCode': 137}):
            with self.subTest(state=state):
                runtime = Runtime('unix:///tmp/fake-docker.sock', 'localhost')

                def run(command, **kwargs):
                    self.assertNotIn('up', command, 'Unconfirmed retirement must block worker replacement')
                    stdout = json.dumps(state) if 'inspect' in command else 'provider-id\n'
                    return subprocess.CompletedProcess(command, 0, stdout=stdout)

                with patch.object(runtime, 'run', side_effect=run):
                    with self.assertRaises(RuntimeError):
                        runtime.up()

    def test_up_propagates_stop_and_inspection_failures(self):
        for operation in ('stop', 'inspect'):
            with self.subTest(operation=operation):
                runtime = Runtime('unix:///tmp/fake-docker.sock', 'localhost')
                failure = subprocess.CalledProcessError(7, [operation])

                def run(command, **kwargs):
                    self.assertNotIn('up', command, 'A failed retirement operation must block startup')
                    if operation in command:
                        raise failure
                    return subprocess.CompletedProcess(command, 0, stdout='provider-id\n')

                with patch.object(runtime, 'run', side_effect=run):
                    with self.assertRaises(subprocess.CalledProcessError) as caught:
                        runtime.up()

                self.assertIs(caught.exception, failure)

    def test_up_rejects_multiple_providers_before_stopping_services(self):
        runtime = Runtime('unix:///tmp/fake-docker.sock', 'localhost')

        with patch.object(runtime, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='first\nsecond\n')) as execute:
            with self.assertRaises(ValueError):
                runtime.up()

        execute.assert_called_once()
        self.assertIn('ps', execute.call_args.args[0])

    def test_model_failure_retires_project_before_propagating(self):
        for failed_phase in ('inference', 'startup', 'manual', 'validation'):
            with self.subTest(failed_phase=failed_phase):
                runtime = Runtime('unix:///tmp/fake-docker.sock', 'localhost')
                failure = subprocess.CalledProcessError(7, [failed_phase])
                failed = False

                def run(command, **kwargs):
                    nonlocal failed
                    current = phase(command)

                    if failed:
                        self.assertEqual(current, 'cleanup', 'A failed test must not start another test stage')
                    if current == failed_phase:
                        failed = True
                        raise failure
                    return subprocess.CompletedProcess(command, 0, stdout='{"replies": []}\n')

                with patch.object(runtime, 'run', side_effect=run) as execute:
                    with self.assertRaises(subprocess.CalledProcessError) as caught:
                        runtime.test_model()

                self.assertIs(caught.exception, failure)
                commands = [call.args[0] for call in execute.call_args_list]
                self.assertEqual(phase(commands[-1]), 'cleanup')
                projects = {command[command.index('--project-name') + 1] for command in commands}
                self.assertEqual(len(projects), 1)

    def test_cleanup_failure_is_reported_without_replacing_a_test_failure(self):
        for fail_test in (False, True):
            with self.subTest(fail_test=fail_test):
                runtime = Runtime('unix:///tmp/fake-docker.sock', 'localhost')
                failure = subprocess.CalledProcessError(7, ['inference'])
                cleanup_failure = subprocess.CalledProcessError(9, ['cleanup'])

                def run(command, **kwargs):
                    if phase(command) == 'cleanup':
                        raise cleanup_failure
                    if fail_test:
                        raise failure
                    return subprocess.CompletedProcess(command, 0, stdout='{"replies": []}\n')

                with patch.object(runtime, 'run', side_effect=run):
                    with self.assertRaises(subprocess.CalledProcessError) as caught:
                        runtime.test_model()

                self.assertIs(caught.exception, failure if fail_test else cleanup_failure)
                if fail_test:
                    self.assertTrue(any('down' in note for note in failure.__notes__))
