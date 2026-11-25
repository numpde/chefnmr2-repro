"""Runtime startup and model tests preserve the SDK's process lifecycle."""

import json
import subprocess
import unittest
from unittest.mock import patch

from runtime import ROOT, Runtime


class RuntimeNormativeTests(unittest.TestCase):
    def test_up_starts_without_an_existing_provider(self):
        runtime = Runtime('unix:///tmp/fake-docker.sock', 'localhost')

        with patch.object(runtime, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='')) as execute:
            runtime.up()

        commands = [call.args[0] for call in execute.call_args_list]
        self.assertFalse(any('stop' in command or 'inspect' in command for command in commands))
        startup = commands[-1]
        self.assertIn('up', startup)
        self.assertIn('--no-build', startup)
        self.assertEqual(startup[startup.index('--pull') + 1], 'never')
        self.assertIn('--wait', startup)
        self.assertEqual(set(startup[-3:]), {'provider', 'worker', 'interpreter-proxy'})

    def test_up_confirms_provider_retirement_before_replacing_worker(self):
        for initially_running in (False, True):
            with self.subTest(initially_running=initially_running):
                runtime = Runtime('unix:///tmp/fake-docker.sock', 'localhost')
                running = initially_running
                events = []

                def run(command, **kwargs):
                    nonlocal running
                    stdout = ''

                    if 'ps' in command:
                        stdout = 'provider-id\n'
                    elif 'stop' in command:
                        self.assertEqual(command[command.index('stop') + 1:], ['provider'])
                        running = False
                        events.append('stopped')
                    elif 'inspect' in command:
                        self.assertEqual(command[-1], 'provider-id')
                        stdout = json.dumps({'Running': running, 'ExitCode': 0})
                        events.append('inspected')
                    elif 'up' in command:
                        self.assertFalse(running)
                        events.append('started')
                    return subprocess.CompletedProcess(command, 0, stdout=stdout)

                with patch.object(runtime, 'run', side_effect=run):
                    runtime.up()

                self.assertEqual(events, ['stopped', 'inspected', 'started'])

    def test_model_passes_ordered_batches_and_replies_then_retires_project(self):
        runtime = Runtime('unix:///tmp/fake-docker.sock', 'localhost')
        batches = json.loads((ROOT / 'tests/model/cases/batch.json').read_text())['batches']
        replies = [{'replies': ['first', 'second']}, {'replies': ['third']}]

        def run(command, **kwargs):
            stdout = ''.join(json.dumps(reply) + '\n' for reply in replies) if command[-1] == 'manual' else ''
            return subprocess.CompletedProcess(command, 0, stdout=stdout)

        with patch.object(runtime, 'run', side_effect=run) as execute:
            runtime.test_model()

        calls = execute.call_args_list
        manual = next(call for call in calls if call.args[0][-1] == 'manual')
        validator = next(call for call in calls if '--batch-results' in call.args[0])

        self.assertEqual([json.loads(line) for line in manual.kwargs['input'].splitlines()], batches)
        self.assertEqual(json.loads(validator.kwargs['input']), replies)
        self.assertLess(calls.index(manual), calls.index(validator))
        commands = [call.args[0] for call in calls]
        projects = {command[command.index('--project-name') + 1] for command in commands}

        self.assertEqual(len(projects), 1)
        self.assertIn('down', commands[-1])
        self.assertIn('--remove-orphans', commands[-1])
        self.assertIn('--volumes', commands[-1])
