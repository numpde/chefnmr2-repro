"""Model test orchestration uses one isolated project and ordered SDK replies."""

import json
import subprocess
import unittest
from unittest.mock import patch

from runtime import ROOT, Runtime


class RuntimeNormativeTests(unittest.TestCase):
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
