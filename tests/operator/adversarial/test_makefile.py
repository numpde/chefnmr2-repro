"""Adversarial Make inputs and authority rejection behavior."""

import os
import shutil
from unittest.mock import patch

from operator_fixtures import OperatorFixture, ROOT


class MakefileAdversarialTests(OperatorFixture):
    def test_sensitive_proxy_and_cancellation_do_not_replace_history(self):
        endpoint = 'socks5h://127.0.0.1:1080'
        result = self.run_choose('PROXY_URL', endpoint)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        history = self.directory / 'repo/tmp/inputs/PROXY_URL'

        for sensitive in ('http://user:password@proxy:8080', 'https://proxy:8080?token=secret', 'https://proxy:8080#secret'):
            result = self.run_choose('PROXY_URL', sensitive)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((self.directory / 'selected').read_text(), sensitive + '\n')
            self.assertEqual(history.read_text(), endpoint + '\n')

        result = self.run_choose('PROXY_URL', '')

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(history.read_text(), endpoint + '\n')

    def test_ambient_context_does_not_select_a_noninteractive_operation(self):
        result = self.run_make('check', DOCKER_CONTEXT='ambient-context')

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('a terminal is required', result.stderr)
        self.assertEqual(self.calls(), [['context', 'ls', '--format', '{{.Name}}']])
        self.assertFalse(self.python_log.exists())

    def test_nested_make_does_not_inherit_parent_options(self):
        makefile = self.directory / 'ambient.mk'
        makefile.write_text('$(error Unexpected inherited makefile)\n')

        with patch.dict(os.environ, {
            'MAKEFLAGS': ' -- DOCKER_CONTEXT=science',
            'GNUMAKEFLAGS': '-n',
            'MAKEFILES': str(makefile),
        }):
            result = self.run_make('check', DOCKER_CONTEXT='ambient-context')

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('a terminal is required', result.stderr)
        self.assertEqual(self.calls(), [['context', 'ls', '--format', '{{.Name}}']])
        self.assertFalse(self.python_log.exists())

    def test_remote_context_is_rejected_before_runtime_command(self):
        result = self.run_make('check', 'DOCKER_CONTEXT=remote', FAKE_DOCKER_ENDPOINT='tcp://remote.example:2376')

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must use a local Unix socket', result.stderr)
        self.assertFalse(self.python_log.exists())

    def test_literal_profile_and_shell_sensitive_pool_are_not_evaluated(self):
        pool = '/tmp/cache with $HOME $(touch SHOULD_NOT_RUN)'

        result = self.run_make(
            'build',
            'PROFILE=worker-cpu',
            'DOCKER_CONTEXT=local',
            f'WHEELHOUSE={pool}',
            cwd='/tmp',
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.python_arguments()

        self.assertEqual(arguments[1], str(ROOT / 'make/build.py'))
        self.assertEqual(arguments[2:5], ['build', '--profile', 'worker-cpu'])
        self.assertEqual(arguments[5:7], ['--host', 'unix:///tmp/fake-docker.sock'])
        self.assertEqual(arguments[-2:], ['--pool', pool])
        self.assertFalse((self.directory / 'SHOULD_NOT_RUN').exists())

    def test_failed_gpu_probe_blocks_cuda_before_pool_prompt_or_docker(self):
        self.write_executable('nvidia-smi', '#!/bin/bash\nprintf "%s\\n" "fixture driver failure" >&2\nexit 9\n')
        result = self.run_make('build', 'PROFILE=worker-cuda')

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('fixture driver failure', result.stderr)
        self.assertIn('Cannot confirm a host NVIDIA GPU', result.stderr)
        self.assertIn('ASSUME_GPU=1', result.stderr)
        self.assertNotIn('terminal is required', result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse(self.python_log.exists())

    def test_successful_gpu_probe_with_empty_inventory_is_rejected(self):
        self.write_executable('nvidia-smi', '#!/bin/bash\nprintf " \\n"\n')
        result = self.run_make('gpu/check')

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('returned no devices', result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse(self.python_log.exists())

    def test_missing_probe_and_ambient_assumption_cannot_bypass_cuda_gate(self):
        for executable in ('make', 'dirname', 'env'):
            (self.bin / executable).symlink_to(shutil.which(executable))

        result = self.run_make(
            'build', 'PROFILE=worker-cuda',
            PATH=str(self.bin), ASSUME_GPU='1', REPRO_ASSUME_GPU='1',
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('nvidia-smi is unavailable', result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse(self.python_log.exists())

    def test_invalid_assumptions_are_literal_and_fail_before_hardware_access(self):
        marker = self.directory / 'unexpected'
        payload = '$(shell touch ' + str(marker) + ')'
        self.write_executable('nvidia-smi', '#!/bin/bash\nprintf "%s\\n" "unexpected probe" >&2\nexit 99\n')

        for value in ('', '2', 'yes', payload):
            for target in ('gpu/check', 'build'):
                with self.subTest(value=value, target=target):
                    result = self.run_make(target, 'PROFILE=worker-cuda', 'ASSUME_GPU=' + value)

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('ASSUME_GPU must be 0 or 1', result.stderr)
                    self.assertNotIn('unexpected probe', result.stderr)
                    self.assertEqual(self.calls(), [])
                    self.assertFalse(self.python_log.exists())
                    self.assertFalse(marker.exists())

    def test_cuda_build_dry_run_never_executes_the_reused_lane(self):
        self.write_executable('nvidia-smi', '#!/bin/bash\nprintf "%s\\n" "unexpected probe" >&2\nexit 99\n')
        result = self.run_make('-n', 'build', 'PROFILE=worker-cuda', 'WHEELHOUSE=/pool', 'DOCKER_CONTEXT=science')

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('unexpected probe', result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse(self.python_log.exists())
