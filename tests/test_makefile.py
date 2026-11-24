"""Exercise the operator Make interface without contacting a Docker engine."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MakefileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='chefnmr2-make-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.bin = self.directory / 'bin'
        self.bin.mkdir()
        self.docker_log = self.directory / 'docker.log'
        self.docker_environment_log = self.directory / 'docker-environment.log'
        self.python_log = self.directory / 'python.log'
        self.proxy_log = self.directory / 'proxy.log'
        self.write_executable('docker', '''#!/bin/bash
printf '%s\\0' "$@" >> "$TEST_DOCKER_LOG"
printf '\\n' >> "$TEST_DOCKER_LOG"
printf '%s|%s|%s|%s\\n' "${DOCKER_HOST-unset}" "${DOCKER_CONFIG-unset}" "${COMPOSE_FILE-unset}" "${BUILDX_METADATA_PROVENANCE-unset}" >> "$TEST_DOCKER_ENV_LOG"
printf '%s\\n' "${PROXY_URL-unset}" >> "$PROXY_LOG"
if [[ $1 == context && $2 == inspect ]]; then
  printf '%s\\n' "${FAKE_DOCKER_ENDPOINT:-unix:///tmp/fake-docker.sock}"
fi
''')
        self.write_executable('python3', '''#!/bin/bash
printf '%s\\0' "$@" > "$PYTHON_LOG"
''')

    def write_executable(self, name, contents):
        path = self.bin / name
        path.write_text(contents)
        path.chmod(0o755)

    def environment(self, **values):
        return {
            **os.environ,
            'PATH': str(self.bin) + os.pathsep + os.environ['PATH'],
            'TEST_DOCKER_LOG': str(self.docker_log),
            'TEST_DOCKER_ENV_LOG': str(self.docker_environment_log),
            'PYTHON_LOG': str(self.python_log),
            'PROXY_LOG': str(self.proxy_log),
            'DOCKER_HOST': 'tcp://ambient.example:2376',
            'DOCKER_CONFIG': '/ambient/docker-config',
            'COMPOSE_FILE': '/ambient/compose.yaml',
            'BUILDX_METADATA_PROVENANCE': 'ambient',
            'PROXY_URL': 'ambient',
            **values,
        }

    def run_make(self, *arguments, cwd=None, **environment):
        return subprocess.run(
            ['make', '-f', str(ROOT / 'Makefile'), *arguments],
            cwd=cwd or self.directory,
            env=self.environment(**environment),
            capture_output=True,
            text=True,
        )

    def calls(self):
        if not self.docker_log.exists():
            return []
        return [line.rstrip('\n').split('\0')[:-1] for line in self.docker_log.read_text().splitlines()]

    def python_arguments(self):
        return self.python_log.read_bytes().decode().rstrip('\0').split('\0')

    def test_help_and_dry_run_do_not_call_docker_or_scripts(self):
        help_result = self.run_make('help')
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn('make weights/sanitize', help_result.stdout)
        self.assertIn('Sanitize known checkpoint files', help_result.stdout)
        dry_run = self.run_make('-n', 'weights/verify', 'DOCKER_CONTEXT=local')
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertFalse(self.docker_log.exists())
        self.assertFalse(self.python_log.exists())

    def test_environment_context_is_ignored_and_scripts_receive_resolved_socket(self):
        result = self.run_make('check', 'DOCKER_CONTEXT=local', cwd='/tmp')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [[
            'context', 'inspect', 'local', '--format', '{{.Endpoints.docker.Host}}',
        ]])
        self.assertEqual(self.docker_environment_log.read_text().strip(), 'unset|unset|unset|unset')
        arguments = self.python_arguments()
        self.assertEqual(arguments[1], str(ROOT / 'scripts/runtime.py'))
        self.assertEqual(arguments[2:6], ['check', '--host', 'unix:///tmp/fake-docker.sock'])
        self.assertNotIn('/ambient/compose.yaml', arguments)

    def test_missing_explicit_input_does_not_fall_back_to_environment(self):
        result = self.run_make('check', DOCKER_CONTEXT='ambient-context')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('a terminal is required', result.stderr)
        self.assertFalse(self.docker_log.exists())
        self.assertFalse(self.python_log.exists())

    def test_non_local_context_is_rejected_before_runtime_command(self):
        result = self.run_make('check', 'DOCKER_CONTEXT=remote', FAKE_DOCKER_ENDPOINT='tcp://remote.example:2376')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must use a local Unix socket', result.stderr)
        self.assertFalse(self.python_log.exists())

    def test_literal_profile_pool_and_root_relative_script_path(self):
        pool = '/tmp/cache with $HOME $(touch SHOULD_NOT_RUN)'
        result = self.run_make(
            'build', 'PROFILE=worker-cpu', 'DOCKER_CONTEXT=local', f'WHEELHOUSE={pool}', cwd='/tmp',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.python_arguments()
        self.assertEqual(arguments[1], str(ROOT / 'scripts/build.py'))
        self.assertEqual(arguments[2:5], ['build', '--profile', 'worker-cpu'])
        self.assertEqual(arguments[5:7], ['--host', 'unix:///tmp/fake-docker.sock'])
        self.assertEqual(arguments[-2:], ['--pool', pool])
        self.assertFalse((self.directory / 'SHOULD_NOT_RUN').exists())

    def test_weights_workflows_call_explicit_compose_services_and_exclude_parallel_writers(self):
        result = self.run_make('weights/download', 'PROXY_URL=none', 'DOCKER_CONTEXT=local')
        self.assertEqual(result.returncode, 0, result.stderr)
        compose = [call for call in self.calls() if call[0] == '--host']
        self.assertEqual(len(compose), 3)
        self.assertEqual([call[-1] for call in compose], ['volume-init', 'download', 'verify'])
        self.assertIn('--env-file', compose[0])
        self.assertEqual(self.proxy_log.read_text().splitlines(), ['unset', 'none', 'none', 'none'])
        self.docker_log.unlink()
        sanitize = self.run_make('weights/sanitize', 'DOCKER_CONTEXT=local')
        self.assertEqual(sanitize.returncode, 0, sanitize.stderr)
        compose = [call for call in self.calls() if call[0] == '--host']
        self.assertEqual(len(compose), 1)
        self.assertEqual(compose[0][-1], 'sanitize')
        self.assertIn('chefnmr2-weights-writer', compose[0])


if __name__ == '__main__':
    unittest.main()
