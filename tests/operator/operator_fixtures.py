"""Common real-shell fixtures for the operator Make contract tests."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class OperatorFixture(unittest.TestCase):
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
