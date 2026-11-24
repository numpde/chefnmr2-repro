"""Operate the selected deployment and qualify its already-built images."""

import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check',))
    parser.add_argument('--context', required=True)
    args = parser.parse_args()
    if not args.context:
        parser.error('Select DOCKER_CONTEXT explicitly')
    environment = {key: value for key, value in os.environ.items() if not key.startswith(('COMPOSE_', 'BUILDX_')) and key not in ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH')}
    docker = ['docker', '--context', args.context]
    definition = json.loads(subprocess.run(docker + ['buildx', 'bake', '-f', 'docker-bake.hcl', '--print', 'provider'], cwd=ROOT, env=environment, check=True, capture_output=True, text=True).stdout)
    image = definition['target']['provider']['tags'][0]
    command = docker + ['run', '--rm', '--pull=never', '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--cpus', '1', '--memory', '512m', '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=64m', '-e', 'PROVIDER_CORE_IN_CONTAINER=1', '--mount', f'type=bind,src={ROOT / "tests"},dst=/tests,readonly', '--entrypoint', 'python', image, '-B', '-m', 'unittest', 'discover', '-s', '/tests', '-p', 'test_provider.py']
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == '__main__':
    main()
