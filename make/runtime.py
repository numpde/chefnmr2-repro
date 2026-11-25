"""Operate selected deployments with Compose and isolated manual runner state."""

import argparse
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, host, deployment):
        if not host:
            raise ValueError('Select a local Docker host socket explicitly for runtime operations')

        self.environment = {key: value for (key, value) in os.environ.items() if not key.startswith(('COMPOSE_', 'BUILDX_')) and (key not in ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH'))}
        self.docker = ['docker', '--host', host]
        self.deployment = deployment
        self.directory = ROOT / 'deployments' / deployment

        if deployment and (not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', deployment) or not self.directory.is_dir()):
            raise ValueError(f'Cannot select DEPLOYMENT={deployment!r}: choose an existing deployments directory')

    def run(self, command, **kwargs):
        try:
            return subprocess.run(command, cwd=ROOT, env=self.environment, check=True, **kwargs)
        except subprocess.CalledProcessError as error:
            if error.stderr:
                print(error.stderr, file=sys.stderr, end='')

            raise

    def compose(self, api=False, project=None, model_test=False):
        if not self.deployment:
            raise ValueError('Select DEPLOYMENT explicitly for Compose operations')

        command = self.docker + ['compose', '--project-directory', str(ROOT), '--env-file', '/dev/null', '-f', str(ROOT / 'compose/inference.yaml'), '-f', str(self.directory / 'compose.yaml')]

        if not api:
            command += ['--profile', 'manual']

        if api:
            command += ['-f', str(ROOT / 'compose/provider.yaml'), '-f', str(self.directory / 'provider.yaml')]

        if model_test:
            command += ['-f', str(ROOT / 'compose/test-model.yaml')]

        if project:
            command += ['--project-name', project]

        return command

    def config(self, api=False):
        return json.loads(self.run(self.compose(api=api) + ['config', '--format', 'json'], capture_output=True, text=True).stdout)

    def check(self, category='all'):
        definition = json.loads(self.run(self.docker + ['buildx', 'bake', '-f', 'compose/images.yaml', '--print', 'provider'], capture_output=True, text=True).stdout)
        image = definition['target']['provider']['tags'][0]
        command = self.docker + ['run', '--rm', '--pull=never', '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--cpus', '1', '--memory', '512m', '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=64m', '-e', 'PROVIDER_CORE_IN_CONTAINER=1', '-e', 'PYTHONPATH=/tests/provider:/opt/provider', '--mount', f'type=bind,src={ROOT / "tests"},dst=/tests,readonly', '--entrypoint', 'python', image, '-B', '-m', 'unittest', 'discover']

        for scenario in (('normative', 'adversarial') if category == 'all' else (category,)):
            log.info('Running %s provider tests', scenario)
            self.run(command + ['-s', '/tests/provider/' + scenario, '-p', 'test_*.py', '-v'])

        log.info('Provider tests passed')

    @contextmanager
    def isolated(self, model_test=False):
        project = 'chefnmr2-run-' + uuid.uuid4().hex[:12]
        command = self.compose(project=project, model_test=model_test)
        failure = None

        try:
            yield command
        except BaseException as error:
            failure = error

            raise
        finally:
            try:
                log.info('Retiring isolated project %s', project)
                self.run(command + ['down', '--remove-orphans', '--volumes', '--timeout', '30'])
                log.info('Isolated project %s retired', project)
            except subprocess.CalledProcessError:
                if failure is None:
                    raise

                failure.add_note('Isolated project cleanup failed; retry ' + shlex.join(command + ['down', '--remove-orphans', '--volumes']))

    def manual(self):
        with self.isolated() as command:
            log.info('Starting isolated worker for %s', self.deployment)
            self.run(command + ['up', '-d', '--no-build', '--pull', 'never', 'worker'])
            log.info('Starting manual client; reading request batches from stdin until EOF')
            self.run(command + ['run', '--rm', '--no-deps', '-T', 'manual'])

        log.info('Manual run complete')

    def test_model(self):
        batch_case = json.loads((ROOT / 'tests/model/cases/batch.json').read_text())

        with self.isolated(model_test=True) as command:
            checker = command + ['run', '--rm', '--no-deps', '-T', '--entrypoint', 'python', 'worker', '-B', '/tests/model/check.py', '--config', '/config/provider.toml', '--manifest', '/checks/checkpoints.sha256']

            log.info('Checking cold and warm upstream morphine inference')
            self.run(checker + ['--case', '/tests/model/cases/morphine.json'])
            log.info('Cold and warm inference completed; starting SDK worker')
            self.run(command + ['up', '-d', '--no-build', '--pull', 'never', 'worker'])
            log.info('Checking ordered SDK batches and retirement')
            manual = self.run(command + ['run', '--rm', '--no-deps', '-T', 'manual'], input=''.join((json.dumps(batch) + '\n') for batch in batch_case['batches']), stdout=subprocess.PIPE, text=True)
            replies = [json.loads(line) for line in manual.stdout.splitlines()]

            log.info('Validating completed batch results')
            self.run(checker + ['--case', '/tests/model/cases/batch.json', '--batch-results'], input=json.dumps(replies), text=True)

        log.info('Model and SDK tests passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('action', choices=('check', 'test/model', 'config', 'manual/config', 'status', 'logs', 'manual'))
    parser.add_argument('--host', required=True, help='Resolved local Unix socket for the selected Docker context')
    parser.add_argument('--deployment', default='')
    parser.add_argument('--category', choices=('all', 'normative', 'adversarial'), default='all')
    args = parser.parse_args()

    log.info('Starting %s%s', args.action, f' for {args.deployment}' if args.deployment else '')
    runtime = Runtime(args.host, args.deployment)

    if args.action == 'check':
        runtime.check(args.category)
    elif args.action == 'test/model':
        runtime.test_model()
    elif args.action in ('config', 'manual/config'):
        print(json.dumps(runtime.config(api=args.action == 'config'), indent=2))
    elif args.action == 'status':
        runtime.run(runtime.compose(api=True) + ['ps', '--all'])
    elif args.action == 'logs':
        runtime.run(runtime.compose(api=True) + ['logs', '--follow', '--tail', '100', 'provider', 'worker'])
    elif args.action == 'manual':
        runtime.manual()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    try:
        main()
    except subprocess.CalledProcessError as error:
        for note in getattr(error, '__notes__', ()):
            print(note, file=sys.stderr)

        raise SystemExit(error.returncode)
