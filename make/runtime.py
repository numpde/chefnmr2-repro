"""Operate selected deployments with Compose and isolated manual runner state."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid
from work import cleanup_run_directories, create_run_directory

ROOT = Path(__file__).resolve().parents[1]


class Runtime:
    def __init__(self, host, deployment):
        if not host:
            raise ValueError('Select a local Docker host socket explicitly for runtime operations')
        self.environment = {key: value for key, value in os.environ.items() if not key.startswith(('COMPOSE_', 'BUILDX_')) and key not in ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH', 'CHEFNMR2_PROVIDER_IMAGE', 'CHEFNMR2_WORKER_IMAGE')}
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

    @contextmanager
    def artifacts(self):
        """Serialize report creation, promotion and deletion in this checkout."""
        root = ROOT / 'tmp'
        root.mkdir(exist_ok=True)
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            if any((root / kind).is_symlink() for kind in ('qualification', 'rollback')):
                raise ValueError('Cannot manage qualification or rollback records through symbolic links')
            yield
        finally:
            os.close(descriptor)

    def prune_records(self, kind, keep, protected=()):
        """Trim completed history after success; incomplete records stay for investigation."""
        parent = ROOT / 'tmp' / kind
        marker = 'summary.json' if kind == 'qualification' else 'promoted.json'
        protected = [Path(mount['Source']).resolve() for container in protected for mount in container['Mounts']]
        count = 0
        obsolete = []
        for directory in sorted(parent.iterdir(), reverse=True):
            if (directory.is_symlink() or not directory.is_dir() or directory.stat().st_uid != os.getuid()
                    or not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', directory.name)):
                continue
            try:
                proof = json.loads((directory / marker).read_text())
                compose = json.loads((directory / 'compose.json').read_text())
                if kind == 'qualification':
                    if proof['qualification_project_retired'] is not True or proof['deployment'] != self.deployment:
                        continue
                    if Path(compose['configs']['deployment_provider']['file']).resolve() != (self.directory / 'provider.toml').resolve():
                        continue
                else:
                    owner = compose.get('x-chefnmr2-run', {})
                    if owner != {'engine': self.docker[-1], 'checkout': str(ROOT), 'deployment': self.deployment}:
                        continue
                    if not proof or not all(container['State']['Running'] is True for container in proof):
                        continue
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
            count += 1
            if count > keep and not any(path.is_relative_to(directory.resolve()) for path in protected):
                obsolete.append(directory)
        cleanup_run_directories(obsolete)
        for directory in obsolete:
            print('Removed obsolete record: ' + str(directory), flush=True)

    def compose(self, api=False, project=None, qualification=False):
        if not self.deployment:
            raise ValueError('Select DEPLOYMENT explicitly for Compose operations')
        command = self.docker + ['compose', '--project-directory', str(ROOT), '--env-file', str(self.directory / 'images.env'), '-f', str(ROOT / 'compose/inference.yaml'), '-f', str(self.directory / 'compose.yaml')]
        if not api:
            command += ['--profile', 'manual']
        if api:
            command += ['-f', str(ROOT / 'compose/provider.yaml'), '-f', str(self.directory / 'provider.yaml')]
        if qualification:
            command += ['-f', str(ROOT / 'compose/qualification.yaml')]
        if project:
            command += ['--project-name', project]
        return command

    def config(self, api=False, project=None, qualification=False):
        return json.loads(self.run(self.compose(api, project, qualification) + ['config', '--format', 'json'], capture_output=True, text=True).stdout)

    def check(self, category='all'):
        definition = json.loads(self.run(self.docker + ['buildx', 'bake', '-f', 'docker-bake.hcl', '--print', 'provider'], capture_output=True, text=True).stdout)
        image = definition['target']['provider']['tags'][0]
        command = self.docker + ['run', '--rm', '--pull=never', '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--cpus', '1', '--memory', '512m', '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=64m', '-e', 'PROVIDER_CORE_IN_CONTAINER=1', '-e', 'PYTHONPATH=/tests/provider:/opt/provider', '--mount', f'type=bind,src={ROOT / "tests"},dst=/tests,readonly', '--entrypoint', 'python', image, '-B', '-m', 'unittest', 'discover']
        for scenario in (('normative', 'adversarial') if category == 'all' else (category,)):
            self.run(command + ['-s', '/tests/provider/' + scenario, '-p', 'test_*.py', '-v'])

    @contextmanager
    def isolated(self, qualification=False):
        project = 'chefnmr2-qual-' + uuid.uuid4().hex[:12]
        command = self.compose(project=project, qualification=qualification)
        failure = None
        try:
            yield command
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                self.run(command + ['down', '--remove-orphans', '--volumes', '--timeout', '30'])
            except subprocess.CalledProcessError:
                if failure is None:
                    raise
                failure.add_note('Qualification cleanup failed; retry ' + shlex.join(command + ['down', '--remove-orphans', '--volumes']))

    def manual(self):
        with self.isolated() as command:
            self.run(command + ['up', '-d', '--no-build', '--pull', 'never', 'worker'])
            self.run(command + ['run', '--rm', '--no-deps', '-T', 'manual'])

    def qualification_inputs(self):
        science = self.config()
        configuration = Path(science['configs']['deployment_provider']['file'])
        return {
            'science_compose_sha256': hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest(),
            'provider_toml_sha256': hashlib.sha256(configuration.read_bytes()).hexdigest(),
            'checkpoint_manifest_sha256': hashlib.sha256((ROOT / 'model/checkpoints.sha256').read_bytes()).hexdigest(),
            'model_source_sha256': {name: hashlib.sha256((ROOT / 'model' / name).read_bytes()).hexdigest() for name in ('binding.py', 'wire.py')},
            'checker_sha256': hashlib.sha256((ROOT / 'tests/model/qualify.py').read_bytes()).hexdigest(),
            'case_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT / 'tests/model/cases').glob('*.json'))},
        }

    def qualify(self):
        if os.getuid() == 0:
            raise ValueError('Run qualification as the non-root checkout owner so retained evidence keeps that ownership')
        directory = create_run_directory(ROOT, 'qualification')
        self.environment['CHEFNMR2_QUALIFICATION_DIRECTORY'] = str(directory)
        configuration = self.config(qualification=True)
        (directory / 'compose.json').write_text(json.dumps(configuration, indent=2))
        (directory / 'inputs.json').write_text(json.dumps(self.qualification_inputs(), indent=2))
        worker = configuration['services']['worker']['image']
        images = json.loads(self.run(self.docker + ['image', 'inspect', worker, configuration['services']['manual']['image']], capture_output=True, text=True).stdout)
        (directory / 'images.json').write_text(json.dumps(images, indent=2))
        upstream = self.run(['git', '-C', str(ROOT / 'model/upstream'), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        batch_case = json.loads((ROOT / 'tests/model/cases/batch.json').read_text())
        with self.isolated(qualification=True) as command:
            common = command + ['run', '--rm', '--no-deps', '-T', '--entrypoint', 'python', 'worker', '-B', '/qualification/qualify.py', '--config', '/config/provider.toml', '--manifest', '/checks/checkpoints.sha256']
            direct = command + ['run', '--rm', '--no-deps', '-T', '--env', 'CHEFNMR2_WORKER_IMAGE_ID=' + images[0]['Id'], '--env', 'CHEFNMR2_UPSTREAM_REVISION=' + upstream, '--entrypoint', 'python', 'worker', '-B', '/qualification/qualify.py', '--config', '/config/provider.toml', '--manifest', '/checks/checkpoints.sha256', '--case', '/qualification/cases/morphine.json']
            print('Checking cold and warm upstream morphine inference in ' + images[0]['Id'], flush=True)
            model = self.run(direct, capture_output=True, text=True)
            (directory / 'model.json').write_text(model.stdout)
            (directory / 'model.stderr').write_text(model.stderr)
            if json.loads(model.stdout)['model_source_sha256'] != self.qualification_inputs()['model_source_sha256']:
                raise ValueError('Cannot qualify these images: the worker model contract differs from this checkout; rebuild and select the matching images')
            self.run(command + ['up', '-d', '--no-build', '--pull', 'never', 'worker'])
            print('Checking ordered SDK batches and retirement in the selected provider image', flush=True)
            manual = self.run(command + ['run', '--rm', '--no-deps', '-T', 'manual'], input=''.join(json.dumps(batch) + '\n' for batch in batch_case['batches']), capture_output=True, text=True)
            replies = [json.loads(line) for line in manual.stdout.splitlines()]
            (directory / 'manual.json').write_text(json.dumps(replies, indent=2))
            (directory / 'manual.stderr').write_text(manual.stderr)
            validated = self.run(common + ['--case', '/qualification/cases/batch.json', '--batch-results', '/evidence/manual.json'], capture_output=True, text=True)
            (directory / 'batch-validation.json').write_text(validated.stdout)
            report = {'deployment': self.deployment, 'worker_image': images[0]['Id'], 'provider_image': images[1]['Id'], 'upstream_revision': upstream, 'model': json.loads(model.stdout), 'sdk_batches': json.loads(validated.stdout), 'sdk_session_retired': True}
        report['qualification_project_retired'] = True
        (directory / 'summary.json').write_text(json.dumps(report, indent=2))
        print('Qualification passed; evidence: ' + str(directory), flush=True)
        self.prune_records('qualification', 3)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'qualify', 'deploy', 'config', 'manual/config', 'status', 'logs', 'manual'))
    parser.add_argument('--host', required=True, help='Resolved local Unix socket for the selected Docker context')
    parser.add_argument('--deployment', default='')
    parser.add_argument('--qualification', default='')
    parser.add_argument('--category', choices=('all', 'normative', 'adversarial'), default='all')
    args = parser.parse_args()
    runtime = Runtime(args.host, args.deployment)
    if args.action == 'check':
        runtime.check(args.category)
    elif args.action in ('qualify', 'deploy'):
        if args.action == 'deploy' and not args.qualification:
            parser.error('Select QUALIFICATION=<evidence directory> so deployment promotes proved artifacts')
        with runtime.artifacts():
            if args.action == 'qualify':
                runtime.qualify()
            else:
                from deployment import deploy
                deploy(runtime, Path(args.qualification))
    elif args.action in ('config', 'manual/config'):
        print(json.dumps(runtime.config(api=args.action == 'config'), indent=2))
    elif args.action == 'status':
        runtime.run(runtime.compose(api=True) + ['ps', '--all'])
    elif args.action == 'logs':
        runtime.run(runtime.compose(api=True) + ['logs', '--follow', '--tail', '100', 'provider', 'worker'])
    elif args.action == 'manual':
        runtime.manual()


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        for note in getattr(error, '__notes__', ()):
            print(note, file=sys.stderr)
        raise SystemExit(error.returncode)
