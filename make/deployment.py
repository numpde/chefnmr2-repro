"""Promote qualified images only after the current SDK session retires."""

from copy import deepcopy
import json
import logging
import os
import hashlib
from pathlib import Path
import subprocess
import time

from work import create_run_directory

log = logging.getLogger(__name__)


def inspect(runtime, identities):
    if not identities:
        return []
    return json.loads(runtime.run(runtime.docker + ['inspect', *identities], capture_output=True, text=True).stdout)


def inventory(runtime, command):
    identities = runtime.run(command + ['ps', '--all', '--quiet'], capture_output=True, text=True).stdout.split()
    return inspect(runtime, identities)


def retire(runtime, containers):
    providers = [container for container in containers if container['Config']['Labels']['com.docker.compose.service'] == 'provider']
    if len(providers) > 1:
        raise ValueError('Cannot replace the worker: more than one provider container owns the selected deployment')
    if providers:
        provider = providers[0]
        if provider['State']['Running']:
            log.info('Stopping the current provider and waiting for SDK retirement')
            runtime.run(runtime.docker + ['stop', '--time', '90', provider['Id']])
        state = inspect(runtime, [provider['Id']])[0]['State']
        if state['Running'] or state['ExitCode'] != 0:
            raise RuntimeError(f"Cannot replace the worker: provider {provider['Id']} did not confirm SDK retirement; {state['ExitCode']=}")


def deploy(runtime, qualification):
    if os.getuid() == 0:
        raise ValueError('Promote images as the non-root checkout owner so rollback evidence keeps that ownership')
    log.info('Checking qualification evidence: %s', qualification)
    summary = json.loads((qualification / 'summary.json').read_text())
    inputs = json.loads((qualification / 'inputs.json').read_text())
    if not summary['qualification_project_retired'] or inputs != runtime.qualification_inputs():
        raise ValueError('Cannot deploy: scientific configuration changed or qualification did not finish; run make qualify for the selected deployment')
    definition = runtime.config(api=True)
    expected = {'worker': summary['worker_image'], 'provider': summary['provider_image'], 'interpreter-proxy': summary['provider_image']}
    for service, evidence in expected.items():
        actual = inspect(runtime, [definition['services'][service]['image']])[0]['Id']
        if actual != evidence:
            raise ValueError(f'Cannot deploy {service}: its selected image differs from the qualified artifact')
    command = runtime.compose(api=True)
    containers = inventory(runtime, command)
    root = runtime.directory.parents[1]
    for container in containers:
        labels = container['Config']['Labels']
        if labels['com.docker.compose.project'] != definition['name'] or not Path(labels['com.docker.compose.project.working_dir']).resolve().is_relative_to(root):
            raise ValueError('Cannot replace containers: their Compose ownership differs from this checkout and deployment')
    backup = create_run_directory(root, 'rollback')
    log.info('Saving rollback state: %s', backup)
    (backup / 'containers.json').write_text(json.dumps(containers, indent=2))
    rollback = deepcopy(definition)
    rollback['x-chefnmr2-run'] = {'engine': runtime.docker[-1], 'checkout': str(root), 'deployment': runtime.deployment}
    for container in containers:
        service = container['Config']['Labels']['com.docker.compose.service']
        rollback['services'][service]['image'] = container['Image']
        if service == 'provider' and container['State']['Running']:
            mounted = runtime.run(runtime.docker + ['exec', container['Id'], 'cat', '/config/provider.toml'], capture_output=True, text=True).stdout
            (backup / 'provider.toml').write_text(mounted)
            rollback['configs']['deployment_provider']['file'] = str(backup / 'provider.toml')
    (backup / 'compose.json').write_text(json.dumps(rollback, indent=2))
    retire(runtime, containers)
    try:
        log.info('Starting the qualified provider, worker and interpreter proxy')
        runtime.run(command + ['up', '-d', '--no-build', '--pull', 'never', '--wait', '--wait-timeout', '150', 'worker', 'interpreter-proxy', 'provider'])
        # Services without healthchecks can exit immediately after Compose's
        # running-state observation. Check again before recording promotion.
        log.info('Waiting five seconds before checking the running deployment')
        time.sleep(5)
        promoted = inventory(runtime, command)
        actual_services = {container['Config']['Labels']['com.docker.compose.service']: container for container in promoted}
        if set(actual_services) != set(expected) or len(promoted) != len(expected):
            raise RuntimeError('The running deployment should contain exactly worker, provider and interpreter-proxy')
        for service, image in expected.items():
            container = actual_services[service]
            if not container['State']['Running'] or container['Image'] != image:
                raise RuntimeError(f'Cannot confirm promotion: {service} should be running the exact qualified image {image}')
        mounted = runtime.run(runtime.docker + ['exec', actual_services['provider']['Id'], 'cat', '/config/provider.toml'], capture_output=True).stdout
        if hashlib.sha256(mounted).hexdigest() != inputs['provider_toml_sha256']:
            raise RuntimeError('Cannot confirm promotion: the mounted provider TOML differs from the qualified configuration')
    except (subprocess.CalledProcessError, RuntimeError) as failure:
        log.info('Promotion failed; attempting rollback from %s', backup)
        try:
            retire(runtime, inventory(runtime, command))
            if containers:
                runtime.run(runtime.docker + ['compose', '--project-directory', str(root), '--project-name', definition['name'], '-f', str(backup / 'compose.json'), 'up', '-d', '--no-build', '--pull', 'never', 'worker', 'interpreter-proxy', 'provider'])
        except (subprocess.CalledProcessError, RuntimeError) as rollback_failure:
            failure.add_note(f'Rollback could not finish; retain {backup}: {rollback_failure}')
        raise
    (backup / 'promoted.json').write_text(json.dumps(promoted, indent=2))
    log.info('Promoted qualified artifacts; rollback evidence: %s', backup)
    runtime.prune_records('rollback', 2, protected=promoted)
