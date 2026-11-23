"""Run the recorded preparation probes in disposable offline CPU containers."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
REPOS = ROOT.parents[3]
NMRPEAK_SOURCE = '3288d5cbe4eac02e45645c321a52e4e68c90a370'
NMRPEAK = {
    'hf': ('89e3f1cd0bb00fa6abe31651d91557d491de2ed4d7bbf6a56a0d01ee9a85a505', 'abab0ff1e6787edfdf268312fba7605f090596c60335307b101357915b77d4a6'),
    'chf': ('ba8f79d694f567077b33c4524890d7d623445397db54c9ad356e18ea3fd63901', '375ecd6db031300d2f301a0eb7042c3f42cbd527b17d110e3f0a3d47eecc3a02'),
}


def bind(source, destination):
    return ['--mount', f'type=bind,src={source},dst={destination},readonly']


def configuration(name):
    if name.startswith('magnet-'):
        family = name.removeprefix('magnet-')
        image = {'e01': '964ba65f5a8b681a6aa96dd8e55d58427798043bb778cc85f42f959ab7ca2edc',
                 'e02': 'c93c8e7dd4b948b146131c7bcd14c93d29585f0410674cb7702b0d90dd575c5b'}[family]
        return image, bind(ROOT / 'magnet_preparation.py', '/probe.py'), ['-P', '-B', '/probe.py', family]
    if name == 'secs':
        mounts = [
            *bind(ROOT / 'secs_preparation.py', '/probe.py'),
            *bind(REPOS / 'secs-repro/cache/molformer', '/cache'),
            *bind(REPOS / 'secs-repro/checkpoints/residual_augment_resolution_physics_finetune_20250708_1747', '/checkpoint'),
            *bind(REPOS / 'secs-repro/molformer.lock.toml', '/input/molformer.lock.toml'),
            *bind(REPOS / 'secs-repro/tests/fixtures', '/fixtures'),
            *bind(REPOS / 'secs-repro/tools/materialize_molformer_cache.py', '/verify_cache.py'),
            '--tmpfs', '/modules:rw,noexec,nosuid,nodev,size=32m,mode=1777',
            '--env', 'HF_HUB_CACHE=/cache/hub', '--env', 'HF_MODULES_CACHE=/modules',
            '--env', 'TRANSFORMERS_OFFLINE=1',
        ]
        return '783470ab055e970e6468e3aa9d58da209c7a0555dd721f511b5e4bc8a0580f22', mounts, ['-P', '-B', '/probe.py']
    if name.startswith('nmrnet-'):
        phase = name.removeprefix('nmrnet-')
        image = {
            'smiles': 'ac65b48cb927105349c103dc5a8d623b89b9b604330f79a2e6f3b21adbf76788',
            'prediction': '6ad9aee418867a3dba64ef66b69d395bfb05c8ba31232610c9f5a51daa97c706',
        }[phase]
        mounts = bind(ROOT / 'nmrnet_preparation.py', '/probe.py')
        if phase == 'prediction':
            mounts += ['--user', '1000:1000']
            mounts += [
                *bind(REPOS / 'nmrnet-repro/.work/prepared/weight', '/weight'),
                *bind(REPOS / 'nmrnet-repro/.work/prepared/checkpoints', '/checkpoints'),
                *bind(REPOS / 'nmrnet-repro/secrets/deployments/nmr-localhost/bundles/generation-UQyiNtDR/prediction-models.json', '/input/prediction-models.json'),
                *bind(ROOT / 'nmrnet-smiles.json', '/generated.json'),
            ]
        return image, mounts, ['-I', '-B', '/probe.py', phase]
    family = name.removeprefix('nmrpeak-')
    image, checkpoint = NMRPEAK[family]
    mounts = [
        '--mount', f'type=volume,src=nmrpeak-{family}-checkpoint-{checkpoint},dst=/checkpoint,readonly,volume-nocopy',
        *bind(ROOT / 'nmrpeak_preparation.py', '/probe.py'),
        *bind(REPOS / 'nmrpeak-repro/benchmark_tools/cases.json', '/corpus.json'),
    ]
    arguments = [
        '-P', '-B', '/probe.py', '--family', family, '--checkpoint', f'sha256:{checkpoint}',
        '--corpus', '/corpus.json', '--source-revision', NMRPEAK_SOURCE, '--image', f'sha256:{image}',
    ]
    return image, mounts, arguments


def run(name):
    image, mounts, arguments = configuration(name)
    interpreter = {'nmrnet-smiles': '/opt/rdkit/environment/bin/python',
                   'nmrnet-prediction': '/usr/local/bin/python'}.get(name, 'python')
    environment = {key: value for key, value in os.environ.items() if key not in {
        'DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH',
    }}
    with TemporaryDirectory(prefix='preparation-probe-') as temporary:
        cidfile = Path(temporary) / 'container'
        output = Path(temporary) / 'result.json'
        command = [
            'docker', '--context', 'science', 'run', '--rm', '--pull', 'never', '--init',
            '--cidfile', str(cidfile), '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges:true', '--pids-limit', '256', '--cpus', '4',
            '--memory', '8g', '--memory-swap', '8g', '--shm-size', '256m',
            '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=512m,mode=1777',
            '--env', 'OMP_NUM_THREADS=4', '--env', 'MKL_NUM_THREADS=4',
            *mounts, '--entrypoint', interpreter, f'sha256:{image}', *arguments,
        ]
        with output.open('wb') as stdout, (Path('/tmp') / f'batch2-{name}.log').open('wb') as stderr:
            process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr)
            try:
                result = process.wait(timeout=900)
            except BaseException:
                if cidfile.exists():
                    container = cidfile.read_text().strip()
                    assert re.fullmatch('[0-9a-f]{64}', container)
                    subprocess.run(['docker', '--context', 'science', 'stop', '--timeout', '5', container], env=environment, check=True)
                process.wait(timeout=15)
                raise
        if result:
            raise SystemExit(f'Probe {name} exited {result}; see /tmp/batch2-{name}.log')
        document = json.loads(output.read_bytes())
        # Capture the actual argv; only the disposable cidfile path is run-specific
        document['container_command'] = command
        (ROOT / f'{name}.json').write_text(json.dumps(document, sort_keys=True, indent=2) + '\n')
        print(f'{name}: recorded {ROOT / (name + ".json")}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=('nmrpeak-hf', 'nmrpeak-chf', 'nmrnet-smiles', 'nmrnet-prediction', 'secs', 'magnet-e01', 'magnet-e02'))
    run(parser.parse_args().name)
