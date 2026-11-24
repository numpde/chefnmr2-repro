"""Check the selected worker's real model with public upstream benchmark input."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import tomllib

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import torch
from chefnmr2.infer import ChefNMR2, Peaks


def validate(result, request, members):
    assert result['n_samples'] == request['n_samples'] * members, 'All requested draws should complete'
    assert result['candidates'], 'Real inference should return reconstructed candidate structures'
    for candidate in result['candidates']:
        molecule = Chem.MolFromSmiles(candidate['smiles'])
        assert molecule is not None, f"{candidate['smiles']!r} should be a readable structure"
        actual_formula = rdMolDescriptors.CalcMolFormula(molecule)
        assert actual_formula == request['formula'], f"{actual_formula=} should match {request['formula']=}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--batch-results', type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error(f'{args.repeats=} should be positive so qualification executes an inference request')
    config = tomllib.loads(args.config.read_text())
    settings = config['model']
    case = json.loads(args.case.read_text())
    if args.batch_results is not None:
        results = json.loads(args.batch_results.read_text())
        assert len(results) == len(case['batches']), 'Every SDK batch should return one ordered result'
        entries = 0
        for batch, result in zip(case['batches'], results, strict=True):
            assert len(result['replies']) == len(batch['requests']), 'Every SDK batch entry should retain its reply position'
            for request, reply in zip(batch['requests'], result['replies'], strict=True):
                validate(reply, {**request, 'n_samples': config['offering']['n_samples']}, len(settings['checkpoint_dirs']))
                entries += 1
        print(json.dumps({'validated_batches': len(results), 'validated_entries': entries}))
        return
    expected_device = torch.device(settings['device']).type
    if expected_device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA qualification requires an available CUDA device'
        torch.cuda.reset_peak_memory_stats()
    hashes = {}
    for line in args.manifest.read_text().splitlines():
        if line and not line.startswith('#'):
            checksum, _maximum_bytes, name = line.split()
            hashes[name] = checksum
    checkpoint_evidence = {}
    members = []
    for name in settings['checkpoint_dirs']:
        directory = Path(name)
        for filename in ('model.pt', 'config.yaml'):
            path = directory / filename
            with path.open('rb') as source:
                actual = hashlib.file_digest(source, 'sha256').hexdigest()
            identity = directory.name + '/' + filename
            assert actual == hashes[identity], f'{identity}: checkpoint SHA-256 should match the release manifest'
            checkpoint_evidence[identity] = actual
        members.append((directory.name, directory / 'model.pt', directory / 'config.yaml'))
    started = time.monotonic()
    model = ChefNMR2(members, device=settings['device'], recon_workers=settings['recon_workers'])
    initialized_seconds = time.monotonic() - started
    observations = []
    try:
        actual_devices = {next(member['chef'].model.parameters()).device.type for member in model.members}
        assert actual_devices == {expected_device}, f'{actual_devices=} should contain only {expected_device=}'
        assert model.recon_pool is not None, 'Qualification requires the owned reconstruction pool'
        for index in range(args.repeats):
            started = time.monotonic()
            result = model.elucidate(Peaks(**case['peaks']), formula=case['formula'], n_samples=case['n_samples'], seed=case['seed'])
            validate(result, case, len(members))
            observations.append({'request': index + 1, 'elapsed_seconds': time.monotonic() - started, 'result': result})
    finally:
        if model.recon_pool is not None:
            model.recon_pool.shutdown()
    report = {
        'worker_image': os.environ['CHEFNMR2_WORKER_IMAGE_ID'],
        'upstream_revision': os.environ['CHEFNMR2_UPSTREAM_REVISION'],
        'python': platform.python_version(),
        'torch': torch.__version__,
        'sdk': importlib.metadata.version('provider-core'),
        'device': settings['device'],
        'checkpoints': checkpoint_evidence,
        'case': args.case.name,
        'initialization_seconds': initialized_seconds,
        'observations': observations,
        'reconstruction_pool_retired': True,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if expected_device == 'cuda' else None,
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if expected_device == 'cuda' else None,
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
