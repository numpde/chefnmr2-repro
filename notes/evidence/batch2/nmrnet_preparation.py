"""Measure captured geometry bytes and worker-local tensors in installed NMRNet images."""

import argparse
import base64
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time


def measure(operation):
    Path('/proc/self/clear_refs').write_text('5')
    before = memory()['VmRSS']
    started = time.perf_counter()
    result = operation()
    return result, {'seconds': time.perf_counter() - started, 'rss_before_bytes': before, **memory()}


def memory():
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    return {name: int(status[name].split()[0]) * 1024 for name in ('VmRSS', 'VmHWM')}


def generate():
    from nmrnet_provider.smiles import generate_conformers

    records = []
    for smiles, count in ((b'F[C@H](Cl)Br', 2), (b'CCCC', 3)):
        result, timing = measure(lambda: generate_conformers(smiles, count, seed=17, max_iterations=200, timeout_seconds=1))
        records.append({
            'source': smiles.decode(), 'requested_count': count, 'measurement': timing,
            'provenance': json.loads(result.provenance),
            'conformers': [
                {'id': conformer.conformer_id, 'request_base64': base64.b64encode(conformer.request_bytes).decode(),
                 'request_sha256': hashlib.sha256(conformer.request_bytes).hexdigest(),
                 'stereochemistry': json.loads(conformer.stereochemistry)}
                for conformer in result.conformers
            ],
        })
    return {'cases': records, 'rdkit_version': importlib.metadata.version('rdkit')}


def tensors(session):
    geometries = []
    for heads in session._prepared[2]:
        geometries.append([
            {'element': element, 'selected_atom_indices': composed.selected_atom_indices,
             'tensors': {name: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                               'bytes': tensor.numel() * tensor.element_size(),
                               'sha256': hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()}
                         for name, tensor in composed.model_inputs.items()}}
            for element, composed in heads
        ])
    return geometries


def predict():
    import torch
    from nmrnet_inference.predict import PredictionSession
    from nmrnet_inference.release import SelectedReleaseModels
    from nmrnet_provider.input_geometry import coordinate_request
    from nmrnet_provider.worker_protocol import frame

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    models = SelectedReleaseModels(declaration_path=Path('/input/prediction-models.json'), weight_root=Path('/weight'), checkpoint_root=Path('/checkpoints'))
    ordinary = coordinate_request(['C', 'H'] * 8, [[i, i * i % 7, i * 3 % 11] for i in range(16)])
    maximum = coordinate_request(['C', 'H'] * 64, [[i, i * i % 97, i * 7 % 89] for i in range(128)])
    generated = json.loads(Path('/generated.json').read_bytes())
    cases = {
        'one_atom': (coordinate_request(['H'], [[0, 0, 0]]),),
        'sixteen_atoms': (ordinary,),
        'generated_once': tuple(base64.b64decode(row['request_base64']) for row in generated['cases'][0]['conformers']),
        'maximum_count_and_atoms': (maximum,) * 128,
    }
    records = []
    for name, requests in cases.items():
        wire = frame({'operation': 'prepare', 'requests_base64': [base64.b64encode(raw).decode() for raw in requests]})
        session = PredictionSession(models.load_dictionary, models.load_selected_element)
        handle, preparation = measure(lambda: session.prepare(requests))
        accepted = tensors(session)
        replacement = PredictionSession(models.load_dictionary, models.load_selected_element)
        _, reconstruction = measure(lambda: replacement.prepare(requests))
        assert tensors(replacement) == accepted
        del replacement
        if name == 'maximum_count_and_atoms':
            prediction = None
        else:
            outcomes, prediction = measure(lambda: tuple(session.execute(handle)))
            prediction['geometry_count'] = len(outcomes)
            assert len(outcomes) == len(requests)
            assert session._prepared is None
        record = {
            'case': name, 'geometry_count': len(requests), 'request_bytes': sum(map(len, requests)),
            'wire_bytes': len(wire), 'request_sha256': [hashlib.sha256(raw).hexdigest() for raw in requests],
            'captured_requests_base64': [base64.b64encode(raw).decode() for raw in dict.fromkeys(requests)],
            'preparation': preparation, 'replacement_preparation': reconstruction, 'prediction_including_head_loads': prediction,
            'geometries': accepted,
            'tensor_bytes': sum(tensor['bytes'] for geometry in accepted for head in geometry for tensor in head['tensors'].values()),
        }
        if name == 'maximum_count_and_atoms':
            assert all(facts == accepted[0] for facts in accepted)
            record['geometries'] = accepted[:1]
            record['identical_geometry_count'] = len(accepted)
        records.append(record)
        print(json.dumps({'case': name, 'preparation': preparation, 'prediction': prediction}), file=sys.stderr, flush=True)
        del session
    return {'cases': records, 'declaration_sha256': models.declaration_sha256,
            'observed_artifacts': models.observed_artifacts(), 'torch': torch.__version__,
            'generated_input_sha256': hashlib.sha256(Path('/generated.json').read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('smiles', 'prediction'))
    options = parser.parse_args()
    with redirect_stdout(sys.stderr):
        result = generate() if options.phase == 'smiles' else predict()
    modules = ('nmrnet_provider.smiles', 'nmrnet_provider.input_geometry') if options.phase == 'smiles' else ('nmrnet_inference.predict', 'nmrnet_inference.model', 'nmrnet_inference.release')
    result['installed_source_sha256'] = {name: hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest() for name in modules}
    result.update(probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), python=platform.python_version(),
                  kernel=platform.release(), phase=options.phase,
                  memory_method='Linux per-phase clear_refs=5 VmHWM, includes imported modules and model/head loads')
    print(json.dumps(result, sort_keys=True, indent=2))
