"""Compare the real admission and generation codec inputs for a Magnet offering."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, fields, is_dataclass
import hashlib
import importlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest.mock import patch


def memory():
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    return {name: int(status[name].split()[0]) * 1024 for name in ('VmRSS', 'VmHWM')}


def measure(operation):
    Path('/proc/self/clear_refs').write_text('5')
    before = memory()['VmRSS']
    started = time.perf_counter()
    result = operation()
    return result, {'seconds': time.perf_counter() - started, 'rss_before_bytes': before, **memory()}


def tensor_facts(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device),
                'nbytes': value.numel() * value.element_size(),
                'sha256': hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
    if is_dataclass(value):
        return {field.name: tensor_facts(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: tensor_facts(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_facts(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unmeasured model-input member type: {type(value)}')


def run(family):
    import torch
    from runner import serving_worker as worker
    from magnet.generation import DecodeConfig

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    branch = 'e_01_jvae_nmrexp' if family == 'e01' else 'e_02_ch_nmrexp'
    contract = importlib.import_module(f'magnet_runner_contracts.runner_{branch}.session')
    client_module = importlib.import_module(f'magnet.stairway.{branch}.client_input')
    inference, loading = measure(lambda: worker._load_inference(device='cpu'))
    if family == 'e01':
        semantic = {'atom_counts': {'C': 2, 'H': 6, 'O': 1},
                    'nmr': {'mhz': 400.0, 'solvent': 'CDCl3', 'peaks': [
                        {'integral': 3.0, 'j_vals': [7.0], 'multiplicity': 'd', 'shift_hi': 1.1, 'shift_lo': 1.0}]}}
        client_input = client_module.Client1HNmrInput.from_json_dict(semantic)
        codec = inference.resources.client_nmr_codec
    else:
        peak = {'integral': 1.0, 'j_vals': [], 'multiplicity': 's', 'shift_hi': 1.0, 'shift_lo': 1.0}
        semantic = {'atom_inventory': [{'atomic_number': 6, 'isotope': 0, 'formal_charge': 0, 'count': 2},
                                       {'atomic_number': 8, 'isotope': 0, 'formal_charge': 0, 'count': 1}],
                    'nmr': {'1H': {'mhz': 400.0, 'solvent': 'CDCl3', 'peaks': [peak]},
                            '13C': {'mhz': 100.0, 'solvent': 'CDCl3', 'peaks': [{**peak, 'shift_hi': 50.0, 'shift_lo': 50.0}]}}}
        client_input = client_module.ClientNmrInput.from_json_dict(semantic)
        codec = inference.codec
    encoded = contract.encode_model_input(client_input)
    admission_options = {'inference': inference}
    if family == 'e01':
        admission_options['atom_slots'] = inference.resources.cfg.model.atom_prototypes.slot_count
    admitted, admission = measure(lambda: worker._accepted_model_input(encoded, **admission_options))
    assert contract.encode_model_input(admitted) == encoded
    timings = []
    for _ in range(20):
        prepared, timing = measure(lambda: codec.to_model(admitted))
        facts = tensor_facts(prepared)
        if not timings:
            accepted_facts = facts
        assert facts == accepted_facts
        timings.append(timing)
        del prepared
    # Match the fixed offering policy; output candidates are deliberately not compared
    policy = DecodeConfig(do_sample=True, max_new_tokens=96, min_new_tokens=1, num_beams=1,
                          num_return_sequences=100, temperature=0.9, top_k=None, top_p=0.9, use_cache=True)
    attempt = 'batch2-fixture'
    seed = int.from_bytes(hashlib.sha256(f'magnet.{family}.seed.v1\0{attempt}'.encode()).digest()[:8], 'big') & contract.MAX_SEED
    observed = []
    original = type(codec).to_model

    def observed_conversion(instance, scientific_input):
        started = time.perf_counter()
        prepared = original(instance, scientific_input)
        elapsed = time.perf_counter() - started
        observed.append({'seconds': elapsed, 'facts': tensor_facts(prepared)})
        return prepared

    with patch.object(type(codec), 'to_model', observed_conversion):
        operation = (lambda: inference.generate_from_nmr(admitted, decode=policy, seed=seed)) if family == 'e01' else (lambda: inference.generate(admitted, decode=policy, seed=seed))
        candidates, generation = measure(operation)
    assert len(observed) == 1 and observed[0]['facts'] == accepted_facts
    generation['candidate_count'] = len(candidates)
    return {
        'family': family, 'semantic_input': semantic, 'encoded_input': encoded,
        'input_json_bytes': len(json.dumps(encoded, separators=(',', ':')).encode()),
        'load': loading, 'admission': admission, 'preparations': timings,
        'preparation_median_seconds': statistics.median(row['seconds'] for row in timings),
        'admitted_tensor_facts': accepted_facts, 'generation': generation,
        'generation_conversion_seconds': observed[0]['seconds'], 'generation_tensor_identity': True,
        'decode_policy': asdict(policy), 'attempt': attempt, 'seed': seed,
        'checkpoint_sha256': hashlib.sha256(worker._CHECKPOINT_PATH.read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(worker._RUN_CONFIG_PATH.read_bytes()).hexdigest(),
        'installed_worker_sha256': hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest(),
        'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': platform.python_version(), 'torch': torch.__version__, 'threads': 4,
        'memory_method': 'Per-phase Linux clear_refs=5 VmHWM; includes resident checkpoint; generation includes tensor-witness overhead',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=('e01', 'e02'))
    with redirect_stdout(sys.stderr):
        result = run(parser.parse_args().family)
    print(json.dumps(result, sort_keys=True, indent=2))
