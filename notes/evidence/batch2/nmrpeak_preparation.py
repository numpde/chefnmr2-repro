"""Compare admitted and reconstructed NMRPeak tensors in an installed CPU image."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time


def resident_memory():
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    return {name: int(status[name].split()[0]) * 1024 for name in ('VmRSS', 'VmHWM')}


def measure(operation):
    # Linux resets this process's RSS high-water mark; model memory remains included
    Path('/proc/self/clear_refs').write_text('5')
    before = resident_memory()['VmRSS']
    started = time.perf_counter()
    result = operation()
    seconds = time.perf_counter() - started
    memory = resident_memory()
    return result, {'seconds': seconds, 'rss_before_bytes': before, 'peak_rss_bytes': memory['VmHWM']}


def tensor_facts(prepared):
    tensor = prepared.source_tokens
    return {
        'token_count': prepared.token_count, 'shape': list(tensor.shape),
        'dtype': str(tensor.dtype), 'device': str(tensor.device),
        'nbytes': tensor.numel() * tensor.element_size(), 'values': tensor.tolist(),
    }


def run(options):
    import torch
    from families.nmrpeak.checkpoint_file import open_verified_checkpoint
    from families.nmrpeak.runner_runtime import BART_CONFIG_PATH, DICTIONARY_PATH, NmrpeakRuntimeInputRejected
    from nmrpeak_provider.input_rejection import NmrpeakInputRejectionReason

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    family = importlib.import_module(f'models.nmrpeak_{options.family}_v1.runner.runtime')
    binding = importlib.import_module(f'nmrpeak_provider.{options.family}_binding')
    parse_input = getattr(binding, f'parse_{options.family}_runner_input')
    load = getattr(family, f'load_nmrpeak_{options.family}_runtime')
    with open_verified_checkpoint(options.checkpoint) as checkpoint:
        runtime, loading = measure(lambda: load(checkpoint))

    corpus = json.loads(Path(options.corpus).read_bytes())
    documents = {case['id']: case['inputs'][options.family] for case in corpus['cases']}
    maximal = json.loads(json.dumps(documents['small']))
    base_count = runtime.prepare(parse_input(maximal)).token_count
    maximal['h_nmr_peaks'][0]['j_values'] = '1_' * (511 - base_count)
    positions = runtime._stack._model.spec_encoder.embed_positions
    model_positions = positions.num_embeddings - positions.offset
    model_limit = json.loads(json.dumps(documents['small']))
    model_limit['h_nmr_peaks'][0]['j_values'] = '1_' * (model_positions - 2 - base_count)
    documents['model_limit'] = model_limit
    beyond_model = json.loads(json.dumps(model_limit))
    beyond_model['h_nmr_peaks'][0]['j_values'] += '1_'
    documents['beyond_model_limit'] = beyond_model
    documents['max_tokens'] = maximal
    oversized = json.loads(json.dumps(maximal))
    oversized['h_nmr_peaks'][0]['j_values'] += '1_'
    try:
        runtime.prepare(parse_input(oversized))
    except NmrpeakRuntimeInputRejected as error:
        assert error.reason is NmrpeakInputRejectionReason.TOKEN_LIMIT_EXCEEDED
        assert error.token_count == 512
        rejection = {'reason': error.reason.value, 'token_count': error.token_count}
    else:
        raise AssertionError('The 512-token proposal should be rejected before encoding')

    records = []
    try:
        for name, document in documents.items():
            captured = json.dumps(document, sort_keys=True, separators=(',', ':')).encode()
            accepted, validation = measure(lambda: runtime.prepare(parse_input(json.loads(captured))))
            admitted_facts = tensor_facts(accepted)
            dictionary = runtime._stack._dictionary
            tokens = runtime._stack.tokenize(runtime._materialize(parse_input(json.loads(captured))))
            expected = [dictionary.bos(), *(dictionary.indices[token] for token in tokens), dictionary.eos()]
            expected += [dictionary.pad()] * (-len(expected) % 8)
            assert admitted_facts['values'] == [expected]
            assert admitted_facts['dtype'] == 'torch.int64'
            assert admitted_facts['shape'] == [1, len(expected)]
            preparations = []
            for _ in range(20):
                repeated, timing = measure(lambda: runtime.prepare(parse_input(json.loads(captured))))
                assert tensor_facts(repeated) == admitted_facts
                preparations.append(timing)
                del repeated

            generations = []
            for _ in range(3 if name in ('small', 'medium', 'long') else 1):
                # Prediction uses independently rebuilt input; candidate equality is not the identity test
                repeated = runtime.prepare(parse_input(json.loads(captured)))
                assert tensor_facts(repeated) == admitted_facts
                started = time.perf_counter()
                try:
                    candidates, timing = measure(lambda: runtime.generate(repeated))
                except IndexError as error:
                    generations.append({
                        'failure_type': type(error).__name__, 'message': str(error),
                        'seconds': time.perf_counter() - started, **resident_memory(),
                    })
                    del repeated
                    break
                assert tensor_facts(accepted) == admitted_facts
                generations.append({**timing, 'candidate_count': len(candidates)})
                del repeated, candidates

            records.append({
                'case': name, 'captured_input': json.loads(captured), 'captured_bytes': len(captured),
                'captured_sha256': hashlib.sha256(captured).hexdigest(),
                'admitted_tensor': admitted_facts, 'validation': validation,
                'repreparations': preparations, 'generations': generations,
                'preparation_median_seconds': statistics.median(row['seconds'] for row in preparations),
                'generation_median_seconds': statistics.median(row['seconds'] for row in generations) if all('candidate_count' in row for row in generations) else None,
                'encoded_values_json_bytes': len(json.dumps(admitted_facts['values'], separators=(',', ':')).encode()),
            })
            print(json.dumps({'family': options.family, 'case_measurement': records[-1]}, default=str), file=sys.stderr, flush=True)
            del accepted
    finally:
        runtime._stack.close()

    return {
        'family': options.family, 'checkpoint': options.checkpoint, 'source_revision': options.source_revision,
        'image': options.image, 'load': loading, 'threads': 4, 'device': 'cpu', 'precision': 'float32',
        'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': platform.python_version(), 'kernel': platform.release(),
        'cpu_model': next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'transformers', 'provider-core')},
        'dictionary_sha256': hashlib.sha256(DICTIONARY_PATH.read_bytes()).hexdigest(),
        'bart_config_sha256': hashlib.sha256(BART_CONFIG_PATH.read_bytes()).hexdigest(),
        'tokenizer_mode': asdict(getattr(family, f'{options.family.upper()}_TOKENIZER_MODE')),
        'decode_policy': asdict(runtime._stack._decode_policy),
        'model_position_capacity': model_positions,
        'rejected_proposal': rejection, 'corrected_proposal': 'model_limit', 'cases': records,
        'memory_method': 'Linux /proc/self/clear_refs=5 per phase; VmHWM includes resident model and page allocation granularity',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('hf', 'chf'), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--image', required=True)
    options = parser.parse_args()
    with redirect_stdout(sys.stderr):
        results = run(options)
    print(json.dumps(results, sort_keys=True, indent=2, default=str))
