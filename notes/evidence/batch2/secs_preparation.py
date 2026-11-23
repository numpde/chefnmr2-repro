"""Measure real SEC-S preparation/embedding and a bounded static-candidate CPU search."""

from base64 import b64decode, b64encode
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import subprocess
from tempfile import TemporaryDirectory
import time
from zipfile import ZipFile, ZipInfo


def memory():
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    return {name: int(status[name].split()[0]) * 1024 for name in ('VmRSS', 'VmHWM')}


def measure(operation):
    Path('/proc/self/clear_refs').write_text('5')
    before = memory()['VmRSS']
    started = time.perf_counter()
    result = operation()
    return result, {'seconds': time.perf_counter() - started, 'rss_before_bytes': before, **memory()}


def run():
    subprocess.run([sys.executable, '-P', '/verify_cache.py', '--verify-only', '--lock', '/input/molformer.lock.toml', '--output', '/cache'], check=True, stdout=sys.stderr)
    import numpy as np
    import torch
    from secs_inference import SecsInference
    from secs_inference.provider.input_adapter import InputAdapter
    from secs_inference.provider.input_operations import SourceRef
    from secs_inference.provider.source_access import SourceAccess, InputReadError
    from secs_inference.elucidation import SecsElucidator, canonical_formula
    from secs.elucidation import GraphGAOptimizer, StaticCandidateSource

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    prepared_cases = []
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive = root / 'processed.zip'
        with ZipFile(archive, 'w') as output:
            for member in ('1r', 'procs'):
                output.writestr(ZipInfo('pdata/1/' + member, (1980, 1, 1, 0, 0, 0)), (Path('/fixtures/bruker/F3697/1/pdata/1') / member).read_bytes())
        cases = {
            'jcamp': Path('/fixtures/jcamp/4-chlorobenzylamine/4-chlorobenzylamine.jdx').read_bytes(),
            'bruker_processed': archive.read_bytes(),
        }
        for name, captured in cases.items():
            access = SourceAccess({'upload:fixture': captured}, root)
            adapter = InputAdapter(token_key=b'fixed-local-measurement-key-only')
            attempt = 'batch2-fixture'
            inventory, inspection = measure(lambda: adapter.discover(access, attempt, SourceRef('upload:fixture', None)))
            choices = [entry for entry in inventory['representations'] if entry['kind'] == 'spectrum' and entry['metadata'].get('nucleus') == '1H']
            assert len(choices) == 1, choices
            formula = 'C7H8ClN'
            selection = {'representation_id': choices[0]['id'], 'formula': formula,
                         'formula_evidence': {'kind': 'job_specification', 'quote': formula},
                         'processing': 'as_stored', 'explanation': 'Synthetic measurement selection'}
            timings = []
            for _ in range(5):
                prepared, timing = measure(lambda: adapter.prepare(access, attempt, selection))
                timings.append(timing)
                assert prepared.values.shape == (10000,) and prepared.values.dtype == np.float32
                assert np.isfinite(prepared.values).all()
                if len(timings) == 1:
                    expected = prepared.values.copy()
                np.testing.assert_array_equal(expected, prepared.values)
            metadata = {
                'representation_id': prepared.representation_id, 'formula': canonical_formula(formula),
                'sources': [asdict(source) for source in prepared.sources],
                'from_fid': prepared.from_fid, 'magnitude': prepared.magnitude,
                'dtype': str(prepared.values.dtype), 'shape': list(prepared.values.shape),
            }
            transferable = json.dumps({**metadata, 'values_base64': b64encode(prepared.values.tobytes()).decode()}, separators=(',', ':')).encode()
            np.testing.assert_array_equal(np.frombuffer(b64decode(json.loads(transferable)['values_base64']), dtype=np.float32), expected)
            float_document = {**metadata, 'source_sha256': hashlib.sha256(captured).hexdigest(),
                              'grid': {'ppm_from': -2.0, 'ppm_to': 10.0, 'points': 10000},
                              'values': tuple(float(value) for value in prepared.values)}
            transfers = []
            for _ in range(5):
                encoded, encoding = measure(lambda: json.dumps(float_document, allow_nan=False, separators=(',', ':')).encode())
                decoded, decoding = measure(lambda: json.loads(encoded))
                restored, reconstruction = measure(lambda: np.asarray(decoded.pop('values'), dtype=np.float32))
                np.testing.assert_array_equal(prepared.values.view(np.uint32), restored.view(np.uint32))
                assert decoded == {key: value for key, value in float_document.items() if key != 'values'}
                transfers.append({'encode': encoding, 'decode': decoding, 'restore_float32': reconstruction})
            assert len(encoded) + 4 < 2 * 1024 * 1024
            replacement = SourceAccess({'upload:fixture': b'replacement source'}, root)
            try:
                adapter.prepare(replacement, attempt, selection)
            except InputReadError as error:
                replacement_rejection = str(error)
            else:
                raise AssertionError('Changed source bytes should fail the admitted representation identity')
            prepared.values.flags.writeable = False
            prepared_cases.append((prepared, {
                'case': name, 'source_sha256': hashlib.sha256(captured).hexdigest(), 'source_bytes': len(captured),
                'inspection': inspection, 'preparations': timings, 'preparation_median_seconds': statistics.median(row['seconds'] for row in timings),
                'metadata': metadata, 'values_sha256': hashlib.sha256(prepared.values.tobytes()).hexdigest(),
                'raw_array_bytes': prepared.values.nbytes, 'candidate_transfer_json_bytes': len(transferable),
                'changed_source_rejected': replacement_rejection,
                'ordinary_json_body_bytes': len(encoded), 'ordinary_json_transfer': transfers,
                'ordinary_json_values_bit_exact': True,
            }))

    inference, loading = measure(lambda: SecsInference.load('/checkpoint/manifest.json', molformer_lock='/input/molformer.lock.toml', device='cpu', compute_dtype=torch.float32, smiles_batch_size=8))
    records = []
    for prepared, record in prepared_cases:
        timings = []
        for _ in range(3):
            embedding, timing = measure(lambda: inference.embed_spectrum(prepared.values))
            assert embedding.shape == (1024,) and np.isfinite(embedding).all()
            timings.append(timing)
        record['real_checkpoint_embeddings'] = timings
        record['embedding_median_seconds'] = statistics.median(row['seconds'] for row in timings)
        records.append(record)
    candidates = ['CCO', 'COC', 'CC', 'CCC', 'CCN', 'CC=O', 'C=C', 'CO']
    elucidator = SecsElucidator(inference, StaticCandidateSource(candidates), GraphGAOptimizer(population_size=8, offspring_size=8, max_generations=1, seed=42), initial_population_size=8)
    search, search_timing = measure(lambda: elucidator.elucidate(prepared_cases[0][0].values, 'C2H6O'))
    return {
        'cases': records, 'load': loading, 'bounded_search': {**search_timing, 'initial_candidates': candidates,
            'formula': 'C2H6O', 'population': 8, 'offspring': 8, 'generation_limit': 1, 'seed': 42,
            'generations': search.optimization.generations if search.optimization else None},
        'production_index_present': all(Path('/checkpoint/candidates', name).is_file() for name in ('smiles.faiss', 'candidates.parquet')),
        'checkpoint_manifest_sha256': hashlib.sha256(Path('/checkpoint/manifest.json').read_bytes()).hexdigest(),
        'molformer_lock_sha256': hashlib.sha256(Path('/input/molformer.lock.toml').read_bytes()).hexdigest(),
        'python': platform.python_version(), 'torch': torch.__version__, 'threads': 4,
        'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'memory_method': 'Linux clear_refs=5 per phase; process VmHWM includes loaded components, not allocation size',
    }


if __name__ == '__main__':
    with redirect_stdout(sys.stderr):
        result = run()
    print(json.dumps(result, sort_keys=True, indent=2))
