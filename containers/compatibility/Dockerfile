# Check whether ChefNMR2 can load a model and predict structures on Python 3.12
# using the CPU. Building installs dependencies from chefnmr2-upstream. Running
# uses that source and one checkpoint/config pair, both mounted read-only.
#
# Build from the repository root. Docker reads upstream's dependency list without
# copying its source into the image. This step needs internet access:
#   docker --context <context> build --file containers/compatibility/Dockerfile \
#     --tag chefnmr2-python312:check ./chefnmr2-upstream
#
# Complete checkpoint acquisition and verification before running this check.
# Run from the repository root, selecting one epoch directory explicitly. This
# command disables container networking, limits resources and removes the
# container when it exits. Source and image files stay read-only; /tmp provides
# temporary space that disappears with the container:
#   docker --context <context> run --rm --init --network none --read-only \
#     --cap-drop ALL --security-opt no-new-privileges --pids-limit 64 --cpus 2 \
#     --memory 8g --memory-swap 8g --tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m \
#     --mount type=bind,src="$(pwd)/chefnmr2-upstream",dst=/upstream,readonly \
#     --mount type=volume,src=chefnmr2-weights_checkpoints,dst=/checkpoints,readonly \
#     chefnmr2-python312:check /checkpoints/chefnmr2-epoch949
#
# Upstream pins its direct dependencies. Pip chooses the packages those depend
# on during installation, so fresh installs may produce different environments.
# Rebuild when upstream's dependency list changes. Passing this check establishes
# that one model loads and produces valid structures with the input formula.
# Whether those structures include morphine is reported separately: one model
# and ten samples do not reproduce the upstream ensemble's accuracy claim.
# Full-ensemble and adaptive sampling behavior, accuracy across other molecules,
# and GPU support need separate tests. The run-command limits above apply to
# this single-model check.

# Pin the Python release and base-image contents so rebuilding uses the same base.
FROM docker.io/library/python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

# Keep source imports on the mounted checkout without writing bytecode into it.
# Print output immediately and limit numerical-library threads for this CPU check.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    PYTHONPATH=/upstream \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# Read package versions from upstream's pyproject.toml so this file does not
# maintain a second version list. Choose PyTorch's CPU build first, then install
# the other packages from PyPI. Fail the build if package requirements conflict.
RUN --mount=type=bind,target=/upstream,readonly python - <<'PY'
import subprocess
import sys
import tomllib
from pathlib import Path

project = tomllib.loads(Path('/upstream/pyproject.toml').read_text())['project']
dependencies = project['dependencies']
torch_requirement = next(item for item in dependencies if item.startswith('torch=='))
remaining = [item for item in dependencies if item != torch_requirement]
pip = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir']
subprocess.run(pip + ['--index-url', 'https://download.pytorch.org/whl/cpu', torch_requirement], check=True)
subprocess.run(pip + ['--index-url', 'https://pypi.org/simple', *remaining], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
PY

# Upstream declares Python 3.11 only, which blocks a normal package installation
# on 3.12. Import its mounted source directly to test whether the model can run.
# This leaves upstream unchanged and does not establish that packaging supports 3.12.
# Keep the check inside the image so the mounted checkout needs no added files.
COPY <<'PY' /usr/local/libexec/check-chefnmr2.py
import argparse
import json
import sys
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from chefnmr2.chem.match import match_smiles
from chefnmr2.infer import ChefNMR2, Peaks


def main():
    """Run the upstream morphine example through one real checkpoint on CPU."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('checkpoint', type=Path, help='Directory containing model.pt and config.yaml')
    checkpoint = parser.parse_args().checkpoint
    for name in ('model.pt', 'config.yaml'):
        if not (checkpoint / name).is_file():
            parser.error(f'Cannot test inference: {checkpoint / name} must be a completed checkpoint file')

    assert sys.version_info[:2] == (3, 12), sys.version
    assert torch.version.cuda is None, 'This check requires CPU-only PyTorch'
    print(f'Loading {checkpoint} on Python {sys.version.split()[0]}, torch {torch.__version__}', flush=True)
    members = [(checkpoint.name, checkpoint / 'model.pt', checkpoint / 'config.yaml')]
    model = ChefNMR2(members, device='cpu', recon_workers=2)
    try:
        # Use the upstream README's peaks and reference molecule. A fixed budget
        # works with one model; upstream's adaptive policy requires all ten.
        peaks = Peaks(
            c=[145.2, 138.0, 133.2, 130.9, 128.5, 126.6, 120.2, 116.8, 91.9,
               66.6, 59.1, 46.6, 43.3, 43.1, 40.7, 35.6, 20.7],
            h=[(6.63, 1, 'd'), (6.5, 1, 'd'), (5.67, 1, 'dddd'), (5.28, 1, 'ddd'),
               (4.89, 1, 'dd'), (4.2, 1, 'ddd'), (3.41, 1, 'dd'), (3.03, 1, 'd'),
               (2.73, 1, 'm'), (2.66, 1, 'dd'), (2.47, 3, 's'), (2.45, 1, 'td'),
               (2.33, 1, 'dd'), (2.12, 1, 'td'), (1.88, 1, 'ddd')],
        )
        print('Sampling ten structures and reconstructing candidates', flush=True)
        formula = 'C17H19NO3'
        result = model.elucidate(peaks, formula=formula, n_samples=10, seed=0)
        reported = 'CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5'
        matched = any(match_smiles(reported, candidate['smiles']) for candidate in result['candidates'][:10])
        print(json.dumps({'python': sys.version, 'torch': torch.__version__,
                          'morphine_in_top10': matched, 'result': result}, indent=2), flush=True)
        assert result['ensemble']['n_checkpoints'] == 1, 'The check must use exactly one checkpoint'
        assert result['n_samples'] == 10, 'The check must complete ten samples'
        assert model.recon_pool is not None, 'The reconstruction worker pool failed during inference'
        assert result['candidates'], 'Inference check failed: sampling produced no candidate structures'
        for candidate in result['candidates']:
            molecule = Chem.MolFromSmiles(candidate['smiles'])
            assert molecule is not None, f"Inference returned an unreadable structure: {candidate['smiles']}"
            actual_formula = rdMolDescriptors.CalcMolFormula(molecule)
            assert actual_formula == formula, f'Inference returned formula {actual_formula}; input was {formula}'
        print('PASS: Python 3.12 loaded one checkpoint and produced valid structures with the input formula', flush=True)
    finally:
        if model.recon_pool is not None:
            model.recon_pool.shutdown()


if __name__ == '__main__':
    main()
PY

# Read model/source inputs as an unprivileged user; scratch space stays in /tmp.
USER 65532:65532
WORKDIR /tmp
ENTRYPOINT ["python", "-P", "/usr/local/libexec/check-chefnmr2.py"]
