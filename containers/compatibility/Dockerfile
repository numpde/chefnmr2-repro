# Draft: one-file Python 3.12 CPU compatibility experiment.
# The upstream checkout is the only source/build input; no wheelhouse, lockfile,
# copied package, or companion test script is required.
#
# Build (the bind mount below reads this build context without copying it):
#   docker --context <context> build --file containers/compatibility/Dockerfile \
#     --tag chefnmr2-python312:check ./chefnmr2-upstream
#
# Run with the same upstream checkout mounted read-only:
#   docker --context <context> run --rm --init --network none --read-only \
#     --cap-drop ALL --security-opt no-new-privileges --pids-limit 64 --cpus 2 \
#     --memory 2g --memory-swap 2g --tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m \
#     --mount type=bind,src="$(pwd)/chefnmr2-upstream",dst=/upstream,readonly \
#     chefnmr2-python312:check
#
# Build-time networking installs the versions declared by upstream. Transitive
# dependencies are resolved at build time, so this draft is not a hashed lock.
# No checkpoints are needed for the default import check; this is not an
# inference or accuracy qualification. Rebuild when upstream dependencies change.

FROM docker.io/library/python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    PYTHONPATH=/upstream \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# Read dependency authority directly from the mounted upstream pyproject.
# Install CPU PyTorch explicitly, then resolve the remaining declared packages.
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

# Import mounted source directly: upstream's ==3.11.* packaging restriction is
# unchanged. This tests execution on 3.12, not whether pip accepts that metadata.
COPY <<'PY' /usr/local/libexec/check-chefnmr2.py
import sys
import torch
from rdkit.Chem import rdDetermineBonds
from chefnmr2.infer import ChefNMR2, Peaks
from chefnmr2.model.build import build_model

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.version.cuda is None, 'This experiment requires CPU-only PyTorch'
print(f'Imports passed: Python {sys.version.split()[0]}, torch {torch.__version__}')
print('Checkpoint loading and inference have not been tested.')
PY

USER 65532:65532
WORKDIR /tmp
CMD ["python", "-P", "/usr/local/libexec/check-chefnmr2.py"]
