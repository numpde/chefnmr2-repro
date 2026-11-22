# Check whether ChefNMR2's core modules can load on Python 3.12 with
# CPU-only libraries. Building installs dependencies; running performs the check.
# The only local input is chefnmr2-upstream, mounted read-only at both steps.
#
# Build from the repository root. Docker reads upstream's dependency list without
# copying its source into the image. This step needs internet access:
#   docker --context <context> build --file containers/compatibility/Dockerfile \
#     --tag chefnmr2-python312:check ./chefnmr2-upstream
#
# Run from the repository root with the same upstream checkout. This command
# disables container networking, limits resources and removes the container when
# it exits. Source and image files stay read-only; /tmp provides temporary space
# that disappears with the container:
#   docker --context <context> run --rm --init --network none --read-only \
#     --cap-drop ALL --security-opt no-new-privileges --pids-limit 64 --cpus 2 \
#     --memory 2g --memory-swap 2g --tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m \
#     --mount type=bind,src="$(pwd)/chefnmr2-upstream",dst=/upstream,readonly \
#     chefnmr2-python312:check
#
# Upstream pins its direct dependencies. Pip chooses the packages those depend
# on during installation, so fresh installs may produce different environments.
# Rebuild when upstream's dependency list changes. Passing this check establishes
# that the imports work; checkpoint loading and predictions need separate tests.

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
# on 3.12. Import its mounted source directly to test whether the code can load.
# This leaves upstream unchanged and does not establish that packaging supports 3.12.
# Keep the check inside the image so the mounted checkout needs no added files.
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

# The check only needs to read source and installed libraries, so run without root.
USER 65532:65532
WORKDIR /tmp
CMD ["python", "-P", "/usr/local/libexec/check-chefnmr2.py"]
