# Build the model container's filesystem with ChefNMR2 and the SDK runner
# installed in one Python environment. Compose supplies the mounted checkpoints
# and the runner's configuration and worker command.
FROM docker.io/library/python:3.13.18-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72 AS base

# This temporary stage reads ChefNMR2's pyproject.toml and produces two dependency
# lists. Separating Torch lets Docker reuse its large installation layer when
# other dependencies change.
FROM base AS requirements
RUN --network=none --mount=from=upstream,source=pyproject.toml,target=/tmp/pyproject.toml \
    python - <<'PY'
import tomllib
from pathlib import Path

with open("/tmp/pyproject.toml", "rb") as source:
    dependencies = tomllib.load(source)["project"]["dependencies"]
torch, = (pin for pin in dependencies if pin.startswith("torch=="))
Path("/torch.txt").write_text(torch + "\n")
Path("/inference.txt").write_text("\n".join(pin for pin in dependencies if pin != torch) + "\n")
PY

# Download the published SDK and its native dependencies, checking every wheel
# against the release hashes. The final stage installs these wheels offline.
FROM base AS supervisor
RUN --mount=from=dependency-pins,source=provider-core.txt,target=/tmp/provider-core.txt \
    --mount=type=cache,id=chefnmr2-repro-pip,target=/root/.cache/pip \
    python -m pip download --timeout 120 --index-url https://pypi.org/simple \
    --only-binary=:all: --require-hashes \
    --dest /wheels -r /tmp/provider-core.txt

# Assemble the runtime filesystem. The temporary stages supply dependency lists
# and the SDK wheels, while package installations share this stage's Python.

FROM base
# Mount the Torch dependency list from the requirements stage for this build step.
# Install its pinned version from PyTorch's CPU wheel index. Keep downloaded
# packages in a build cache so subsequent builds can reuse them.
RUN --mount=type=bind,from=requirements,source=/torch.txt,target=/tmp/torch.txt \
    --mount=type=cache,id=chefnmr2-repro-pip,target=/root/.cache/pip \
    python -m pip install --timeout 120 --index-url https://download.pytorch.org/whl/cpu -r /tmp/torch.txt

# Mount the remaining dependency list and install those packages from PyPI.
# pip check then verifies that installed package versions satisfy their declared
# dependencies. Each pip invocation allows 120 seconds of socket inactivity.
RUN --mount=type=bind,from=requirements,source=/inference.txt,target=/tmp/inference.txt \
    --mount=type=cache,id=chefnmr2-repro-pip,target=/root/.cache/pip \
    python -m pip install --timeout 120 --index-url https://pypi.org/simple -r /tmp/inference.txt \
    && python -m pip check

# Mount the verified SDK and dependency wheels from the supervisor stage and
# install them into the same Python environment. Every dependency is supplied
# explicitly, so disable network access, index lookup, and dependency resolution.
# Run pip check again to verify the combined environment after installing it.
RUN --network=none --mount=from=supervisor,source=/wheels,target=/tmp/sdk-wheels \
    python -m pip install --no-index --no-deps /tmp/sdk-wheels/*.whl \
    && python -m pip check

# Use writable temporary storage for HOME, avoid bytecode writes on the read-only
# runtime filesystem, and send Python log output without buffering.
ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# The runner starts model processes in private session scratch directories.
# Keep the copied binding and ChefNMR2 importable independently of that cwd.
ENV PYTHONPATH=/opt/inference
WORKDIR /opt/inference

# The SDK runner requires a private endpoint directory and lease file owned by
# its runtime user. Docker copies these into a newly created runner volume.
RUN install -d -m 700 -o 65532 -g 65532 /run/chefnmr2 \
    && install -m 600 -o 65532 -g 65532 /dev/null /run/chefnmr2/.owner-session.lock

COPY --from=upstream chefnmr2/ ./chefnmr2/
COPY --from=shared binding.py wire.py ./

USER 65532:65532

# Compose supplies the socket path, working directories, shutdown budgets, and
# child command, python -m provider_core.process_worker. The SDK worker loads
# binding:construct with the deployment's model settings and owns the typed
# operation protocol and contribution lifecycle.
ENTRYPOINT ["python", "-m", "provider_core.runner"]
