FROM docker.io/library/python:3.13.18-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72 AS base

# Separate Torch's pin so edits to other dependencies can reuse the CPU Torch layer.
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

FROM base
RUN --mount=type=bind,from=requirements,source=/torch.txt,target=/tmp/torch.txt \
    --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --timeout 120 --index-url https://download.pytorch.org/whl/cpu -r /tmp/torch.txt

RUN --mount=type=bind,from=requirements,source=/inference.txt,target=/tmp/inference.txt \
    --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --timeout 120 --index-url https://pypi.org/simple -r /tmp/inference.txt \
    && python -m pip check

ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /opt/inference
RUN install -d -o 65532 -g 65532 /run/chefnmr2
COPY --from=upstream chefnmr2/ ./chefnmr2/
COPY worker.py wire.py ./
USER 65532:65532
ENTRYPOINT ["python", "worker.py"]
