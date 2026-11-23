# Run the NMR API job consumer with the SDK and shared scientific declarations.
# The worker container owns model execution and checkpoints; this container
# receives the API credential and forwards scientific calls to the worker.
FROM docker.io/library/python:3.13.18-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72 AS base

# Download the SDK and its API dependencies from the pinned requirements,
# verifying each wheel's hash. The requirements mount is read-only; the named
# pip cache belongs to this project's builds and stays outside the image.
FROM base AS supervisor
RUN --mount=from=dependency-pins,source=provider-api.txt,target=/tmp/provider-core.txt \
    --mount=type=cache,id=chefnmr2-repro-pip,target=/root/.cache/pip \
    python -m pip download --timeout 120 --index-url https://pypi.org/simple \
    --only-binary=:all: --require-hashes \
    --dest /wheels -r /tmp/provider-core.txt

# Start a fresh runtime stage so downloaded wheels remain build inputs.
FROM base
# Install the verified wheels through a temporary read-only mount. All packages
# are supplied explicitly, so disable network access and dependency resolution.
# pip check verifies that their declared dependencies are satisfied.
RUN --network=none --mount=from=supervisor,source=/wheels,target=/tmp/sdk-wheels \
    python -m pip install --no-index --no-deps /tmp/sdk-wheels/*.whl \
    && python -m pip check

# Use temporary storage for HOME, avoid bytecode writes on the read-only runtime
# filesystem, and emit Python output immediately for container logs.
ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /opt/provider

# Prepare private mount points for the SDK lifetime lock and runtime credential.
# Credentials are supplied at deployment; no key material enters the build.
RUN install -d -m 700 -o 65532 -g 65532 /run/chefnmr2-provider /run/chefnmr2-credentials

# Both containers use these declarations to agree on scientific inputs, outputs,
# and operation identity. The provider adapter connects API jobs to that contract.
COPY --from=shared binding.py wire.py ./
COPY provider.py application.py main.py manual.py ./

# Match the worker's UID so the provider can connect to its private runner socket.
USER 65532:65532
# Start the API job consumer; Compose supplies --config for the deployment.
# The manual test client is selected explicitly by overriding this entrypoint.
ENTRYPOINT ["python", "main.py"]
