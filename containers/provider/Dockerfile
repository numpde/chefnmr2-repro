# The provider and manual client share one science-free SDK/API environment.
ARG PYTHON_BASE
FROM ${PYTHON_BASE}
RUN --network=none --mount=from=wheels,target=/wheels \
    --mount=from=installation,target=/installation \
    python -m pip install --no-index --no-deps --require-hashes -r /installation/install.txt \
    && python -m pip check

# Use temporary storage for HOME, avoid bytecode writes on the read-only runtime
# filesystem, and emit Python output immediately for container logs.
ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /opt/provider

# Prepare private mount points for the SDK lifetime lock and runtime credential.
# Credentials are supplied at deployment; no key material enters the build.
RUN install -d -m 700 -o 65532 -g 65532 /run/chefnmr2-provider /run/chefnmr2-credentials /run/chefnmr2-interpreter

# Both containers use these declarations to agree on scientific inputs, outputs,
# and operation identity. The provider adapter connects API jobs to that contract.
COPY --from=model binding.py wire.py ./
COPY provider.py application.py job_input.py main.py manual.py ./
# Keep interpreter instructions and public descriptions beside their loaders.
COPY prompts/ ./prompts/
COPY about/ ./about/

# Match the worker's UID so the provider can connect to its private runner socket.
USER 65532:65532
# Start the API job consumer; Compose supplies --config for the deployment.
# The manual test client is selected explicitly by overriding this entrypoint.
ENTRYPOINT ["python", "main.py"]
