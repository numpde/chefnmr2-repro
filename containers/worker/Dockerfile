# Bake supplies the pinned interpreter and selected, verified offline artifacts.
ARG PYTHON_BASE
FROM ${PYTHON_BASE}
RUN --network=none --mount=from=wheels,target=/wheels \
    --mount=from=installation,target=/installation \
    python -m pip install --no-index --no-deps --require-hashes -r /installation/install.txt \
    && python -m pip check

# Use writable temporary storage for HOME, avoid bytecode writes on the read-only
# runtime filesystem, and send Python log output without buffering.
ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# The runner starts model processes in private session scratch directories.
# Keep the copied binding and model package importable independently of that cwd.
ENV PYTHONPATH=/opt/inference
WORKDIR /opt/inference

# The SDK runner requires a private endpoint directory and lease file owned by
# its runtime user. Docker copies these into a newly created runner volume.
RUN install -d -m 700 -o 65532 -g 65532 /run/runner \
    && install -m 600 -o 65532 -g 65532 /dev/null /run/runner/.owner-session.lock

ARG UPSTREAM_PACKAGE
COPY --from=upstream ${UPSTREAM_PACKAGE}/ ./${UPSTREAM_PACKAGE}/
COPY --from=model binding.py wire.py ./

USER 65532:65532

# Compose supplies the socket path, working directories, shutdown budgets, and
# child command, python -m provider_core.process_worker. The SDK worker loads
# binding:construct with the deployment's model settings and owns the typed
# operation protocol and contribution lifecycle.
ENTRYPOINT ["python", "-m", "provider_core.runner"]
