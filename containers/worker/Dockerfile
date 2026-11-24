# Bake supplies the pinned interpreter and selected, verified offline artifacts.
ARG PYTHON_BASE
FROM ${PYTHON_BASE} AS source_wheels
ENV SOURCE_DATE_EPOCH=315532800
RUN --network=none --mount=from=wheels,target=/wheels \
    python -m pip install --no-index --no-deps --require-hashes -r /wheels/build.txt
# OmegaConf has one source-only dependency. Build it separately with the locked
# setuptools input; retain its wheel hash as build evidence in the final image.
RUN --network=none --mount=from=wheels,target=/wheels \
    python -m pip wheel --no-index --no-deps --no-build-isolation --require-hashes \
        -r /wheels/sources.txt --wheel-dir /built \
    && cd /built && sha256sum *.whl > SHA256SUMS

FROM ${PYTHON_BASE}
RUN --network=none --mount=from=wheels,target=/wheels \
    --mount=from=source_wheels,source=/built,target=/built \
    python -m pip install --no-index --no-deps --require-hashes -r /wheels/install.txt \
    && cd /built && sha256sum --check SHA256SUMS \
    && python -m pip install --no-index --no-deps /built/*.whl \
    && python -m pip check
COPY --from=source_wheels /built/SHA256SUMS /opt/inference/source-wheel-sha256.txt

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
COPY --from=model binding.py wire.py ./

USER 65532:65532

# Compose supplies the socket path, working directories, shutdown budgets, and
# child command, python -m provider_core.process_worker. The SDK worker loads
# binding:construct with the deployment's model settings and owns the typed
# operation protocol and contribution lifecycle.
ENTRYPOINT ["python", "-m", "provider_core.runner"]
