ARG PYTHON_BASE
FROM ${PYTHON_BASE}
RUN --network=none --mount=from=wheels,target=/wheels \
    python -m pip install --no-index --no-deps --require-hashes -r /wheels/install.txt \
    && python -m pip check
COPY sanitize.py /opt/sanitize/sanitize.py
# Match the owner of the checkpoint volume initialized for curlimages/curl.
USER 100:101
ENTRYPOINT ["python", "-IB", "/opt/sanitize/sanitize.py"]
