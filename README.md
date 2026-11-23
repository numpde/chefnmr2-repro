# ChefNMR2 repackaging for the NMR Hub

See [NMR Hub](https://nmr.dapp32.com).

Molecular structure inference from a formula and unassigned ¹H/¹³C NMR peaks,
using [ChefNMR2](https://github.com/ml-struct-bio/chefnmr2).

The provider consumes localhost NMR API jobs through the SDK and submits results
to the shared delivery service. The worker loads the model and checkpoints.
They communicate through a private runner socket; only the provider receives the
API and interpreter credentials. Both containers have networking disabled. API
traffic passes through the shared localhost Unix proxy socket. A separate SDK
proxy has external networking and admits only `api.openai.com:443` for interpretation.
It has no credentials, checkpoints, or repository mount and exposes no host port.

The localhost offering uses eight samples and seed 0. A job specification can be
JSON, for example `{"formula":"CH4","peaks":{"h":[[1.0,4,"s"]]}}`, or prose such
as `Formula CH4. Proton NMR: 1.0 ppm, integration 4H, singlet.` Attachments are not
read. Structured input goes directly to the typed decoder; malformed JSON cannot
fall back to the language model. Prose uses the SDK Interpreter with bounded
turns, corrections, retries, and an overall deadline. Both routes use the same
input type; sampling settings belong to the deployment and cannot be overridden.

API results use `chefnmr2.elucidation_report.v1`: `input` contains the actual formula and
peaks supplied to inference, and `elucidation` contains ChefNMR2's scientific output.
Prose transcription is model-generated; typed validation does not independently
prove it matches the source. Inspect the returned input. Interpreter failure is
reported separately from invalid structured input and does not start inference.

The deployment requires:

- `chefnmr2-weights_checkpoints`, containing `chefnmr2-epoch949`.
- The localhost shared delivery deployment's proxy and outbox socket volumes.
- `chefnmr2-inference_credentials`, containing `credential.json` for
  `provider:chefnmr2-fw-a` and `interpreter.key`, each owned by UID:GID
  `65532:65532` with mode `0600`. Use project-owned copies; no other provider's
  credential directory is mounted.
- The localhost API's public CA certificate, selected explicitly below.
- A captured `resolv.conf` with non-loopback IPv4 DNS servers for the SDK proxy.
  Its DNS and outbound connections bind to the proxy container's `eth0`.

```sh
export CHEFNMR2_API_CA="$PWD/secrets/localhost/api-ca.pem"
export CHEFNMR2_DNS_CONFIG="$PWD/secrets/localhost/resolv.conf"
docker --context science compose -f docker/compose/inference.yaml -f docker/compose/provider.yaml build worker provider
docker --context science compose -f docker/compose/inference.yaml -f docker/compose/provider.yaml up -d --no-build worker provider
docker --context science compose -f docker/compose/inference.yaml -f docker/compose/provider.yaml logs -f provider worker
```

`config/deployments/localhost/provider.toml` selects the endpoint, identity,
scientific settings, and execution limits. Keep credential contents outside the
repository. Stopping the provider requests retirement of its active job and
model; the worker's SDK runner remains available for another provider session.
The SDK preserves selected results through delivery backpressure. It does not
resume an existing API Attempt after the provider process restarts.

Input routing follows NMRPeak's structured/prose split. SECS supplies the precedent
for extracting observations without inventing missing evidence; Magnet shares SDK
endpoint retry state across jobs; NMRNet keeps interpreter egress in a separate
proxy. ChefNMR2 reuses the SDK's current `Interpreter`, `ProseService`, chat carrier,
and proxy rather than copying those providers' conversation or transport code.

For manual model testing, use the separate client in the provider image:

```sh
docker --context science compose -f docker/compose/inference.yaml --profile manual build worker manual
docker --context science compose -f docker/compose/inference.yaml up -d --no-build worker
printf '%s\n' '{"requests":[{"formula":"CH4","peaks":{"h":[[1.0,4,"s"]]}}]}' |
  docker --context science compose -f docker/compose/inference.yaml run --rm -T manual
```

Stop the API provider before using the manual client: both use the same model
runner. The manual client loads the model once for its input stream and writes
one JSON batch reply per line. Closing stdin retires its model session.

The worker accepts `{"requests": [...]}` and returns `{"replies": [...]}`. Entries
run sequentially, and reply positions match request positions. A model exception
produces an `exception_type`/`message` diagnostic for that entry; later entries
still run. A lost worker or connection can lose the entire reply and does not
authorize replay. SDK byte and execution limits apply to the whole batch. The API
service still processes one job at a time, sending a one-entry batch; concurrent
discovery and automatic batch assembly are not implemented.

The SDK dependency files in `requirements/` are generated for CPython 3.13 on
Linux/amd64. Each package has one selected wheel URL and hash. To update them,
run the released SDK's `wheelhouse-tool locate` and `requirements` commands inside
the pinned Python image, using the release's native and API install locks as the
source authority. Regenerate both files when changing the SDK, Python, or platform.
