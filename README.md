# ChefNMR2 repackaging for the NMR Hub

See [NMR Hub](https://nmr.dapp32.com).

Molecular structure inference from a formula and unassigned ¹H/¹³C NMR peaks,
using [ChefNMR2](https://github.com/ml-struct-bio/chefnmr2).

The provider consumes localhost NMR API jobs through the SDK and submits results
to the shared delivery service. The worker loads the model and checkpoints.
They communicate through a private runner socket; only the provider receives the
API credential. Both containers have networking disabled. API traffic passes
through the shared localhost Unix proxy socket.

The localhost offering uses eight samples and seed 0. Its job specification is
JSON text, for example `{"formula":"CH4","peaks":{"h":[[1.0,4,"s"]]}}`.
Sampling settings belong to the deployment and cannot be overridden by a job.

The deployment requires:

- `chefnmr2-weights_checkpoints`, containing `chefnmr2-epoch949`.
- The localhost shared delivery deployment's proxy and outbox socket volumes.
- `chefnmr2-inference_credentials`, containing `credential.json` for
  `provider:chefnmr2-fw-a`, owned by UID:GID `65532:65532` with mode `0600`.
- The localhost API's public CA certificate, selected explicitly below.

```sh
export CHEFNMR2_API_CA="$PWD/secrets/localhost/api-ca.pem"
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

For manual model testing, use the separate client in the provider image:

```sh
docker --context science compose -f docker/compose/inference.yaml --profile manual build worker manual
docker --context science compose -f docker/compose/inference.yaml up -d --no-build worker
printf '%s\n' '{"formula":"CH4","peaks":{"h":[[1.0,4,"s"]]}}' |
  docker --context science compose -f docker/compose/inference.yaml run --rm -T manual
```

Stop the API provider before using the manual client: both use the same model
runner. The manual client loads the model once for its input stream and writes
one JSON result per line. Closing stdin retires its model session.

The SDK dependency files in `requirements/` are generated for CPython 3.13 on
Linux/amd64. Each package has one selected wheel URL and hash. To update them,
run the released SDK's `wheelhouse-tool locate` and `requirements` commands inside
the pinned Python image, using the release's native and API install locks as the
source authority. Regenerate both files when changing the SDK, Python, or platform.
