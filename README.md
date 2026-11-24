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
- The localhost API's public CA certificate, stored at `secrets/localhost/api-ca.pem`.
- A captured `resolv.conf` with non-loopback IPv4 DNS servers for the SDK proxy.
  Store it at `secrets/localhost/resolv.conf`. Its DNS and outbound connections bind to the proxy container's `eth0`.

```sh
make build PROFILE=worker-cpu DOCKER_CONTEXT=science WHEELHOUSE="$WHEELHOUSE"
make build PROFILE=provider DOCKER_CONTEXT=science WHEELHOUSE="$WHEELHOUSE"
make config DEPLOYMENT=localhost DOCKER_CONTEXT=science
make status DEPLOYMENT=localhost DOCKER_CONTEXT=science
make logs DEPLOYMENT=localhost DOCKER_CONTEXT=science
```

`deployments/localhost/provider.toml` selects the endpoint, identity,
scientific settings, and execution limits. Keep credential contents outside the
repository. Stopping the provider requests retirement of its active job and
model; the worker's SDK runner remains available for another provider session.
The SDK preserves selected results through delivery backpressure. It does not
resume an existing API Attempt after the provider process restarts.

Interpreter instructions live in `containers/provider/prompts/`; public
provider and offering descriptions live in `containers/provider/about/`.
The offering template uses the deployment's sample count and seed. These files
are packaged in the provider image; rebuild it after editing them.

Input routing follows NMRPeak's structured/prose split. SECS supplies the precedent
for extracting observations without inventing missing evidence; Magnet shares SDK
endpoint retry state across jobs; NMRNet keeps interpreter egress in a separate
proxy. ChefNMR2 reuses the SDK's current `Interpreter`, `ProseService`, chat carrier,
and proxy rather than copying those providers' conversation or transport code.

For manual model testing, use the separate client in the provider image:

```sh
printf '%s\n' '{"requests":[{"formula":"CH4","peaks":{"h":[[1.0,4,"s"]]}}]}' |
  make manual DEPLOYMENT=localhost DOCKER_CONTEXT=science
```

Manual runs use a unique Compose project and private runner state, sharing only
the checkpoint volume read-only. API services and credentials are unnecessary.
The manual client loads the model once for its input stream and writes
one JSON batch reply per line. Closing stdin retires its model session.

The worker accepts `{"requests": [...]}` and returns `{"replies": [...]}`. Entries
run sequentially, and reply positions match request positions. A model exception
produces an `exception_type`/`message` diagnostic for that entry; later entries
still run. A lost worker or connection can lose the entire reply and does not
authorize replay. SDK byte and execution limits apply to the whole batch.

The API service admits up to four jobs and prepares up to two concurrently.
It collects at most four prepared, permitted jobs per call, waiting up to 50 ms
from the first activation. Deployment `[batch]` and `[service.limits]` settings
bound collection and retained work. Slow delivery retains its Attempt capacity
without immediately blocking other jobs.

Cancelling a queued job removes it. After dispatch, its sequential entry may still
run; the SDK keeps its participation until the batch settles and suppresses its
publication. Sibling jobs retain their own results. A malformed or lost whole
reply stops the dispatcher and retires the shared host, with no automatic replay.
Unconfirmed retirement retains ownership; shutdown does not claim success.

Build inputs live in `builds/<profile>/requirements.in`. The generated
`requirements.lock` is a target-specific PEP 751 artifact closure with URLs,
hashes, the resolver proof and an input identity. `builds/sdk.constraints.txt`
selects the common SDK release. Bake owns pinned interpreters and build contexts;
Compose uses built images and never resolves packages.

Select an existing external package pool explicitly with `WHEELHOUSE`. Prepare
SDK tooling with `make sdk/acquire DOCKER_CONTEXT=science WHEELHOUSE="$WHEELHOUSE"`.
`make lock PROFILE=provider DOCKER_CONTEXT=science WHEELHOUSE="$WHEELHOUSE"`
resolves against cached artifacts and reads package-index metadata to select
published URLs; it does not download package archives. Repeat for `worker-cpu`.
The baseline constraints preserve all installed CPU package versions captured
before this refactor. Change them deliberately when qualifying a new CPU environment.

`make acquire PROFILE=<profile> DOCKER_CONTEXT=science WHEELHOUSE="$WHEELHOUSE"`
is the separate networked package acquisition step. Builds require every selected
artifact and all pinned base/tool images locally, fail on stale locks or missing
inputs, and run without build networking. The worker builds the declared antlr
source archive separately with locked setuptools and retains its resulting wheel
hash at `/opt/inference/source-wheel-sha256.txt`.

Run `make check DOCKER_CONTEXT=science` for the provider's offline behavior tests.

`compose/` owns common service topology and security defaults. Each deployment
owns resource values, TOML and selected artifact identities. `images.env` is read
by Compose itself; its two immutable image IDs serve provider/proxy/manual and
the worker respectively. Builds print IDs; selecting artifacts for qualification
and promotion is an explicit edit of that deployment file.

The localhost API overlay is separate from its science overlay because Compose
interpolates and validates all loaded config definitions, including inactive
services. Manual operations never load API credentials, CA or DNS definitions.
`make manual/config DEPLOYMENT=localhost DOCKER_CONTEXT=science` renders that
science-only composition. Paths resolve from the repository root even when Make
is invoked with `-C` from another directory. The live project name remains
`chefnmr2-inference`; isolated manual projects clean up their own runner volumes.

Checkpoint acquisition and verification use `make weights/download` and
`make weights/verify`, both with `DOCKER_CONTEXT=science`. Acquisition requires
explicit `PROXY_URL=none` or a container-reachable proxy URL. The pinned curl
image must already be present; verification never pulls or downloads anything.
