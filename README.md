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
- `chefnmr2-inference_credentials`, containing `credential.json` and `interpreter.key`,
  each owned by UID:GID `65532:65532` with mode `0600`. The API credential must
  match the identity selected in the deployment's `provider.toml`. Use
  project-owned copies; no other provider's credential directory is mounted.
- The localhost API's public CA certificate, stored at `secrets/localhost/api-ca.pem`.
- A captured `resolv.conf` with non-loopback IPv4 DNS servers for the SDK proxy.
  Store it at `secrets/localhost/resolv.conf`. Its DNS and outbound connections bind to the proxy container's `eth0`.

```sh
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

Tests are grouped by boundary, then by ordinary behavior (`normative`) or
failure and interruption behavior (`adversarial`):

| Directory | Command | Environment |
| --- | --- | --- |
| `tests/operator/` | `make test/operator` | Host shell with a fake Docker executable |
| `tests/unit/` | `make test/unit DOCKER_CONTEXT=science` | Cached worker image, synthetic checkpoints |
| `tests/provider/` | `make test/provider DOCKER_CONTEXT=science` | Cached provider image, real SDK with controlled science and HTTP |

Append `/normative` or `/adversarial` to run one category. Each boundary keeps
its reusable fixtures beside the category directories. `tests/model/` retains
the public scientific cases and checker used by `make qualify`.

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

With SDK 0.7.7, the localhost policy requests stopping after 60 seconds for the
whole execution, including initialization and every batch entry. The first stop
request, including cancellation, starts a 10-second grace. If work remains
unfinished, the SDK retires its worker using the separate cleanup allowances.
Interrupted work is never replayed.

## Setup from a new checkout

The host needs Git, Bash, Python 3.11 or newer, and Docker with Compose and
Buildx. Configure a local Docker context first; the commands below run as the
non-root checkout owner. GitHub SSH access is needed for the pinned submodule.

```bash
git clone git@github.com:numpde/chefnmr2-repro.git
cd chefnmr2-repro
make upstream/init
```

The checkout includes dependency locks and the generated acquisition bootstrap.
No separate SDK checkout or host installation of the SDK is needed.

Select your local Docker context, an external package directory and an artifact
download route. Replace `local` with your configured context name. An existing
package pool can be reused across checkouts and profiles.

```bash
CONTEXT=local
WHEELHOUSE="$HOME/.cache/chefnmr2/wheelhouse"
PROXY_URL=none
WORKER_PROFILE=worker-cpu
mkdir -p "$WHEELHOUSE"
```

`PROXY_URL=none` selects direct access. To use an upstream proxy, set it to your
HTTP, SOCKS5 or SOCKS5h URL, including its port. SOCKS5h delegates destination DNS
to the proxy. The examples pass values explicitly: Make ignores ambient input
settings and prompts when a required command-line input is omitted.

Pull the Python base images selected by Bake before acquisition or building.
This reads their pinned identities from the repository rather than repeating
image versions here. Run the examples in Bash; the subshell stops on failure.

```bash
(
  set -euo pipefail
  docker --context "$CONTEXT" buildx bake \
      --print provider "$WORKER_PROFILE" weights-sanitize |
    python3 -c 'import json,sys; print(*sorted({
        t["args"]["PYTHON_BASE"]
        for t in json.load(sys.stdin)["target"].values()
    }), sep="\n")' |
    while IFS= read -r image; do
      docker --context "$CONTEXT" pull "$image"
    done
)
```

Image pulls use the Docker daemon's configured proxy independently of
`PROXY_URL`. Artifact acquisition uses the shared SDK runner: only its temporary
gateway has host networking. Download tools share an isolated loopback HTTP
relay, which reaches that gateway through a private Unix socket. A failed proxy
connection never falls back to direct access. A loopback upstream proxy on a
rootless engine requires Docker 29.5+ with RootlessKit's detached network namespace.

Acquire the pinned SDK tooling, then each profile's dependencies, and build:

```bash
(
  set -euo pipefail
  make sdk/acquire DOCKER_CONTEXT="$CONTEXT" \
      WHEELHOUSE="$WHEELHOUSE" PROXY_URL="$PROXY_URL"

  for profile in provider "$WORKER_PROFILE" weights-sanitize; do
    make acquire PROFILE="$profile" DOCKER_CONTEXT="$CONTEXT" \
        WHEELHOUSE="$WHEELHOUSE" PROXY_URL="$PROXY_URL"
    make build PROFILE="$profile" DOCKER_CONTEXT="$CONTEXT" \
        WHEELHOUSE="$WHEELHOUSE"
  done
)
```

`sdk/acquire` places the SDK wheel, wheelhouse tool, release locks and checksum
manifest in the pool. `acquire` downloads missing profile artifacts and verifies
the locked hashes. Builds use those artifacts offline and print the resulting
image IDs. They fail on missing inputs or stale locks.

A normal new checkout uses the checked-in locks; it does not run `make lock`.
For dependency changes, `builds/<profile>/requirements.in` and its constraints
are the inputs. `make lock` resolves against cached packages and reads index
metadata to select published URLs; it needs the pinned uv image from Bake too.
The generated `requirements.lock` records target-specific URLs, hashes, resolver
output and input identity. `builds/sdk.constraints.txt` selects the SDK release.

Run `make check DOCKER_CONTEXT="$CONTEXT"` for the provider's offline behavior tests.
Prepare checkpoints next, then select and qualify deployment images as described
below. API deployment also requires the credentials and services listed above.

## Artifact storage and copies

The selected `WHEELHOUSE` is the persistent package pool. Downloads write there;
creating another checkout does not automatically create another pool. Builds
currently make actual file copies of the selected artifacts, not hard links.
This gives BuildKit a separate input directory containing only the selected
artifacts and generated installation lists. Each copy is hashed against the lock;
later changes to the pool do not change those staged bytes. The cost is temporary
disk space and copying I/O before BuildKit imports the context. Hard links would
share mutable file contents and require the same filesystem; this implementation
uses ordinary copies so the pool and temporary directory can live on different
filesystems.

| Location | Contents | Lifetime |
| --- | --- | --- |
| `$WHEELHOUSE` | Downloaded wheels, source archives and SDK tooling, reused across selected profiles | Persistent until explicitly removed |
| Temporary `chefnmr2-inputs-*/wheels/` directory, normally under `/tmp` | Copies of the selected profile's artifacts, rehashed before building | Removed when the build command unwinds normally, including ordinary failures; abrupt process termination can leave residue |
| Docker/BuildKit storage | Build inputs, intermediate stages and cached layers; it may retain additional artifact copies | Managed by Docker's cache lifecycle |
| Final images | Installed Python packages and the Python base image's bundled pip wheel; selected input wheels are mounted during installation rather than copied into the final filesystem | Retained with the images |
| `make/bootstrap/acquisition/` | Generated acquisition source, bundled dependency source and a pinned pip wheel | Included in every checkout |
| Docker volume `chefnmr2-weights_checkpoints` | Model checkpoints and preparation receipts | Persistent, separate from the checkout and wheelhouse |

The bootstrap is generated from the shared SDK source and verified before use;
its provenance is recorded in [make/bootstrap/README.md](make/bootstrap/README.md).
It is a checked-in copy, not a separately maintained implementation.

The worker builds the source-only ANTLR dependency into a wheel in an intermediate
Docker stage. That wheel is not written back to the host pool. Its resulting hash
is retained in the final image at `/opt/inference/source-wheel-sha256.txt`.

## Checkpoints and deployment

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

Prepare the checkpoints after building the sanitizer image:

```bash
make weights/download DOCKER_CONTEXT="$CONTEXT" PROXY_URL="$PROXY_URL"
make weights/sanitize DOCKER_CONTEXT="$CONTEXT"
make weights/verify DOCKER_CONTEXT="$CONTEXT"
```

Download acquires all ten pinned checkpoints into `chefnmr2-weights_checkpoints`.
The pinned curl image is pulled if missing; the shared acquisition runner uses
the provider Python base image pulled earlier. Verification and sanitization
have networking disabled and never pull images. The sanitizer's CPU environment
needs neither an inference image nor a GPU.

`containers/weights-sanitize/sanitize.py` owns preparation and marker publication.
Compose selects extraction, precision and companion files.
Chef's released inference files need no rewrite: sanitization checks
their pinned hashes and restricted loading, then publishes
`model.pt.sanitized.json` beside each checkpoint. The receipt covers both
`model.pt` and `config.yaml`.

Model initialization requires receipts only for the selected checkpoints.
`model/binding.py` hashes the selected weights and config against their receipt,
checks its generation, then passes those same open files to the loader. Managed
writers remove the receipt before replacing covered files and publish a fresh
receipt last. Unselected checkpoints are not opened. `weights/verify` checks
prepared hashes; `weights/download` verifies and restores pinned source bytes.
After an interrupted conversion leaves an unmarked output, restore the source
before sanitizing again. Qualification uses the same model initialization gate.
Upstream source remains unchanged.

Qualify the exact selected artifacts with
`make qualify DEPLOYMENT=localhost DOCKER_CONTEXT=science`. It checks checkpoint
hashes, cold/warm ten-draw upstream morphine inference, scientific validity,
ordered eight-draw SDK batches, and session/project retirement. The two budgets
are reported separately. It records images, versions, configuration identity,
checkpoint hashes and results under `.work/qualification/`. Qualification shares
checkpoints read-only and needs neither API access nor credentials.

Successful qualification automatically keeps the three newest completed runs
for that deployment. Unfinished or malformed records remain for investigation.
Qualification and promotion share a checkout lock, so cleanup cannot race them.

For GPU builds, set `WORKER_PROFILE=worker-cuda` before the acquisition and build
steps. It uses the same worker Dockerfile with a separate locked CUDA environment.
Only the `localhost` deployment is currently defined. A GPU deployment must select
its GPU and built image IDs explicitly and pass qualification before API jobs are
enabled; building the CUDA image alone does not establish that qualification.

Promote a completed qualification explicitly:

```sh
make deploy DEPLOYMENT=localhost DOCKER_CONTEXT=science QUALIFICATION="$QUALIFICATION"
```

Set `QUALIFICATION` to the directory printed by `make qualify`. Deployment checks
that the checker, cases, science configuration and selected image IDs still
match that evidence. It saves the current running image IDs and mounted TOML
under `.work/rollback/`, then requires clean SDK retirement before replacing the
worker. Startup never builds or pulls. The running services and mounted TOML
are checked against the qualified artifacts before promotion is recorded.
Rollback restores images and configuration together; uncertain retirement stops
replacement and retains recovery evidence.

Successful promotion keeps two completed rollback snapshots for that deployment
and Docker engine, preserving snapshots referenced by the running containers.
Failed promotions and older snapshots without recorded engine ownership are
left intact. Cleanup runs in these workflows; no separate cleanup command or
scheduled service is required.

The checkpoint manifest retains its original provenance comments; current
checkpoint acquisition and verification live in `compose/weights.yaml`.
