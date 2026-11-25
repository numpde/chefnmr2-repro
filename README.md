# ChefNMR2 for the NMR Hub

Molecular structure inference from a formula and unassigned ¹H/¹³C NMR peaks,
using [ChefNMR2](https://github.com/ml-struct-bio/chefnmr2), packaged as a provider
for the [NMR Hub](https://nmr.dapp32.com).

The provider handles API jobs and delivery through the SDK; the worker owns the
model and checkpoints. They communicate through a private runner socket. Both
have networking disabled: API traffic uses the shared delivery service's Unix
proxy, and prose interpretation uses a separate proxy restricted to
`api.openai.com:443`. Only the provider receives credentials.

## From a clean checkout

Use a Linux host with Git, GNU Make, Bash, Python 3.11+, and Docker with Compose
and Buildx. Run as the non-root checkout owner and select a Docker context with
a local Unix socket. Images target `linux/amd64`. The pinned submodule requires
GitHub SSH access. No SDK checkout or host SDK installation is needed.

Run the following blocks in the same Bash session. Replace `local` with your
Docker context; reuse an existing wheelhouse outside the checkout if available.

```bash
git clone git@github.com:numpde/chefnmr2-repro.git
cd chefnmr2-repro
make upstream/init

CONTEXT=local
WHEELHOUSE="$HOME/.cache/chefnmr2/wheelhouse"
PROXY_URL=none
mkdir -p "$WHEELHOUSE"
```

`PROXY_URL=none` selects direct artifact downloads; otherwise use an HTTP,
SOCKS5 or SOCKS5h URL with its port. Image pulls independently use the Docker
engine's proxy. A loopback proxy on a rootless engine needs Docker 29.5+ with
RootlessKit's detached network namespace. Failed proxy connections do not fall
back to direct access. Pass Make inputs explicitly as below; omitted inputs
prompt on a terminal, and environment variables alone do not select them.
Docker contexts, build profiles and deployments are shown as numbered choices.
Interactive wheelhouse and proxy prompts also offer the five most recent values
from `tmp/inputs/`; type a number or a new value. This is local, disposable history,
not a default. Proxy URLs with userinfo, queries or fragments are not saved.

**Build the CPU images.** First pull the pinned external images, then acquire
SDK tooling and locked dependencies. Builds run offline; each subshell stops on
failure.

```bash
(
  set -euo pipefail
  make images/pull DOCKER_CONTEXT="$CONTEXT"

  make sdk/acquire DOCKER_CONTEXT="$CONTEXT" \
      WHEELHOUSE="$WHEELHOUSE" PROXY_URL="$PROXY_URL"
  for profile in provider worker-cpu weights-sanitize; do
    make acquire PROFILE="$profile" DOCKER_CONTEXT="$CONTEXT" \
        WHEELHOUSE="$WHEELHOUSE" PROXY_URL="$PROXY_URL"
    make build PROFILE="$profile" DOCKER_CONTEXT="$CONTEXT" \
        WHEELHOUSE="$WHEELHOUSE"
  done
)
```

`compose/images.yaml` declares the pinned Python bases, curl downloader and local
builds. Bake uses its build definitions; `images/pull` uses Compose to pull only
the prebuilt inputs. It needs no wheelhouse or `PROXY_URL`; Docker uses its engine
proxy. It does not build application images or acquire the uv tool used only for
lock regeneration.

A new checkout uses the checked-in locks; do not regenerate them for setup.
Acquisition verifies artifact hashes, and builds fail on missing inputs or stale
locks. The shared acquisition bootstrap is included in the checkout.

**Prepare checkpoints.** This downloads all ten pinned checkpoints into the
persistent `chefnmr2-weights_checkpoints` Docker volume. The small pinned curl
image is pulled if missing; sanitization and verification run offline.

```bash
(
  set -euo pipefail
  make weights/download DOCKER_CONTEXT="$CONTEXT" PROXY_URL="$PROXY_URL"
  make weights/sanitize DOCKER_CONTEXT="$CONTEXT"
  make weights/verify DOCKER_CONTEXT="$CONTEXT"
)
```

Use `make weights/watch DOCKER_CONTEXT="$CONTEXT"` in another terminal to watch
an active download. Downloads resume partial files. Sanitization checks pinned
hashes and restricted loading, then writes a receipt covering `model.pt` and
`config.yaml`; these released checkpoints need no rewrite. Model loading checks
the selected checkpoints against those receipts. After replacing checkpoint
files, run sanitization again.

Runtime services inherit the local image tags from `compose/images.yaml`.
The localhost deployment selects `worker-cpu`; `nmr-dapp32-com` selects
`worker-cuda`. Runtime definitions clear the inherited build settings and
never pull images, so the selected images must already exist on the engine.

**Test the images.** Manual runs need no API services or credentials. They use
an isolated Compose project, share checkpoints read-only, and retire the model
session when stdin closes. One input line produces one JSON batch reply.

```bash
make check DOCKER_CONTEXT="$CONTEXT"
printf '%s\n' '{"requests":[{"formula":"CH4","peaks":{"h":[[1.0,4,"s"]]}}]}' |
  make manual DEPLOYMENT=localhost DOCKER_CONTEXT="$CONTEXT"
make test/model DEPLOYMENT=localhost DOCKER_CONTEXT="$CONTEXT"
```

The model test checks cold/warm ten-sample morphine inference, ordered
eight-sample SDK batches, scientific validity and retirement. Results and timings
go to stdout, progress goes to stderr, and a failed check returns a nonzero exit
status. Redirect output when a saved test log is useful. Test projects are
removed on completion or failure.

**GPU:** Build `worker-cuda` instead of `worker-cpu`; `images/pull` includes its
Python base. `deployments/nmr-dapp32-com` selects one Ada GPU by UUID and uses
`cuda:0` inside the worker. On another host, select its GPU explicitly. Run
`make test/model DEPLOYMENT=nmr-dapp32-com DOCKER_CONTEXT="$CONTEXT"` to exercise
that deployment's images and device.

## API deployment

`nmr-dapp32-com` targets the public API as `provider:chefnmr2-pa-a`, using system
CA trust. Its interpreter proxy reads the host's upstream DNS servers from
`/run/systemd/resolve/resolv.conf`; its interface-bound DNS cannot use Docker's
loopback resolver. It requires the shared delivery volumes below and
`chefnmr2-nmr-dapp32-com_credentials`, containing its dedicated `credential.json`
and `interpreter.key` with the same ownership and permissions as localhost.

`deployments/localhost/provider.toml` selects the API endpoint and identity,
model settings, batching and execution limits. Before deployment, provide:

- An already-running local API and shared delivery service, with the external
  proxy/outbox volumes named in [provider.yaml](deployments/localhost/provider.yaml).
  This repository does not provision those services.
- `chefnmr2-inference_credentials` containing `credential.json` and
  `interpreter.key`, each UID:GID `65532:65532`, mode `0600`. The API credential
  must match the configured provider identity. Use dedicated credentials.
- `secrets/localhost/api-ca.pem`, the local API's public CA certificate.
- `secrets/localhost/resolv.conf` containing reachable non-loopback IPv4 DNS
  servers for the interpreter proxy.

Keep credentials outside Git. Manual operations load only the science overlay;
the API overlay adds credentials and delivery sockets. Localhost also supplies
its private CA and resolver configuration.

Run builds, tests and startup sequentially on the same selected engine. Inspect
the API configuration, test the built images, then start them. Use
`DEPLOYMENT=nmr-dapp32-com` instead for the production deployment.

```bash
make config DEPLOYMENT=localhost DOCKER_CONTEXT="$CONTEXT" &&
make test/provider DOCKER_CONTEXT="$CONTEXT" &&
make test/model DEPLOYMENT=localhost DOCKER_CONTEXT="$CONTEXT" &&
make up DEPLOYMENT=localhost DOCKER_CONTEXT="$CONTEXT" &&
make status DEPLOYMENT=localhost DOCKER_CONTEXT="$CONTEXT"
```

`make up` uses the current local image tags and deployment configuration. It
stops an existing provider with its configured signal and grace period, requires
exit code zero, then lets Compose start or replace services. Startup never
builds or pulls and does not rerun tests. A startup failure returns an error;
there is no automatic rollback. Inspect service state and logs before retrying:

```bash
make status DEPLOYMENT=localhost DOCKER_CONTEXT="$CONTEXT" &&
make logs DEPLOYMENT=localhost DOCKER_CONTEXT="$CONTEXT"
```

## Requests and execution

The localhost offering uses eight samples and seed 0. API job input may be JSON
such as `{"formula":"CH4","peaks":{"h":[[1.0,4,"s"]]}}`, or prose such as
`Formula CH4. Proton NMR: 1.0 ppm, integration 4H, singlet.` Attachments are not read.
Structured input is decoded directly; malformed JSON cannot fall back to the
language model. Prose uses the SDK Interpreter with bounded retries and a deadline.
Sampling settings belong to the deployment and cannot be overridden by a job.

Results use `chefnmr2.elucidation_report.v1`: `input` records the formula and peaks
used; `elucidation` contains the scientific output. Inspect `input`: typed
validation does not prove prose was transcribed faithfully. Interpretation failure
is distinct from invalid structured input and does not start inference.

The worker accepts `{"requests": [...]}` and returns position-matched
`{"replies": [...]}` sequentially. A model exception yields a diagnostic for that
entry; later entries still run. The localhost API service holds up to four jobs,
prepares two concurrently and collects up to four entries per batch, waiting at
most 50 ms. Slow delivery retains capacity without rerunning science.

Cancellation removes queued jobs. A dispatched entry may still run, but its
result is not published; sibling results remain independent. A malformed or lost
whole reply retires the shared host without replay. Provider restart does not
resume an existing API Attempt.

With SDK 0.7.7, the localhost execution policy requests a stop after 60 seconds,
including initialization and the whole batch. The first stop request starts a
10-second grace; unfinished work then triggers worker retirement with separate
cleanup allowances. Unconfirmed retirement is reported, not treated as success.

## Storage and maintenance

| Location | Contents and lifetime |
| --- | --- |
| `$WHEELHOUSE` | Persistent wheels, source archives and SDK tooling; reusable across checkouts and profiles |
| `tmp/{build,lock}/YYYYMMDDTHHMMSSZ-xxxxxxxx/` | Installation lists and resolver scratch; removed on scope exit, including exceptions; abrupt termination can leave residue |
| Docker/BuildKit storage | Imported wheelhouse context and cached layers; managed by Docker's cache lifecycle |
| Final images | Installed packages; dependency archives are mounted during installation, not copied into the final image |
| `chefnmr2-weights_checkpoints` volume | Persistent checkpoints and preparation receipts |

Bake reads the wheelhouse directly, with no host wheel-staging copy or intermediate
wheel image. Build steps mount it read-only and install selected files with
`--no-index --no-deps --require-hashes`. BuildKit can copy/cache the **whole pool**;
this is not a zero-copy host mount. The base image bundles pip's wheel, and the
[acquisition bootstrap](make/bootstrap/README.md) includes dependency source and a
pinned pip wheel. ANTLR is built into a wheel in an intermediate Docker stage;
its hash is retained at `/opt/inference/source-wheel-sha256.txt`.

`make/work.py` removes build and resolver scratch on scope exit. Model tests and
manual runs remove their isolated Compose projects when they finish.

For dependency changes, edit `builds/<profile>/requirements.in` or its constraints,
then run `make lock` with explicit `PROFILE`, `WHEELHOUSE`, `DOCKER_CONTEXT` and
`PROXY_URL`. Resolution uses cached packages plus index metadata and requires a
rootless engine and the pinned uv image from Bake. `builds/sdk.constraints.txt`
selects the SDK release. Normal builds use the resulting checked-in lock.

Prompts and offering text live in `containers/provider/prompts/` and
`containers/provider/about/`; rebuild the provider after editing them.

| Tests | Command |
| --- | --- |
| Host operator workflows with fake Docker (uses util-linux `script`) | `make test/operator` |
| Checkpoint preparation and model admission in cached images | `make test/unit DOCKER_CONTEXT="$CONTEXT"` |
| Provider behavior with real SDK and controlled science/HTTP | `make test/provider DOCKER_CONTEXT="$CONTEXT"` |
| Real inference and SDK batches on the selected deployment | `make test/model DEPLOYMENT=localhost DOCKER_CONTEXT="$CONTEXT"` |

The operator, unit and provider targets accept `/normative` or `/adversarial`
for one category. `tests/model/` supplies the scientific cases and checker used
by `make test/model`.
