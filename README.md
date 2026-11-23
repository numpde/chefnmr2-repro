# ChefNMR2 repackaging for the NMR Hub

See [NMR Hub](https://nmr.dapp32.com).

Molecular structure inference from a formula and unassigned ¹H/¹³C NMR peaks,
using [ChefNMR2](https://github.com/ml-struct-bio/chefnmr2).
NMR API provider integration is in progress. The provider image is not yet
buildable: `requirements/provider-api.txt` is missing, and the job service and
retry permission hook require a newer SDK release than the pinned 0.7.1.
The localhost API service also still needs its Compose wiring.

The provider container's manual testing interface is in
[`docker/containers/provider/manual.py`](docker/containers/provider/manual.py).
It accepts a deployment configuration through `--config`, reads one JSON request
per line from stdin, and writes results to stdout. The provider connects to the
worker through the shared runner socket; only the worker mounts checkpoints.

Once those dependency blockers are resolved, build the images and start the
worker on the `science` Docker context:

```sh
docker --context science compose -f docker/compose/inference.yaml --profile manual build worker manual
docker --context science compose -f docker/compose/inference.yaml up -d --no-build worker
```

Send an eight-sample request using the checked-in test deployment:

```sh
printf '%s\n' '{"formula":"CH4","peaks":{"h":[[1.0,4,"s"]]}}' |
  docker --context science compose -f docker/compose/inference.yaml run --rm -T manual
```

The provider loads the model once for the input stream. Each request produces one
JSON result on stdout; logs go to stderr. Closing stdin retires the model process.
The runner remains available for the next provider session. The external
`chefnmr2-weights_checkpoints` volume must already contain the configured checkpoint.
