# ChefNMR2 repackaging for the NMR Hub

See [NMR Hub](https://nmr.dapp32.com).

Molecular structure inference from a formula and unassigned ¹H/¹³C NMR peaks,
using [ChefNMR2](https://github.com/ml-struct-bio/chefnmr2).
NMR API provider integration is planned.

The provider container's manual testing interface is in
[`docker/containers/provider/manual.py`](docker/containers/provider/manual.py).
It accepts a deployment configuration through `--config`, reads one JSON request
per line from stdin, and writes results to stdout. The provider image and Compose
service are not yet defined.
