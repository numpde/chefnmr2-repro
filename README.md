# ChefNMR2 NMR API provider

Molecular structure inference from a molecular formula and unassigned ¹H/¹³C
NMR peak lists, using [ChefNMR2](https://github.com/ml-struct-bio/chefnmr2).

The provider serves `mol_from_1h_13c_formula` through the NMR API and returns
candidate structures as SMILES ranked by sampling frequency. Predictions describe
planar structures and do not establish stereochemistry.

ChefNMR2 source is pinned as the `chefnmr2-upstream` submodule. Checkpoints and
runtime dependencies are pinned separately. The model runs in its own Python
environment. Shared provider tools handle API communication and result delivery.

Run `make weights/download` to acquire and verify the checkpoints. It asks for a
local Docker context and a proxy URL; Enter selects `none` for direct downloads.
Transfers are quiet; filenames and verification results are printed. Ctrl+C stops the
operation and cleans up containers while preserving completed and partial files
for the next run.

In another terminal, run `make weights/watch DOCKER_CONTEXT=<same-context>` to
watch checkpoint file sizes every two seconds. Files ending in `.part` are
unfinished downloads. Ctrl+C stops only the watcher.

Run `make compatibility/check DOCKER_CONTEXT=science` to build the Python 3.12
CPU-only PyTorch image, verify the epoch949 checkpoint/config pair, and run the
single-model inference check. Select another declared pair with
`CHECKPOINT=chefnmr2-epoch999`. Only the selected pair needs to be downloaded;
the checkpoint operation lock prevents testing while a download is running.

`make compatibility/image DOCKER_CONTEXT=science` builds the image without
running inference. Its pinned Python base must already be available on that
engine. Build steps need network access; inference runs offline with read-only
source and checkpoints. Containers and temporary files are removed after each
run; the reusable image, build cache and checkpoint volume are retained.
