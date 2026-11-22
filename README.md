# ChefNMR2 NMR API provider

Molecular structure inference from a molecular formula and unassigned ¹H/¹³C
NMR peak lists, using [ChefNMR2](https://github.com/ml-struct-bio/chefnmr2).

The provider serves `mol_from_1h_13c_formula` through the NMR API and returns
candidate structures as SMILES ranked by sampling frequency. Predictions describe
planar structures and do not establish stereochemistry.

ChefNMR2 source is pinned as the `chefnmr2-upstream` submodule. Checkpoints and
runtime dependencies are pinned separately. The model runs in its own Python
environment. Shared provider tools handle API communication and result delivery.
