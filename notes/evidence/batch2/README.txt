Preparation-boundary measurements, 2026-11-23
============================================

Purpose
-------

Evidence for batch 2 of ../../003_provider_sdk_simplification_plan_20261123.txt.
These probes select scientific interfaces; they are not production adapters,
throughput benchmarks, scientific accuracy tests or migration qualification.
Inputs are authored fixtures or the owning repository's checked-in fixtures.
No live API jobs, credentials or user uploads are used.

Reproduce from the Chef checkout
-------------------------------

    python3 notes/evidence/batch2/run.py nmrpeak-hf
    python3 notes/evidence/batch2/run.py nmrpeak-chf
    python3 notes/evidence/batch2/run.py nmrnet-smiles
    python3 notes/evidence/batch2/run.py nmrnet-prediction
    python3 notes/evidence/batch2/run.py secs
    python3 notes/evidence/batch2/run.py magnet-e01
    python3 notes/evidence/batch2/run.py magnet-e02

The host script only orchestrates containers and records JSON. Each scientific
probe runs in an immutable cached image on Docker context science (observed
rootless), without network, added capabilities or host writes. It has four CPU
cores, 8 GiB memory including swap, 256 PIDs, bounded tmpfs and a 15-minute
timeout. The runner stops only its recorded container on interruption. This
is an offline measurement lane, not a hostile-code isolation claim.

run.py lists the exact images and read-only inputs. Reproduction requires the
sibling repositories and already acquired assets at those paths. Nothing is
downloaded. NMRPeak uses its own named checkpoint volumes. NMRNet uses its own
weight/checkpoint directories and a model declaration under its deployment
bundle; that JSON declares artifacts, not credentials. Its prediction probe
uses UID 1000, matching the running worker's asset ACL. Other probes use image
UID 65532. SECS mounts its model/cache/fixtures; Magnet's assets are in its images.

Run NMRNet smiles before prediction: prediction consumes that file's exact
captured conformer bytes and records its whole-file digest. Do not regenerate
conformers to retry prediction. Reproduction overwrites the selected result;
stderr goes to /tmp/batch2-<name>.log. Those scratch logs are not durable evidence.

Records and interpretation
--------------------------

Each result records probe SHA256, runtime/model facts, actual container argv,
captured input facts, timings and identity assertions. images.json is a later
engine inspection of those immutable IDs, not a claim about today's image tags.
The recorded cidfile names belonged to disposable host directories. Use run.py
instead of replaying the historical argv literally.

All measurements are CPU float32 where a model is involved. They were collected
on one host and are not exclusive-host timing guarantees. VmHWM is reset with
Linux clear_refs=5 before each phase. It includes imported modules, resident
models and allocator reuse; page accounting is approximate. It is NOT the
phase's allocation size, and subtracting two RSS readings is not an exact
allocation measurement. Tensor byte counts are reported separately. A retained
model can dominate RSS even when preparation is tiny.

NMRPeak: twenty reconstructions per captured input; three generations for each
ordinary case and one for each boundary case. Full int64 tensor values, special
tokens and padding are compared. The 512-token rejection is the current rule;
the corrected proposal has 254 tokens. Actual model capacity is 256 positions:
255 tokens pad to 264 and fail, despite passing preparation. This is recorded as
a defect to fix, not successful generation or desired compatibility. Candidate
strings are stochastic and are not the equivalence criterion.

NMRNet: two once-generated conformer collections; prediction uses the first.
Replacement preparation receives exactly the captured bytes. Tensor hashes,
shape/dtype and atom selections match. The first prepare includes lazy imports
and dictionary loading (1.65 s); do not call this a 1 ms cold path. Prediction
includes loading each model head. The 128-by-128 fixture measures maximum atom
and collection counts, not maximum JSON byte size, and does not run prediction.
For the repeated maximum-size fixture, all 128 tensor witnesses are checked
identical and one is recorded with its repetition count. Tensor byte totals
count logical tensors, including any shared storage; they
are not unique resident storage. The installed scaler emits its existing
scikit-learn version warning; this probe makes no prediction-accuracy claim.

SECS: five preparations per JCAMP/processed-Bruker fixture, three real-checkpoint
embeddings, and one small search with eight static candidates and one GraphGA
generation. The search uses a synthetic C2H6O query, not a reference answer for
the fixture. The full production index is absent; there is no full-retrieval
cost claim. Ordinary JSON float values round-trip to bit-identical float32
values and preserve metadata. Its approximately 8 ms serialization measurement
excludes SDK framing/validation and IPC. The base64 alternative is measured for
size only; it is not a proposed extra binary codec.

Magnet: twenty conversions and one 100-candidate generation for each of E01/E02,
using their offering decode policy and attempt-derived seed. Generation's actual
codec call is observed and compared with the admission input. Witness hashing
adds some generation overhead. G01 has no separate preparation boundary and no
matching cached image was used; no G01 model timing or equivalence is claimed.

These scripts intentionally inspect current private scientific internals to
answer this one decision. They are tied to the recorded images. They neither
create SDK APIs around those internals nor require future code to preserve them.
