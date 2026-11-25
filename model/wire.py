"""Batched worker requests and replies for ChefNMR2 elucidation.

Only Peaks mirrors an upstream dataclass. ElucidationRequest groups the arguments
to ChefNMR2.elucidate(). ElucidateResult describes its return dictionary, and
CandidateVotes and RankedCandidate describe the candidate dictionaries inside it.
"""

from dataclasses import dataclass, field
from typing import Literal
from typing_extensions import TypedDict
from pydantic import JsonValue


@dataclass
class Peaks:
    """Carbon shifts in ppm and proton (shift in ppm, integration, multiplicity) triples.

    Upstream equivalent:
    https://github.com/numpde/fork-of-chefnmr2/blob/29374e57952684533d266832288413ba87c5d809/chefnmr2/infer/api.py#L161-L166

    Multiplicity accepts any string. Use abbreviations such as "d" for doublet:
    ChefNMR2 scans for tokens inside the lowercased string, so a spelled-out
    name can give a different result.

    The model distinguishes s, d, t, q, dd, dt, td, m, and '' for padding.
    Labels such as "ddd", "dddd", "ddt", "p", "h", and "hept" all map to "m".
    An empty string or a string with no recognized tokens maps to padding.

    Ref:
    https://github.com/numpde/fork-of-chefnmr2/blob/29374e57952684533d266832288413ba87c5d809/chefnmr2/infer/api.py#L31-L57
    """

    c: list[float] = field(default_factory=list)
    h: list[tuple[float, float, str]] = field(default_factory=list)


@dataclass
class ElucidationRequest:
    """Arguments for ChefNMR2.elucidate().

    An integer n_samples is the total ensemble budget, not a per-checkpoint budget.
    ChefNMR2 rounds it up to an equal positive number of draws per checkpoint.
    The result's n_samples reports the actual total after rounding.
    With ten checkpoints, requesting eight draws therefore produces ten.

    Method signature and budget:
    https://github.com/numpde/fork-of-chefnmr2/blob/9922b6c571428757c37f127225a62f2070d6ddc0/chefnmr2/infer/api.py#L580-L600
    """

    peaks: Peaks
    formula: str
    n_samples: int | Literal["adaptive"] = 100
    seed: int | None = 0


def request_defaults(offering):
    """Select the scientific defaults owned by the offering."""
    return {"n_samples": offering["n_samples"], "seed": offering["seed"]}


class CandidateVotes(TypedDict):
    """A reconstructed structure and its vote count.

    Candidate fields:
    https://github.com/numpde/fork-of-chefnmr2/blob/29374e57952684533d266832288413ba87c5d809/chefnmr2/infer/api.py#L704-L711
    """

    smiles: str
    votes: int


class RankedCandidate(CandidateVotes):
    """A ranked structure and its share of the ranked votes.

    Vote fraction calculation:
    https://github.com/numpde/fork-of-chefnmr2/blob/29374e57952684533d266832288413ba87c5d809/chefnmr2/infer/api.py#L698-L707
    """

    vote_fraction: float


@dataclass
class ElucidationResult:
    """Wire representation of the result dictionary from ChefNMR2.elucidate().

    Result dictionary:
    https://github.com/numpde/fork-of-chefnmr2/blob/29374e57952684533d266832288413ba87c5d809/chefnmr2/infer/api.py#L703-L745

    With n_samples="adaptive", the adaptive field summarizes the first 100 samples and records the decision
    thresholds, the selected sampling branch, and the total sample budget for each branch.

    Adaptive result fields:
    https://github.com/numpde/fork-of-chefnmr2/blob/29374e57952684533d266832288413ba87c5d809/chefnmr2/infer/api.py#L666-L673
    """

    candidates: list[RankedCandidate]
    implausible_candidates: list[CandidateVotes]
    n_samples: int
    n_valid: int
    n_unique_candidates: int
    validity: float
    status_counts: dict[str, int]
    timing_s: dict[str, float]
    ensemble: dict[str, JsonValue]
    params: dict[str, JsonValue]
    adaptive: dict[str, JsonValue] | None = None


@dataclass
class ModelFailure:
    """A completed model call raised; its diagnostic belongs to this batch entry."""

    exception_type: str
    message: str
    diagnostic: dict[str, JsonValue]


@dataclass
class ElucidationBatch:
    """Requests processed sequentially in their supplied order."""

    requests: list[ElucidationRequest]


@dataclass
class ElucidationBatchResult:
    """One reply per request, in the same order; model failures do not stop later entries.

    A process or transport failure can prevent the entire reply from arriving.
    That does not authorize replaying a batch whose earlier entries may have run.
    """

    replies: list[ElucidationResult | ModelFailure]
