"""Request and result data for one ChefNMR2 elucidation.

Only Peaks mirrors an upstream dataclass. ElucidationRequest groups the arguments
to ChefNMR2.elucidate(). ElucidateResult describes its return dictionary, and
CandidateVotes and RankedCandidate describe the candidate dictionaries inside it.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal, Self, TypedDict

log = logging.getLogger(__name__)


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

    Method signature and defaults:
    https://github.com/numpde/fork-of-chefnmr2/blob/29374e57952684533d266832288413ba87c5d809/chefnmr2/infer/api.py#L583-L599
    """

    peaks: Peaks
    formula: str
    n_samples: int | Literal["adaptive"] = 100
    seed: int | None = 0

    @classmethod
    def from_json_or_none(cls, obj: object) -> Self | None:
        """Construct a request from decoded JSON, logging construction failures."""
        if not isinstance(obj, dict):
            log.warning(f"Expected a JSON object, got {type(obj)=}")
            return None

        message = obj.copy()

        try:
            peaks = Peaks(**message.pop("peaks"))
        except KeyError:
            log.warning(f"Missing field 'peaks' in {obj.keys()=}")
            return None
        except TypeError as error:
            log.warning(f"Could not construct `Peaks` from the 'peaks' field; {error = }")
            return None

        try:
            peaks.h = [(ppm, integration, multiplicity) for (ppm, integration, multiplicity) in peaks.h]
        except (TypeError, ValueError) as error:
            log.warning(f"Could not unpack `peaks.h` into (ppm, integration, multiplicity) triples; {error = }")
            return None

        try:
            return cls(peaks=peaks, **message)
        except TypeError as error:
            log.warning(f"Could not construct {cls} from {obj.keys()=}; {error = }")
            return None


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
    ensemble: dict[str, object]
    params: dict[str, object]
    adaptive: dict[str, object] | None = None


@dataclass
class ModelFailure:
    """Diagnostic returned when ChefNMR2.elucidate() raises an exception."""

    exception_type: str
    message: str
