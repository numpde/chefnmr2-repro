"""Translate a verified NMR API job into ChefNMR2 input and a public result."""

import asyncio
from dataclasses import dataclass
from hashlib import sha256

from provider_core.analysis import AnalysisFailure, AnalysisOutcomeUnresolved, AnalysisSuccess, PreparedAnalysis
from provider_core.attempt_record import ProtectedEvidence
from provider_core.hosting import settle_cleanup
from provider_core.payload import JsonDataclassCodec, PayloadRejected

if __package__:
    from .._shared.wire import ElucidationRequest, Peaks
else:
    from wire import ElucidationRequest, Peaks


@dataclass
class JobInput:
    """Scientific observations; the deployment owns sampling budget and seed."""

    formula: str
    peaks: Peaks


class ElucidationSession:
    """Own one Attempt's input and call while the provider retains the model."""

    def __init__(self, provider):
        self.provider = provider
        self.codec = JsonDataclassCodec(format="chefnmr2.job_input.v1", value_type=JobInput)
        self.prepared = None
        self.request = None
        self.calculation = None

    async def prepare(self, acknowledged, instructions, *, report_condition):
        """Decode the verified API input without allocating scientific work."""
        if acknowledged.start.input != instructions.identity:
            raise ValueError(f"{acknowledged.start.input=} should match {instructions.identity=}")

        evidence = ProtectedEvidence("chefnmr2.input_digest.v1", sha256(instructions.specification_bytes).hexdigest().encode())

        try:
            supplied = self.codec.decode(instructions.specification_bytes)
        except PayloadRejected:
            return AnalysisFailure(
                "invalid_input",
                'Expected JSON with "formula" and "peaks": carbon shifts in "c" and proton [ppm, integration, multiplicity] triples in "h". Sampling settings belong to the deployment.',
                evidence,
            )

        self.request = ElucidationRequest(
            formula=supplied.formula,
            peaks=supplied.peaks,
            n_samples=self.provider.offering["n_samples"],
            seed=self.provider.offering["seed"],
        )
        self.prepared = PreparedAnalysis(evidence)
        return self.prepared

    async def execute(self, prepared, *, report_condition):
        """Call the loaded model once; the SDK rechecks permission before retries."""
        if prepared is not self.prepared or self.calculation is not None:
            raise ValueError(f"{prepared is self.prepared=}, {self.calculation=} should identify this session's unused input")

        self.prepared = None
        self.calculation = asyncio.create_task(
            self.provider.elucidate(self.request, require_execution=report_condition.require_execution)
        )

        try:
            result = await asyncio.shield(self.calculation)
        except Exception as error:
            # An arbitrary worker/transport exception does not establish a public
            # scientific failure. The SDK retains the Attempt without inventing FAIL
            raise AnalysisOutcomeUnresolved("ChefNMR2 did not return a verified result") from error

        content = self.provider.inference.binding.result_codec.encode(result)
        evidence = ProtectedEvidence("chefnmr2.result_digest.v1", sha256(content).hexdigest().encode())
        return AnalysisSuccess("chefnmr2.elucidation_result.v1", content, evidence)

    async def retire(self):
        """Join the original call; withdraw the model after interrupted or failed work."""
        self.prepared = None
        if self.calculation is None:
            return

        if not self.calculation.done():
            self.calculation.cancel()

        await asyncio.wait({self.calculation})

        if self.calculation.cancelled() or self.calculation.exception() is not None:
            # A completed Python task alone cannot prove cleanup after a failed
            # call. Retire the original host before this Attempt releases its work
            await settle_cleanup(control=self.provider.inference.host.retire())
