"""Translate a verified NMR API job into ChefNMR2 input and a public result."""

from dataclasses import dataclass
from hashlib import sha256
import logging

from provider_core.analysis import AnalysisFailure, AnalysisOutcomeUnresolved, AnalysisSuccess, PreparedAnalysis
from provider_core.attempt_record import ProtectedEvidence
from provider_core.interpreter import InterpretationFailed, InterpretationInputRejected
from provider_core.payload import JsonDataclassCodec, PayloadRejected

if __package__:
    from .._shared.wire import ElucidationBatch, ElucidationRequest, ElucidationResult, ModelFailure
    from .job_input import JobInput, interpret
else:
    from wire import ElucidationBatch, ElucidationRequest, ElucidationResult, ModelFailure
    from job_input import JobInput, interpret


log = logging.getLogger(__name__)


@dataclass
class ElucidationReport:
    """Expose the actual observations used, including after prose interpretation."""

    input: JobInput
    elucidation: ElucidationResult


class ElucidationSession:
    """Own one Attempt's input and call while the provider retains the model."""

    def __init__(self, provider, *, new_interpreter):
        self.provider = provider
        self.new_interpreter = new_interpreter
        self.interpreter = None
        self.supplied = None
        self.codec = JsonDataclassCodec(format="chefnmr2.job_input.v1", value_type=JobInput)
        self.prepared = None
        self.elucidation_input = None

    async def prepare(self, acknowledged, instructions, *, report_condition):
        """Route JSON directly; interpret prose before allocating scientific work."""

        if acknowledged.start.input != instructions.identity:
            raise ValueError(f"{acknowledged.start.input=} should match {instructions.identity=}")

        evidence = ProtectedEvidence("chefnmr2.input_digest.v1", sha256(instructions.specification_bytes).hexdigest().encode())

        try:
            source = instructions.specification_bytes.decode("utf-8")

            # Structured-looking input stays on the JSON path, even if malformed
            if source.lstrip().startswith(("{", "[")):
                supplied = self.codec.decode(instructions.specification_bytes)
            else:
                self.interpreter = self.new_interpreter()

                log.info("Interpreting the job specification")

                supplied = await interpret(
                    self.interpreter,
                    source,
                    codec=self.codec,
                    require_execution=report_condition.require_execution,
                )

                log.info("Job interpretation accepted")
        except (PayloadRejected, UnicodeDecodeError):
            return AnalysisFailure(
                "invalid_input",
                'Expected UTF-8 prose or JSON with "formula" and "peaks": carbon shifts in "c" and proton [ppm, integration, multiplicity] triples in "h". Sampling settings belong to the deployment.',
                evidence,
            )
        except (InterpretationFailed, InterpretationInputRejected):
            return AnalysisFailure(
                "interpretation_failed",
                "The interpreter did not establish a complete formula and peak set. This does not establish that the scientific input is invalid; you can submit explicit formula/peaks JSON.",
                evidence,
            )

        report_condition.require_execution()
        self.supplied = supplied

        self.elucidation_input = ElucidationRequest(
            formula=supplied.formula,
            peaks=supplied.peaks,
            n_samples=self.provider.offering["n_samples"],
            seed=self.provider.offering["seed"],
        )

        self.prepared = PreparedAnalysis(evidence)

        return self.prepared

    async def execute(self, prepared, *, report_condition):
        """Call the loaded model once; the SDK rechecks permission before retries."""

        if (prepared is None) or (prepared is not self.prepared):
            raise ValueError(f"{prepared=} should be this session's unused prepared input")

        self.prepared = None

        try:
            batch_result = await self.provider.inference(
                scientific_input=ElucidationBatch(requests=[self.elucidation_input]),
                require_execution=report_condition.require_execution,
            )

            (elucidation_result,) = batch_result.replies
        except Exception as error:
            # An arbitrary worker/transport exception does not establish a public
            # scientific failure. The SDK retains the Attempt without inventing FAIL
            raise AnalysisOutcomeUnresolved("ChefNMR2 did not return a verified result") from error

        if isinstance(elucidation_result, ModelFailure):
            failure_codec = JsonDataclassCodec(format="chefnmr2.model_failure.v1", value_type=ModelFailure)

            return AnalysisFailure(
                "model_failed",
                f"ChefNMR2 raised {elucidation_result.exception_type} while processing this input.",
                ProtectedEvidence(failure_codec.format, failure_codec.encode(elucidation_result)),
            )

        result_codec = JsonDataclassCodec(format="chefnmr2.elucidation_report.v1", value_type=ElucidationReport)
        content = result_codec.encode(ElucidationReport(input=self.supplied, elucidation=elucidation_result))
        evidence = ProtectedEvidence("chefnmr2.result_digest.v1", sha256(content).hexdigest().encode())

        return AnalysisSuccess(result_codec.format, content, evidence)

    async def retire(self):
        """Retire interpretation; the SDK owns scientific call and host cleanup."""
        self.prepared = None

        if self.interpreter is not None:
            await self.interpreter.retire()
