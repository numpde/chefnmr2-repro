"""Translate a verified NMR API job into ChefNMR2 input and a public result."""

from dataclasses import dataclass
from hashlib import sha256
import json
import logging

from provider_core.analysis import AnalysisFailure, AnalysisOutcomeUnresolved, AnalysisSuccess, PreparedAnalysis
from provider_core.attempt_record import MAX_EVIDENCE_BYTES, ProtectedEvidence
from provider_core.diagnostics import failure_evidence_json
from provider_core.failure_collection import FailureCollector
from provider_core.interpreter import InterpretationFailed, InterpretationInputRejected
from provider_core.payload import JsonDataclassCodec, PayloadRejected

if __package__:
    from model.wire import ElucidationRequest, ElucidationResult, ModelFailure
    from .job_input import JobInput, interpret
else:
    from wire import ElucidationRequest, ElucidationResult, ModelFailure
    from job_input import JobInput, interpret


log = logging.getLogger(__name__)


@dataclass
class ElucidationReport:
    """Expose the actual observations used, including after prose interpretation."""

    input: JobInput
    elucidation: ElucidationResult


class ElucidationSession:
    """Own one Attempt's input and call while the provider retains the model."""

    def __init__(self, provider, *, new_interpreter, diagnostic_secrets=()):
        self.provider = provider
        self.new_interpreter = new_interpreter
        self.interpreter = None
        self.supplied = None
        self.codec = JsonDataclassCodec(format="chefnmr2.job_input.v1", value_type=JobInput)
        self.prepared = None
        self.participation = None
        self.diagnostic_secrets = tuple(diagnostic_secrets)
        self.acknowledged = None

    def preparation_failure(self, code, message, error, *, evidence, instructions):
        """Retain the causal graph alongside its input digest, without the input bytes."""

        # todo: Review partial input echoes in endpoint diagnostics; exact source
        # text and configured credentials are removed before the SDK bounds text.
        diagnostic = failure_evidence_json(
            {
                "input_digest": evidence.content.decode("ascii"),
                "failure": error,
            },
            secrets=(*self.diagnostic_secrets, instructions.specification_bytes.decode("utf-8", errors="replace")),
            maximum_bytes=MAX_EVIDENCE_BYTES,
        )

        protected = ProtectedEvidence("chefnmr2.preparation_failure.v1", diagnostic)

        self.report_failure(code, protected)

        return AnalysisFailure(code, message, protected)

    def report_failure(self, code, evidence):
        """Emit the protected record with the API identities that locate its Attempt."""
        admission = self.acknowledged.admission

        # todo: Logs rotate; provide an SDK-owned durable evidence sink before
        # treating this record as an archive independent of the live Attempt.
        log.error(
            f"{admission.job_ref=}; {admission.execution_attempt_ref=}; {code=}; "
            f"evidence={evidence.content.decode('utf-8')}"
        )

    async def prepare(self, acknowledged, instructions, *, report_condition):
        """Route JSON directly; interpret prose before allocating scientific work."""

        if acknowledged.start.input != instructions.identity:
            raise ValueError(f"{acknowledged.start.input=} should match {instructions.identity=}")

        self.acknowledged = acknowledged

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
        except (PayloadRejected, UnicodeDecodeError) as error:
            return self.preparation_failure(
                "invalid_input",
                'Expected UTF-8 prose or JSON with "formula" and "peaks": carbon shifts in "c" and proton [ppm, integration, multiplicity] triples in "h". Sampling settings belong to the deployment.',
                error,
                evidence=evidence,
                instructions=instructions,
            )
        except (InterpretationFailed, InterpretationInputRejected) as error:
            return self.preparation_failure(
                "interpretation_failed",
                "The interpreter did not establish a complete formula and peak set. This does not establish that the scientific input is invalid; you can submit explicit formula/peaks JSON.",
                error,
                evidence=evidence,
                instructions=instructions,
            )

        report_condition.require_execution()
        self.supplied = supplied

        elucidation_input = ElucidationRequest(
            formula=supplied.formula,
            peaks=supplied.peaks,
            n_samples=self.provider.offering["n_samples"],
            seed=self.provider.offering["seed"],
        )

        self.participation = await self.provider.batches.prepare(elucidation_input)
        self.prepared = PreparedAnalysis(evidence)

        return self.prepared

    async def execute(self, prepared, *, report_condition):
        """Activate this job in the shared batch; map only its verified entry result."""

        if (prepared is None) or (prepared is not self.prepared):
            raise ValueError(f"{prepared=} should be this session's unused prepared input")

        self.prepared = None

        try:
            elucidation_result = await self.participation.execute(require_execution=report_condition.require_execution)
        except Exception as error:
            # An arbitrary worker/transport exception does not establish a public
            # scientific failure. The SDK retains the Attempt without inventing FAIL
            raise AnalysisOutcomeUnresolved("ChefNMR2 did not return a verified result") from error

        if isinstance(elucidation_result, ModelFailure):
            failure_codec = JsonDataclassCodec(format="chefnmr2.model_failure.v2", value_type=ModelFailure)

            evidence = ProtectedEvidence(failure_codec.format, failure_codec.encode(elucidation_result))

            self.report_failure("model_failed", evidence)

            return AnalysisFailure(
                "model_failed",
                f"ChefNMR2 raised {elucidation_result.exception_type} while processing this input.",
                evidence,
            )

        result_codec = JsonDataclassCodec(format="chefnmr2.elucidation_report.v1", value_type=ElucidationReport)
        content = result_codec.encode(ElucidationReport(input=self.supplied, elucidation=elucidation_result))
        evidence = ProtectedEvidence("chefnmr2.result_digest.v1", sha256(content).hexdigest().encode())

        return AnalysisSuccess(result_codec.format, content, evidence)

    async def retire(self):
        """Join this job's participation; siblings retain their shared calculation."""
        self.prepared = None

        failures = FailureCollector(message="ChefNMR2 job resources did not settle")

        with failures.capture():
            if self.interpreter is not None:
                await self.interpreter.retire()

        with failures.capture():
            if self.participation is not None:
                await self.participation.retire()

        failures.raise_if_any()
