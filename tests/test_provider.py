"""Exercise the API science adapter inside the provider image, without a model."""

import asyncio
from copy import deepcopy
from contextlib import asynccontextmanager, nullcontext
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from provider_core.analysis import AnalysisFailure, AnalysisOutcomeUnresolved, AnalysisSuccess
from provider_core.batch_call import BatchPolicy
from provider_core.chat_errors import ChatFailed
from provider_core.embedded_host import InProcessExecutionHost
from provider_core.hosting import CleanupObservation, OperationPolicy
from provider_core.interpreter_settings import ManagedInterpreter
from provider_core.lifetime import ProviderLifetime
from provider_core.payload import JsonDataclassCodec, PayloadRejected
from provider_core.process_call import ProcessCall
from provider_core.scientific import Analysis, Contribution, operation
from provider_core.scientific_call import ErrorPolicy, ScientificCall

from application import ElucidationSession, ElucidationReport
from binding import Inference
from wire import ElucidationBatch, ElucidationBatchResult, ElucidationRequest, ElucidationResult, ModelFailure, Peaks
import main as provider_main
from provider import open_provider as configured_provider, batch_requests


class ProviderLifetimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_exclusion_remains_through_service_and_model_cleanup(self):
        for failing_phase in (None, "policy", "execution", "service", "model_cleanup"):
            with self.subTest(failing_phase=failing_phase), TemporaryDirectory() as directory:
                root = Path(directory)
                lifetime = ProviderLifetime(root)
                failure = RuntimeError("fixture unsettled cleanup")
                phases = []

                def observe_exclusion(phase):
                    phases.append(phase)

                    with self.assertRaises(BlockingIOError):
                        ProviderLifetime(root).open()

                    if failing_phase == phase:
                        raise failure

                @asynccontextmanager
                async def started_provider(deployment):
                    observe_exclusion("model_start")

                    try:
                        yield SimpleNamespace(inference=SimpleNamespace(host=object()), batches=nullcontext())
                    finally:
                        observe_exclusion("model_cleanup")

                def open_provider(deployment):
                    if failing_phase == "execution":
                        return configured_provider(deployment)

                    return started_provider(deployment)

                async def serve():
                    observe_exclusion("service")

                config = root / "provider.toml"

                config.write_text(
                    f'''
[api]
[interpreter]
[shared]
project = "test-delivery"
gid = 1000
[execution]
backend = "invalid-fixture-backend"
[offering]
analysis_kind = "mol_from_1h_13c_formula"
n_samples = 8
seed = 0
[service]
lifetime_directory = {json.dumps(str(root))}
maximum_preparations = 2
[service.limits]
maximum_attempts = 4
maximum_control_bytes = 524288
maximum_pending_terminal_bytes = 524288
maximum_input_bytes = 262144
[service.policy]
feed_seconds = {0 if failing_phase == "policy" else 1}
hello_seconds = 60
observation_seconds = 1
recovery_seconds = 10
retry_max_seconds = 2
settle_seconds = 2
''',
                    encoding="utf-8",
                )

                try:
                    with patch.object(provider_main.ProviderClients, "from_config", return_value=Mock()), \
                            patch.object(provider_main.InterpreterSettings, "from_config", return_value=Mock()), \
                            patch.object(provider_main, "termination_signals", return_value=nullcontext()), \
                            patch.object(provider_main, "ProviderLifetime", return_value=lifetime), \
                            patch.object(provider_main, "open_provider", open_provider), \
                            patch.object(provider_main, "JobService", return_value=SimpleNamespace(serve=serve)):
                        if failing_phase in ("policy", "execution"):
                            with self.assertRaises(ValueError):
                                await provider_main.serve(config=config)
                        elif failing_phase is None:
                            await provider_main.serve(config=config)
                        else:
                            with self.assertRaises(RuntimeError) as caught:
                                await provider_main.serve(config=config)

                            self.assertIs(caught.exception, failure)

                    expected = [] if failing_phase in ("policy", "execution") else ["model_start", "service", "model_cleanup"]

                    self.assertEqual(expected, phases)

                    if failing_phase in ("policy", "execution"):
                        self.assertFalse((root / ".checkpoint-owner.lock").exists())

                    if failing_phase in (None, "policy", "execution"):
                        ProviderLifetime(root).open().close()
                    else:
                        with self.assertRaises(BlockingIOError):
                            ProviderLifetime(root).open()
                finally:
                    lifetime.close()


class ElucidationSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scientific_result = ElucidationResult(
            candidates=[],
            implausible_candidates=[],
            n_samples=8,
            n_valid=0,
            n_unique_candidates=0,
            validity=0.0,
            status_counts={},
            timing_s={},
            ensemble={},
            params={},
        )

        self.result_codec = JsonDataclassCodec(format="test.result.v1", value_type=ElucidationResult)

        self.provider = SimpleNamespace(
            offering={"n_samples": 8, "seed": 0},
            inference=AsyncMock(spec=ProcessCall, return_value=ElucidationBatchResult(replies=[self.scientific_result])),
        )

        self.provider.inference.error_policy = ErrorPolicy()
        self.provider.inference._claim_batch = Mock()
        self.provider.inference._release_batch = Mock()
        retirement = SimpleNamespace(observe=AsyncMock(return_value=CleanupObservation(0, True, False, ())))
        self.provider.inference.host = SimpleNamespace(retire=lambda: retirement)
        self.batch_failure = None

        self.batch_policy = BatchPolicy(
            maximum_entries=4,
            collection_seconds=0.01,
            maximum_queued_input_bytes=32768,
            maximum_entry_input_bytes=8192,
        )

        self.provider.batches = batch_requests(self.provider.inference, policy=self.batch_policy)

        self.new_interpreter = Mock(side_effect=AssertionError("Structured input should not call an interpreter"))
        self.session = ElucidationSession(self.provider, new_interpreter=self.new_interpreter)
        self.condition = SimpleNamespace(require_execution=Mock())
        self.acknowledged = SimpleNamespace(
            start=SimpleNamespace(input="verified-input"),
            admission=SimpleNamespace(job_ref="job:fixture", execution_attempt_ref="attempt:fixture"),
        )

    async def asyncSetUp(self):
        await self.provider.batches.__aenter__()

    async def asyncTearDown(self):
        if self.batch_failure is None:
            await self.provider.batches.close()
        else:
            with self.assertRaises(self.batch_failure):
                await self.provider.batches.close()

    async def prepare_input(self, specification):
        instructions = SimpleNamespace(identity="verified-input", specification_bytes=json.dumps(specification).encode())

        return await self.session.prepare(self.acknowledged, instructions, report_condition=self.condition)

    async def test_job_uses_deployment_budget_and_returns_the_typed_result(self):
        prepared = await self.prepare_input({"formula": "CH4", "peaks": {"h": [[1.0, 4, "s"]]}})
        outcome = await self.session.execute(prepared, report_condition=self.condition)

        self.assertIsInstance(outcome, AnalysisSuccess)
        report = JsonDataclassCodec(format="chefnmr2.elucidation_report.v1", value_type=ElucidationReport).decode(outcome.content)

        self.assertEqual(self.scientific_result, report.elucidation)
        self.assertEqual("CH4", report.input.formula)
        self.new_interpreter.assert_not_called()

        invocation = self.provider.inference.call_args
        (elucidation,) = invocation.kwargs["scientific_input"].requests

        self.assertEqual((8, 0, "CH4"), (elucidation.n_samples, elucidation.seed, elucidation.formula))
        self.condition.require_execution.assert_called()

        await self.session.retire()

    async def test_job_cannot_override_the_deployment_sample_budget(self):
        outcome = await self.prepare_input({"formula": "CH4", "peaks": {}, "n_samples": 1000000})

        self.assertIsInstance(outcome, AnalysisFailure)
        self.assertEqual("invalid_input", outcome.code)
        self.provider.inference.assert_not_called()

    async def test_malformed_json_does_not_fall_back_to_prose(self):
        instructions = SimpleNamespace(identity="verified-input", specification_bytes=b'{"formula":')
        outcome = await self.session.prepare(self.acknowledged, instructions, report_condition=self.condition)

        self.assertIsInstance(outcome, AnalysisFailure)
        self.new_interpreter.assert_not_called()
        self.provider.inference.assert_not_called()

    def use_interpreter(self, proposals):
        endpoint = ScriptedEndpoint(proposals)

        self.session.new_interpreter = lambda: ManagedInterpreter(
            (endpoint,),
            interpretation_budget_seconds=5,
            reject_combined_messages_and_tools_over_bytes=32768,
            end_model_visit_after_turns=4,
            end_model_visit_after_rejections=3,
            total_http_attempts_per_turn=1,
            connection_timeout_seconds=1,
            retain_history_up_to_bytes=32768,
            retain_history_up_to_records=8,
            secrets=(),
        )

        return endpoint

    async def prepare_prose(self):
        instructions = SimpleNamespace(
            identity="verified-input",
            specification_bytes=b"Formula C2H6. Carbon shifts 10, 10 ppm. Proton peaks: 1 ppm, 3H, singlet; 2 ppm, 3H, singlet.",
        )

        return await self.session.prepare(self.acknowledged, instructions, report_condition=self.condition)

    async def test_prose_correction_uses_same_codec_and_reports_actual_input(self):
        supplied = {"formula": "C2H6", "peaks": {"c": [10, 10], "h": [[1, 3, "s"], [2, 3, "s"]]}}

        endpoint = self.use_interpreter(
            [
                ("submit_input", {**supplied, "n_samples": 1000}),
                ("submit_input", supplied),
            ],
        )

        prepared = await self.prepare_prose()

        self.provider.inference.assert_not_called()
        self.assertTrue(endpoint.retirement_confirmed)

        outcome = await self.session.execute(prepared, report_condition=self.condition)
        report = JsonDataclassCodec(format="chefnmr2.elucidation_report.v1", value_type=ElucidationReport).decode(outcome.content)

        self.assertEqual([10, 10], report.input.peaks.c)
        self.assertEqual([(1, 3, "s"), (2, 3, "s")], report.input.peaks.h)
        self.assertEqual(2, len(endpoint.requests))
        self.assertEqual(8, self.provider.inference.call_args.kwargs["scientific_input"].requests[0].n_samples)
        self.provider.inference.assert_awaited_once()

    async def test_model_problem_does_not_become_a_scientific_input_rejection(self):
        endpoint = self.use_interpreter([("report_interpretation_problem", {"explanation": "No molecular formula"})])
        outcome = await self.prepare_prose()

        self.assertIsInstance(outcome, AnalysisFailure)
        self.assertEqual("interpretation_failed", outcome.code)
        self.assertTrue(endpoint.retirement_confirmed)
        self.provider.inference.assert_not_called()

    async def test_exhausted_endpoint_does_not_start_science_or_claim_invalid_input(self):
        endpoint = self.use_interpreter([])
        endpoint.complete = AsyncMock(side_effect=ChatFailed("http_status", "Endpoint refused the request"))
        outcome = await self.prepare_prose()

        self.assertIsInstance(outcome, AnalysisFailure)
        self.assertEqual("interpretation_failed", outcome.code)
        self.assertTrue(endpoint.retirement_confirmed)
        self.provider.inference.assert_not_called()

    async def test_interpreter_failure_preserves_redacted_cause_and_attempt_identity(self):
        secret = "fixture-unlabelled-credential"
        self.session.diagnostic_secrets = (secret,)
        endpoint = self.use_interpreter([])
        endpoint.complete = AsyncMock(side_effect=ChatFailed("http_status", "Endpoint refused " + secret))

        with self.assertLogs("application", level="ERROR") as captured:
            outcome = await self.prepare_prose()

        encoded = outcome.evidence.content.decode()
        self.assertIn("ChatFailed", encoded)
        self.assertIn("Endpoint refused", encoded)
        self.assertNotIn(secret, encoded)
        self.assertIn("[redacted]", encoded)
        self.assertIn("input_digest", json.loads(encoded))
        self.assertIn("job:fixture", captured.output[0])
        self.assertIn("attempt:fixture", captured.output[0])
        self.assertNotIn(secret, captured.output[0])

    async def test_model_cause_reaches_attempt_through_batch_wire_without_private_input(self):
        def failed_model(*args, **kwargs):
            private_local = "PRIVATE-FRAME-LOCAL"
            try:
                raise ValueError("token=hidden-token")
            except ValueError as error:
                raise RuntimeError("could not reconstruct CH4 with [123.456, 234.567]") from error

        worker = Inference(checkpoint_dirs=[], device="cpu", recon_workers=1)
        worker.model = SimpleNamespace(elucidate=failed_model)
        batch = ElucidationBatch([ElucidationRequest(formula="CH4", peaks=Peaks(c=[123.456, 234.567]), n_samples=8, seed=0)])
        codec = JsonDataclassCodec(format="test.failure_batch.v1", value_type=ElucidationBatchResult)

        with patch.dict(sys.modules, {"chefnmr2.infer": SimpleNamespace(Peaks=Peaks)}):
            with self.assertLogs("binding", level="ERROR") as worker_logs:
                encoded = codec.encode(worker.elucidate_batch(batch=batch))

        decoded = codec.decode(encoded).replies[0]
        self.assertEqual("RuntimeError", decoded.exception_type)
        self.assertIn("could not reconstruct", decoded.message)
        cause = next(
            (
                relationship["failure"]
                for relationship in decoded.diagnostic["relationships"]
                if relationship["relationship"] == "cause"
            ),
        )
        self.assertEqual("ValueError", cause["exception_type"])
        self.assertTrue(decoded.diagnostic["frames"])

        self.provider.inference.return_value = codec.decode(encoded)
        prepared = await self.prepare_input({"formula": "CH4", "peaks": {"c": [123.456, 234.567]}})
        with self.assertLogs("application", level="ERROR") as attempt_logs:
            outcome = await self.session.execute(prepared, report_condition=self.condition)

        self.assertEqual("model_failed", outcome.code)
        self.assertEqual("chefnmr2.model_failure.v2", outcome.evidence.schema_id)
        self.assertEqual(decoded.diagnostic, json.loads(outcome.evidence.content)["diagnostic"])
        self.assertIn("job:fixture", attempt_logs.output[0])
        self.assertIn("attempt:fixture", attempt_logs.output[0])
        self.provider.inference.assert_awaited_once()
        for secret in ("hidden-token", "CH4", "123.456", "234.567", "PRIVATE-FRAME-LOCAL"):
            self.assertNotIn(secret, encoded.decode())
            self.assertNotIn(secret, worker_logs.output[0])
            self.assertNotIn(secret, outcome.evidence.content.decode())
            self.assertNotIn(secret, attempt_logs.output[0])

        await self.session.retire()

    async def test_revocation_during_interpretation_prevents_accepting_a_proposal(self):
        endpoint = self.use_interpreter([("submit_input", {"formula": "CH4", "peaks": {}})])
        complete = endpoint.complete

        async def revoke_before_reply(*args, **kwargs):
            self.condition.require_execution.side_effect = PermissionError("Job permission revoked")

            return await complete(*args, **kwargs)

        endpoint.complete = revoke_before_reply

        with self.assertRaises(PermissionError):
            await self.prepare_prose()

        self.assertTrue(endpoint.retirement_confirmed)
        self.provider.inference.assert_not_called()

    async def test_cancellation_retires_interpreter_without_starting_science(self):
        endpoint = self.use_interpreter([])
        started = asyncio.Event()

        async def held_completion(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        endpoint.complete = held_completion
        preparation = asyncio.create_task(self.prepare_prose())

        await asyncio.wait_for(started.wait(), timeout=2)
        preparation.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await preparation

        await self.session.retire()

        self.assertTrue(endpoint.retirement_confirmed)
        self.provider.inference.assert_not_called()

    async def test_model_diagnostic_is_a_failure_for_its_job(self):
        self.provider.inference.return_value = ElucidationBatchResult(replies=[ModelFailure("ValueError", "private diagnostic", {"exception_type": "ValueError", "message": "private diagnostic"})])
        prepared = await self.prepare_input({"formula": "CH4", "peaks": {}})
        outcome = await self.session.execute(prepared, report_condition=self.condition)

        self.assertIsInstance(outcome, AnalysisFailure)
        self.assertEqual("model_failed", outcome.code)
        self.assertNotIn("private diagnostic", outcome.message)
        self.assertEqual("private diagnostic", json.loads(outcome.evidence.content)["diagnostic"]["message"])

        await self.session.retire()

    async def test_missing_batch_reply_is_unresolved_not_a_success(self):
        self.batch_failure = ValueError

        self.provider.inference.return_value = ElucidationBatchResult(replies=[])
        prepared = await self.prepare_input({"formula": "CH4", "peaks": {}})

        with self.assertRaises(AnalysisOutcomeUnresolved):
            await self.session.execute(prepared, report_condition=self.condition)

    async def test_batch_wire_keeps_order_and_continues_after_model_failure(self):
        first = deepcopy(self.scientific_result)
        last = deepcopy(self.scientific_result)
        first.params = {"seed": 11}
        last.params = {"seed": 22}
        worker = Inference(checkpoint_dirs=[], device="cpu", recon_workers=1)
        worker.model = SimpleNamespace(elucidate=Mock(side_effect=[asdict(first), ValueError("bad formula"), asdict(last)]))
        request_codec = JsonDataclassCodec(format="test.batch_input.v1", value_type=ElucidationBatch)
        reply_codec = JsonDataclassCodec(format="test.batch_reply.v1", value_type=ElucidationBatchResult)

        batch = ElucidationBatch(
            requests=[
                ElucidationRequest(formula="CH4", peaks=Peaks(), n_samples=8, seed=11),
                ElucidationRequest(formula="invalid", peaks=Peaks(), n_samples=8, seed=0),
                ElucidationRequest(formula="C2H6", peaks=Peaks(), n_samples=16, seed=22),
            ],
        )

        with patch.dict(sys.modules, {"chefnmr2.infer": SimpleNamespace(Peaks=Peaks)}):
            with self.assertLogs("binding", level="ERROR"):
                replies = worker.elucidate_batch(batch=request_codec.decode(request_codec.encode(batch)))

        decoded = reply_codec.decode(reply_codec.encode(replies))

        self.assertEqual(first, decoded.replies[0])
        self.assertEqual(last, decoded.replies[2])
        failure = decoded.replies[1]
        self.assertEqual("bad formula", failure.message)
        self.assertEqual("ValueError", failure.diagnostic["exception_type"])
        self.assertTrue(failure.diagnostic["frames"])

        self.assertEqual(
            [("CH4", 8, 11), ("invalid", 8, 0), ("C2H6", 16, 22)],
            [
                (call.kwargs["formula"], call.kwargs["n_samples"], call.kwargs["seed"])
                for call in worker.model.elucidate.call_args_list
            ],
        )

        with self.assertRaises(PayloadRejected):
            request_codec.decode(b'{"formula":"CH4","peaks":{}}')

    async def use_scientific_call(self):
        await self.provider.batches.close()

        host = InProcessExecutionHost(
            input_bytes=8192,
            result_bytes=65536,
            state_values=1,
            progress_records=1,
            initialization_seconds=5,
            retirement_seconds=5,
        )

        science = ControlledInference(self.scientific_result)
        hosted = host.adopt(contribution=science, requirements=frozenset())

        scientific_call = ScientificCall(
            host=host,
            allocation=hosted.allocation,
            binding=hosted.call(method=ControlledInference.elucidate_batch),
            operation_policy=OperationPolicy(5, 5, 5, 8192, 65536),
            error_policy=ErrorPolicy(max_attempts=1),
        )

        # This fixture owns its embedded host and presents the ProcessCall port;
        # SDK native tests establish the real process owner's custody boundary.
        async def invoke_fixture(*, scientific_input, require_execution, _batch_owner):
            return await scientific_call(scientific_input=scientific_input, require_execution=require_execution)

        self.provider.inference = AsyncMock(spec=ProcessCall, side_effect=invoke_fixture)
        self.provider.inference.error_policy = scientific_call.error_policy
        self.provider.inference.host = host
        self.provider.inference._claim_batch = Mock()
        self.provider.inference._release_batch = Mock()
        self.provider.batches = batch_requests(self.provider.inference, policy=self.batch_policy)

        await self.provider.batches.__aenter__()

        return (host, science)

    async def test_cancellation_settles_original_science_and_keeps_model_reusable(self):
        (host, science) = await self.use_scientific_call()

        science.release.clear()
        prepared = await self.prepare_input({"formula": "CH4", "peaks": {}})

        async with host:
            execution = asyncio.create_task(self.session.execute(prepared, report_condition=self.condition))

            try:
                self.assertTrue(await asyncio.to_thread(science.entered.wait, timeout=2))
                execution.cancel()

                with self.assertRaises(asyncio.CancelledError):
                    await execution

                retiring = asyncio.create_task(self.session.retire())

                await asyncio.sleep(0)

                self.assertFalse(retiring.done())
            finally:
                science.release.set()
                await asyncio.wait_for(retiring, timeout=5)

            with self.assertRaises(asyncio.CancelledError):
                execution.result()

            await self.session.retire()

            self.assertEqual((1, 1), (science.calls, science.closed_analyses))
            self.assertEqual({}, host.analyses)
            self.assertFalse(science.closed)

            self.session = ElucidationSession(self.provider, new_interpreter=self.new_interpreter)

            prepared = await self.prepare_input({"formula": "C2H6", "peaks": {}})
            outcome = await self.session.execute(prepared, report_condition=self.condition)

            self.assertIsInstance(outcome, AnalysisSuccess)
            self.assertEqual((2, 2), (science.calls, science.closed_analyses))
            await self.provider.batches.close()

        self.assertTrue(science.closed)

    async def test_failed_call_is_unresolved_and_outer_context_retires_model(self):
        (host, science) = await self.use_scientific_call()
        failure = OSError("Scientific call failed")
        self.batch_failure = OSError

        science.failure = failure
        prepared = await self.prepare_input({"formula": "CH4", "peaks": {}})

        with self.assertRaises(AnalysisOutcomeUnresolved) as raised:
            async with host:
                try:
                    await self.session.execute(prepared, report_condition=self.condition)
                finally:
                    await self.session.retire()

                    with self.assertRaises(OSError):
                        await self.provider.batches.close()

                    self.assertEqual((1, 1), (science.calls, science.closed_analyses))
                    self.assertTrue(science.closed)

        self.assertIs(failure, raised.exception.__cause__)
        self.assertTrue(science.closed)

    async def test_three_jobs_share_a_call_and_keep_their_own_results(self):
        (host, science) = await self.use_scientific_call()
        sessions = [ElucidationSession(self.provider, new_interpreter=self.new_interpreter) for _ in range(3)]
        codec = JsonDataclassCodec(format="chefnmr2.elucidation_report.v1", value_type=ElucidationReport)

        async with host:
            prepared = []

            for (session, formula) in zip(sessions, ("CH4", "CH4", "BAD"), strict=True):
                instructions = SimpleNamespace(
                    identity="verified-input",
                    specification_bytes=json.dumps({"formula": formula, "peaks": {}}).encode(),
                )

                prepared.append(await session.prepare(self.acknowledged, instructions, report_condition=self.condition))

            outcomes = await asyncio.gather(
                *(
                    session.execute(handle, report_condition=self.condition)
                    for (session, handle) in zip(sessions, prepared, strict=True)
                ),
            )

            self.assertEqual([0, 1], [codec.decode(outcome.content).elucidation.params["batch_index"] for outcome in outcomes[:2]])
            self.assertIsInstance(outcomes[2], AnalysisFailure)
            self.assertEqual("model_failed", outcomes[2].code)
            self.assertEqual((1, 1), (science.calls, science.closed_analyses))
            self.assertEqual([["CH4", "CH4", "BAD"]], science.batch_inputs)
            await asyncio.gather(*(session.retire() for session in sessions))
            await self.provider.batches.close()


class ControlledAnalysis(Analysis):
    def close(self) -> None:
        self.contribution.closed_analyses += 1


class ControlledInference(Contribution):
    """Use real SDK execution and cleanup with a controllable model substitute."""

    analysis_type = ControlledAnalysis

    def __init__(self, elucidation_result):
        super().__init__()

        self.elucidation_result = elucidation_result
        self.entered = Event()
        self.release = Event()

        self.release.set()
        self.failure = None
        self.calls = 0
        self.batch_inputs = []
        self.closed_analyses = 0
        self.closed = False

    @operation()
    def elucidate_batch(self, *, batch: ElucidationBatch) -> ElucidationBatchResult:
        self.calls += 1

        self.batch_inputs.append([request.formula for request in batch.requests])

        self.entered.set()

        if not self.release.wait(timeout=(wait_seconds := 5)):
            raise TimeoutError(f"{self.release.is_set()=} after {wait_seconds=}")

        if self.failure is not None:
            raise self.failure

        return ElucidationBatchResult(
            replies=[
                ModelFailure("ValueError", "fixture bad formula", {})
                if request.formula == "BAD"
                else replace(self.elucidation_result, params={"batch_index": index})
                for (index, request) in enumerate(batch.requests)
            ],
        )

    def close(self) -> None:
        self.closed = True


class ScriptedEndpoint:
    """Replace only model HTTP; use the real SDK conversation and tool dispatch."""

    context_bytes = 32768
    identity = b'{"configuration_id":"test","model":"scripted"}'

    def __init__(self, proposals):
        self.proposals = iter(proposals)
        self.requests = []
        self.retirement_confirmed = False

    def __call__(self, **kwargs):
        return self

    async def complete(self, messages, tools, *, not_after):
        self.requests.append(deepcopy(messages))
        (name, arguments) = next(self.proposals)

        return {
            "role": "assistant",
            "tool_calls": [{"id": str(len(self.requests)), "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}],
        }

    async def retire(self):
        self.retirement_confirmed = True


if __name__ == "__main__":
    unittest.main()
