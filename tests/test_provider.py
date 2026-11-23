"""Exercise the API science adapter inside the provider image, without a model."""

import asyncio
from copy import deepcopy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from provider_core.analysis import AnalysisFailure, AnalysisOutcomeUnresolved, AnalysisSuccess
from provider_core.chat_errors import ChatFailed
from provider_core.hosting import CleanupObservation
from provider_core.interpreter import Interpreter
from provider_core.payload import JsonDataclassCodec

from application import ElucidationSession, ElucidationReport
from wire import ElucidationResult


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

        cleanup = SimpleNamespace(observe=AsyncMock(return_value=CleanupObservation(1, True, False, ())))
        self.host = SimpleNamespace(retire=Mock(return_value=cleanup))
        self.result_codec = JsonDataclassCodec(format="test.result.v1", value_type=ElucidationResult)

        self.provider = SimpleNamespace(
            offering={"n_samples": 8, "seed": 0},
            elucidate=AsyncMock(return_value=self.scientific_result),
            inference=SimpleNamespace(binding=SimpleNamespace(result_codec=self.result_codec), host=self.host),
        )

        self.new_interpreter = Mock(side_effect=AssertionError("Structured input should not call an interpreter"))
        self.session = ElucidationSession(self.provider, new_interpreter=self.new_interpreter)
        self.condition = SimpleNamespace(require_execution=Mock())
        self.acknowledged = SimpleNamespace(start=SimpleNamespace(input="verified-input"))

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

        invocation = self.provider.elucidate.call_args
        elucidation = invocation.args[0]

        self.assertEqual((8, 0, "CH4"), (elucidation.n_samples, elucidation.seed, elucidation.formula))
        self.assertIs(self.condition.require_execution, invocation.kwargs["require_execution"])

        await self.session.retire()

        self.host.retire.assert_not_called()

    async def test_job_cannot_override_the_deployment_sample_budget(self):
        outcome = await self.prepare_input({"formula": "CH4", "peaks": {}, "n_samples": 1000000})

        self.assertIsInstance(outcome, AnalysisFailure)
        self.assertEqual("invalid_input", outcome.code)
        self.provider.elucidate.assert_not_called()

    async def test_malformed_json_does_not_fall_back_to_prose(self):
        instructions = SimpleNamespace(identity="verified-input", specification_bytes=b'{"formula":')
        outcome = await self.session.prepare(self.acknowledged, instructions, report_condition=self.condition)

        self.assertIsInstance(outcome, AnalysisFailure)
        self.new_interpreter.assert_not_called()
        self.provider.elucidate.assert_not_called()

    def use_interpreter(self, proposals):
        endpoint = ScriptedEndpoint(proposals)

        self.session.new_interpreter = lambda: Interpreter(
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

        self.provider.elucidate.assert_not_called()
        self.assertTrue(endpoint.retirement_confirmed)

        outcome = await self.session.execute(prepared, report_condition=self.condition)
        report = JsonDataclassCodec(format="chefnmr2.elucidation_report.v1", value_type=ElucidationReport).decode(outcome.content)

        self.assertEqual([10, 10], report.input.peaks.c)
        self.assertEqual([(1, 3, "s"), (2, 3, "s")], report.input.peaks.h)
        self.assertEqual(2, len(endpoint.requests))
        self.assertEqual(8, self.provider.elucidate.call_args.args[0].n_samples)
        self.provider.elucidate.assert_awaited_once()

    async def test_model_problem_does_not_become_a_scientific_input_rejection(self):
        endpoint = self.use_interpreter([("report_interpretation_problem", {"explanation": "No molecular formula"})])
        outcome = await self.prepare_prose()

        self.assertIsInstance(outcome, AnalysisFailure)
        self.assertEqual("interpretation_failed", outcome.code)
        self.assertTrue(endpoint.retirement_confirmed)
        self.provider.elucidate.assert_not_called()

    async def test_exhausted_endpoint_does_not_start_science_or_claim_invalid_input(self):
        endpoint = self.use_interpreter([])
        endpoint.complete = AsyncMock(side_effect=ChatFailed("http_status", "Endpoint refused the request"))
        outcome = await self.prepare_prose()

        self.assertIsInstance(outcome, AnalysisFailure)
        self.assertEqual("interpretation_failed", outcome.code)
        self.assertTrue(endpoint.retirement_confirmed)
        self.provider.elucidate.assert_not_called()

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
        self.provider.elucidate.assert_not_called()

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
        self.provider.elucidate.assert_not_called()
        self.host.retire.assert_not_called()

    async def test_unknown_scientific_failure_keeps_its_cause_and_retires_the_host(self):
        failure = OSError("Scientific connection failed")
        self.provider.elucidate.side_effect = failure
        prepared = await self.prepare_input({"formula": "CH4", "peaks": {}})

        with self.assertRaises(AnalysisOutcomeUnresolved) as raised:
            await self.session.execute(prepared, report_condition=self.condition)

        self.assertIs(failure, raised.exception.__cause__)

        await self.session.retire()

        self.host.retire.assert_called_once()

    async def test_cancelling_the_observer_leaves_original_science_for_retirement(self):
        started = asyncio.Event()
        retired = asyncio.Event()

        async def held_science(*args, **kwargs):
            started.set()

            try:
                await asyncio.Event().wait()
            finally:
                retired.set()

        self.provider.elucidate.side_effect = held_science
        prepared = await self.prepare_input({"formula": "CH4", "peaks": {}})
        observer = asyncio.create_task(self.session.execute(prepared, report_condition=self.condition))

        await asyncio.wait_for(started.wait(), timeout=2)
        observer.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await observer

        self.assertFalse(retired.is_set())

        await asyncio.wait_for(self.session.retire(), timeout=2)

        self.assertTrue(retired.is_set())
        self.host.retire.assert_called_once()


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
