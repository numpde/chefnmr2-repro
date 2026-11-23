"""Exercise the API science adapter inside the provider image, without a model."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from provider_core.analysis import AnalysisFailure, AnalysisOutcomeUnresolved, AnalysisSuccess
from provider_core.hosting import CleanupObservation
from provider_core.payload import JsonDataclassCodec

from application import ElucidationSession
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

        self.session = ElucidationSession(self.provider)
        self.condition = SimpleNamespace(require_execution=Mock())
        self.acknowledged = SimpleNamespace(start=SimpleNamespace(input="verified-input"))

    async def prepare_input(self, specification):
        instructions = SimpleNamespace(identity="verified-input", specification_bytes=json.dumps(specification).encode())

        return await self.session.prepare(self.acknowledged, instructions, report_condition=self.condition)

    async def test_job_uses_deployment_budget_and_returns_the_typed_result(self):
        prepared = await self.prepare_input({"formula": "CH4", "peaks": {"h": [[1.0, 4, "s"]]}})
        outcome = await self.session.execute(prepared, report_condition=self.condition)

        self.assertIsInstance(outcome, AnalysisSuccess)
        self.assertEqual(self.scientific_result, self.result_codec.decode(outcome.content))

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


if __name__ == "__main__":
    unittest.main()
