"""Normative provider behavior through the SDK adapter."""

from provider_core.analysis import AnalysisSuccess
from provider_core.payload import JsonDataclassCodec
from application import ElucidationReport
from provider_fixtures import ProviderCase


class ProviderNormativeTests(ProviderCase):
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
