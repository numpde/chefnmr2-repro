"""The manual client applies only the model's declared request defaults."""

from contextlib import asynccontextmanager, redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from provider_core.payload import JsonDataclassCodec

import manual
from wire import ElucidationBatch, ElucidationBatchResult, ModelFailure


class ManualClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_defaults_do_not_replace_explicit_request_settings(self):
        input_codec = JsonDataclassCodec(format="test.manual_batch.v1", value_type=ElucidationBatch)
        result_codec = JsonDataclassCodec(format="test.manual_reply.v1", value_type=ElucidationBatchResult)
        result = ElucidationBatchResult(replies=[ModelFailure("ValueError", "fixture", {}) for _ in range(2)])
        inference = AsyncMock(return_value=result)
        inference.binding = SimpleNamespace(input_codec=input_codec, result_codec=result_codec)

        @asynccontextmanager
        async def open_fixture(deployment):
            yield SimpleNamespace(inference=inference, offering={"n_samples": 8, "seed": 0, "analysis_kind": "fixture"})

        batch = {"requests": [
            {"formula": "CH4", "peaks": {}},
            {"formula": "C2H6", "peaks": {}, "n_samples": 16, "seed": None},
        ]}
        stdout = io.StringIO()

        with TemporaryDirectory() as directory:
            config = Path(directory) / "provider.toml"
            config.write_text("")

            with patch.object(manual, "open_provider", open_fixture), patch.object(sys, "stdin", io.StringIO(json.dumps(batch) + "\n")), redirect_stdout(stdout):
                await manual.main(config=config)

        inference.assert_awaited_once()
        requests = inference.call_args.kwargs["scientific_input"].requests
        self.assertEqual([(r.formula, r.n_samples, r.seed) for r in requests], [("CH4", 8, 0), ("C2H6", 16, None)])
        self.assertEqual(result_codec.decode(stdout.getvalue().strip().encode()), result)
