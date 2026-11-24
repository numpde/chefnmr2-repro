"""Provider test fixtures using real SDK batching and controlled science/HTTP."""

from copy import deepcopy
from dataclasses import replace
import json
from threading import Event
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from provider_core.batch_call import BatchPolicy
from provider_core.embedded_host import InProcessExecutionHost
from provider_core.hosting import CleanupObservation, OperationPolicy
from provider_core.interpreter_settings import ManagedInterpreter
from provider_core.payload import JsonDataclassCodec
from provider_core.process_call import ProcessCall
from provider_core.scientific import Analysis, Contribution, operation
from provider_core.scientific_call import ErrorPolicy, ScientificCall
from application import ElucidationSession
from wire import ElucidationBatch, ElucidationBatchResult, ElucidationResult, ModelFailure
from provider import batch_requests


class ProviderCase(unittest.IsolatedAsyncioTestCase):
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
