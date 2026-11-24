"""Run ChefNMR2 operations through the SDK's native scientific host."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

from provider_core.batch_call import BatchCall, BatchPolicy
from provider_core.execution_setup import ExecutionSettings
from provider_core.payload import JsonDataclassCodec
from provider_core.process_call import ProcessCall

if __package__:
    from model.binding import Inference, construct
    from model.wire import ElucidationBatch, ElucidationRequest
else:
    from binding import Inference, construct
    from wire import ElucidationBatch, ElucidationRequest


@dataclass
class Provider:
    inference: ProcessCall
    offering: dict
    batches: BatchCall


def open_provider(deployment: dict) -> AbstractAsyncContextManager[Provider]:
    """Configure the offering now; entering the returned scope starts the model."""

    execution = ExecutionSettings(deployment["execution"])

    inference = execution.open_process_call(
        runner=deployment["runner"],
        contribution_type=Inference,
        constructor=construct,
        startup=deployment["model"],
        method=Inference.elucidate_batch,
    )

    batches = batch_requests(inference, policy=BatchPolicy(**deployment["batch"]))

    return _started_provider(Provider(inference=inference, offering=deployment["offering"], batches=batches))


def batch_requests(inference, *, policy):
    """Chef owns the wire schema; the SDK owns collection, routing and settlement."""

    return BatchCall(
        call=inference,
        policy=policy,
        entry_codec=JsonDataclassCodec(format="chefnmr2.request.v1", value_type=ElucidationRequest),
        pack=lambda requests: ElucidationBatch(requests=list(requests)),
        unpack=lambda batch_result: batch_result.replies,
    )


@asynccontextmanager
async def _started_provider(provider: Provider) -> AsyncIterator[Provider]:
    async with provider.inference:
        yield provider
