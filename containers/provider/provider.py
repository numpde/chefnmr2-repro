"""Run the model operations through the SDK's native scientific host."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

from provider_core.batch_call import BatchCall, BatchPolicy
from provider_core.execution_setup import ExecutionSettings
from provider_core.process_call import ProcessCall

if __package__:
    from model.binding import Inference, construct, batch_operation, batch_type, batch_entry_codec
else:
    from binding import Inference, construct, batch_operation, batch_type, batch_entry_codec


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
        method=batch_operation,
    )

    batches = batch_requests(inference, policy=BatchPolicy(**deployment["batch"]))

    return _started_provider(Provider(inference=inference, offering=deployment["offering"], batches=batches))


def batch_requests(inference, *, policy):
    """The binding owns the wire schema; the SDK owns collection, routing and settlement."""

    return BatchCall(
        call=inference,
        policy=policy,
        entry_codec=batch_entry_codec(),
        pack=lambda requests: batch_type(requests=list(requests)),
        unpack=lambda batch_result: batch_result.replies,
    )


@asynccontextmanager
async def _started_provider(provider: Provider) -> AsyncIterator[Provider]:
    async with provider.inference:
        yield provider
