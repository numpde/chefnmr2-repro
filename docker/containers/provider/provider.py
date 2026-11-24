"""Run ChefNMR2 operations through the SDK's native scientific host."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

from provider_core.execution_setup import ExecutionSettings
from provider_core.process_call import ProcessCall

if __package__:
    from .._shared.binding import Inference, construct
else:
    from binding import Inference, construct


@dataclass
class Provider:
    inference: ProcessCall
    offering: dict


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

    return _started_provider(Provider(inference=inference, offering=deployment["offering"]))


@asynccontextmanager
async def _started_provider(provider: Provider) -> AsyncIterator[Provider]:
    async with provider.inference:
        yield provider
