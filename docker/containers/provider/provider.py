"""Run ChefNMR2 operations through the SDK's native scientific host."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from provider_core.execution_setup import ExecutionSettings
from provider_core.scientific_call import ScientificCall

if __package__:
    from .._shared.binding import Inference, construct
    from .._shared.wire import ElucidationBatch, ElucidationBatchResult
else:
    from binding import Inference, construct
    from wire import ElucidationBatch, ElucidationBatchResult


@dataclass
class Provider:
    inference: ScientificCall[ElucidationBatch, ElucidationBatchResult]
    offering: dict


@asynccontextmanager
async def open_provider(deployment: dict) -> AsyncIterator[Provider]:
    """Configure the ChefNMR2 offering for the selected deployment."""

    execution = ExecutionSettings(deployment["execution"])

    async with execution.open_process_call(
            runner=deployment["runner"],
            contribution_type=Inference,
            constructor=construct,
            startup=deployment["model"],
            method=Inference.elucidate_batch,
    ) as inference:
        yield Provider(inference=inference, offering=deployment["offering"])
