"""Run ChefNMR2 operations through the SDK's native scientific host."""

import argparse
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import os
import pathlib
import stat
import sys
import tomllib

from provider_core.execution_setup import ExecutionSettings
from provider_core.failure_collection import FailureCollector
from provider_core.hosting import NotAdmitted, OperationPolicy, ResultConsumption, settle_cleanup
from provider_core.process_host import ProcessExecutionHost
from provider_core.runner_session import RunnerSession
from provider_core.scientific import HostedContribution
from provider_core.submission import Submission

if __package__:
    from .._shared.binding import Inference
    from .._shared.wire import ElucidationRequest, ElucidationResult
else:
    from binding import Inference
    from wire import ElucidationRequest, ElucidationResult


@dataclass
class Provider:
    execution_host: ProcessExecutionHost
    hosted_predictor: HostedContribution
    operation_policy: OperationPolicy
    offering: dict

    async def elucidate(self, elucidation: ElucidationRequest) -> ElucidationResult:
        """Consume one typed result and release its analysis, including on failure."""
        analysis = self.execution_host.open_analysis(allocations=frozenset({self.hosted_predictor.allocation}))
        call = self.hosted_predictor.call(method=Inference.elucidate)
        decoded: ElucidationResult

        async def receive(encoded: bytes) -> None:
            nonlocal decoded

            decoded = call.result_codec.decode(encoded)

        failures = FailureCollector()

        with failures.capture():
            submission = Submission.from_call(
                analysis_executor=analysis,
                call_binding=call,
                scientific_input=elucidation,
                states={},
                reservation_ref=analysis.reservation(allocation=self.hosted_predictor.allocation),
                operation_policy=self.operation_policy,
            )

            operation = submission.submit()

            if isinstance(operation, NotAdmitted):
                raise RuntimeError(f"Inference was not admitted: {operation.code.value}: {operation.message}")

            observation = await submission.wait()

            async with ResultConsumption(
                operation=operation,
                result=observation.require_result(),
                receive=receive,
            ) as consumption:
                await consumption.consume()

        with failures.capture():
            await settle_cleanup(control=analysis.close())

        failures.raise_if_any()

        return decoded


@asynccontextmanager
async def open_provider(deployment_config_path: pathlib.Path) -> AsyncIterator[Provider]:
    """Start the model and transfer the runner session to its SDK host."""

    with deployment_config_path.open("rb") as configuration:
        deployment = tomllib.load(configuration)

    execution = ExecutionSettings(deployment["execution"])

    if execution.backend != "process":
        raise ValueError(f"ChefNMR2 requires process execution; {execution.backend=}")

    runner = deployment["runner"]
    lease_fd = os.open(runner["lease_file"], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)

    try:
        lease = os.fstat(lease_fd)

        if not (
            stat.S_ISREG(lease.st_mode) and
            (lease.st_uid == os.geteuid()) and
            (stat.S_IMODE(lease.st_mode) == 0o600)
        ):
            raise PermissionError(f"Expected an owner-only lease file; {lease=}")
    finally:
        os.close(lease_fd)

    session = RunnerSession(
        runner["socket"],
        role=runner["role"],
        expected_lock_identity=(lease.st_dev, lease.st_ino),
        startup_seconds=runner["startup_seconds"],
        retirement_seconds=runner["retirement_seconds"],
    )

    try:
        host = execution.create_host(process_session=session)
    except BaseException:
        session.close()

        raise

    async with host:
        predictor = await host.start(
            contribution_type=Inference,
            constructor=f"{Inference.__module__}:construct",
            startup=deployment["model"],
            requirements=execution.resource_requirements,
        )

        yield Provider(host, predictor, execution.operation_policy, deployment["offering"])


async def main(*, config: pathlib.Path) -> None:
    """Read one JSON request per line; keep the loaded model until stdin closes."""

    async with open_provider(config) as provider:
        call = provider.hosted_predictor.call(method=Inference.elucidate)

        for line in sys.stdin:
            document = json.loads(line)

            document.setdefault("n_samples", provider.offering["n_samples"])
            document.setdefault("seed", provider.offering["seed"])
            elucidation = call.input_codec.decode(json.dumps(document).encode())
            result = await provider.elucidate(elucidation)

            print(call.result_codec.encode(result).decode(), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s pid=%(process)d %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--config", required=True, type=pathlib.Path)
    args = vars(parser.parse_args())

    asyncio.run(main(**args))
