"""Run the ChefNMR2 NMR API provider using the selected deployment."""

import argparse
import asyncio
from dataclasses import replace
from hashlib import sha256
import logging
import pathlib
import signal
import tomllib

from provider_core.api import ProviderAPI
from provider_core.api_config import api_settings_with_material
from provider_core.job_service import JobService, JobServicePolicy
from provider_core.lifetime import ProviderLifetime
from provider_core.shared_delivery_client import SharedDeliveryClient
from provider_core.shared_grants import SharedGrants
from provider_core.startup_file import read_startup_file
from provider_core.validation.upstream.nmr.v1.hello import AnalysisOffering

if __package__:
    from .application import ElucidationSession
    from .provider import open_provider
    from .job_input import configure_interpreter
else:
    from application import ElucidationSession
    from provider import open_provider
    from job_input import configure_interpreter


async def serve(*, config: pathlib.Path) -> None:
    """Keep the model loaded while the SDK job service consumes API work."""
    configuration = config.read_bytes()
    deployment = tomllib.loads(configuration.decode())
    (settings, provider_ref, publication) = api_settings_with_material(deployment["api"], read_startup_file)
    grants = SharedGrants(**deployment["shared"])
    settings = replace(settings, http=replace(settings.http, proxy_socket=str(grants.proxy_socket)))
    new_interpreter = configure_interpreter(deployment["interpreter"])
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()

    for number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(number, stopping.set)

    def new_api():
        return ProviderAPI(settings.identity, settings.http.new_transport(), expected_topology=settings.expected_topology)

    def new_delivery():
        return SharedDeliveryClient(
            grants.endpoint(settings.identity.origin, provider_ref),
            publication,
            exchange_seconds=settings.http.exchange_seconds,
            retirement_seconds=settings.http.retirement_seconds,
        )

    lifetime = ProviderLifetime(pathlib.Path(deployment["service"]["lifetime_directory"])).open()

    try:
        async with open_provider(config) as provider:
            service = JobService(
                provider_ref=provider_ref,
                consumer_id="chefnmr2",
                generation=sha256(configuration).hexdigest(),
                offering=AnalysisOffering(provider.offering["analysis_kind"], provider.offering["description"]),
                display_name="ChefNMR2",
                description="Molecular structure candidates from a formula and unassigned proton/carbon NMR peaks. https://github.com/numpde/chefnmr2-repro",
                host=provider.inference.host,
                policy=JobServicePolicy(**deployment["service"]["policy"]),
                new_session=lambda: ElucidationSession(provider, new_interpreter=new_interpreter),
                new_api=new_api,
                new_delivery=new_delivery,
                stopping=stopping,
            )

            await service.serve()
    finally:
        lifetime.close()

        for number in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(number)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s pid=%(process)d %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--config", required=True, type=pathlib.Path)
    args = vars(parser.parse_args())

    asyncio.run(serve(**args))
