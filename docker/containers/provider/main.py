"""Run the ChefNMR2 NMR API provider using the selected deployment."""

import argparse
import asyncio
from hashlib import sha256
import logging
import pathlib
import tomllib

from provider_core.interpreter_settings import InterpreterSettings
from provider_core.provider_clients import ProviderClients
from provider_core.process_signals import termination_signals
from provider_core.job_service import JobService, JobServicePolicy
from provider_core.lifetime import ProviderLifetime
from provider_core.shared_grants import SharedGrants
from provider_core.validation.upstream.nmr.v1.hello import AnalysisOffering

if __package__:
    from .application import ElucidationSession
    from .provider import open_provider
else:
    from application import ElucidationSession
    from provider import open_provider


async def serve(*, config: pathlib.Path) -> None:
    """Keep the model loaded while the SDK job service consumes API work."""
    configuration = config.read_bytes()
    deployment = tomllib.loads(configuration.decode())
    about = pathlib.Path(__file__).with_name("about")
    provider_description = (about / "provider.md").read_text(encoding="utf-8").strip()
    offering_description = (about / "hello.md").read_text(encoding="utf-8").strip().format_map(deployment["offering"])
    generation = sha256(configuration + b"\0" + provider_description.encode() + b"\0" + offering_description.encode()).hexdigest()
    clients = ProviderClients.from_config(deployment["api"], grants=SharedGrants(**deployment["shared"]))
    new_interpreter = InterpreterSettings.from_config(deployment["interpreter"]).new_interpreter
    stopping = asyncio.Event()
    policy = JobServicePolicy(**deployment["service"]["policy"])
    offering = AnalysisOffering(deployment["offering"]["analysis_kind"], offering_description)

    with termination_signals(stopping.set):
        lifetime = ProviderLifetime(pathlib.Path(deployment["service"]["lifetime_directory"])).open()
        async with open_provider(deployment) as provider:
            service = JobService(
                provider_ref=clients.provider_ref,
                consumer_id="chefnmr2",
                generation=generation,
                offering=offering,
                display_name="ChefNMR2",
                description=provider_description,
                host=provider.inference.host,
                policy=policy,
                new_session=lambda: ElucidationSession(provider, new_interpreter=new_interpreter),
                new_api=clients.new_api,
                new_delivery=clients.new_delivery,
                stopping=stopping,
            )

            await service.serve()
        # Release only after both the service and model contexts complete.
        # A failed cleanup retains exclusion until this process exits.
        lifetime.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s pid=%(process)d %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--config", required=True, type=pathlib.Path)
    args = vars(parser.parse_args())

    asyncio.run(serve(**args))
