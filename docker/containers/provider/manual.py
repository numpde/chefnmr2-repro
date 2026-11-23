"""Manually exercise the provider with JSON requests on stdin and replies on stdout."""

import argparse
import asyncio
import json
import logging
import pathlib
import sys

if __package__:
    from .provider import open_provider
else:
    from provider import open_provider


async def main(*, config: pathlib.Path) -> None:
    """Read one JSON request per line; keep the loaded model until stdin closes."""

    async with open_provider(config) as provider:
        call = provider.inference.binding

        for line in sys.stdin:
            document = json.loads(line)

            document.setdefault("n_samples", provider.offering["n_samples"])
            document.setdefault("seed", provider.offering["seed"])

            elucidation = call.input_codec.decode(json.dumps(document).encode())

            result = await provider.elucidate(elucidation)

            # This CLI replies on stdout, one JSON line per input request
            # Flush each reply so a piped client need not wait for the buffer to fill
            print(call.result_codec.encode(result).decode(), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s pid=%(process)d %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--config", required=True, type=pathlib.Path)
    args = vars(parser.parse_args())

    asyncio.run(main(**args))
