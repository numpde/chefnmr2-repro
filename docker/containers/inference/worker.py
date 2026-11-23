"""Serve one client session on the inference socket inherited from the SDK runner.

Load ChefNMR2 once, process successive requests sequentially, and send each reply
on the same connection. The runner owns worker startup and retirement. This
process retains the inherited lifetime lease until it exits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from functools import wraps
from multiprocessing.connection import Connection
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

if __package__:
    from .._shared.wire import ElucidationResult, ElucidationRequest, ModelFailure
else:
    from wire import ElucidationResult, ElucidationRequest, ModelFailure

if TYPE_CHECKING:
    from chefnmr2.infer import ChefNMR2


def swallow_any[**P, R](*exceptions: type[Exception]) -> Callable[[Callable[P, R]], Callable[P, R | None]]:
    """Log the traceback and return None when the function raises a matching exception."""

    def decorate(function: Callable[P, R]) -> Callable[P, R | None]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R | None:
            try:
                return function(*args, **kwargs)
            except exceptions:
                log.exception(f"{function.__qualname__} failed")
                return None

        return wrapped

    return decorate


@contextmanager
def load_model(
        checkpoints: list[tuple[str, str | os.PathLike[str], str | os.PathLike[str]]],
        *,
        device: str,
        recon_workers: int | None,
) -> Iterator[ChefNMR2]:
    """Load ChefNMR2 from the checkpoints on the requested device and yield it.

    When the caller leaves the with block, request shutdown of the model's
    reconstruction pool if it still exists, including when the block raises.
    """

    from chefnmr2.infer import ChefNMR2

    log.info(f"Loading {len(checkpoints)=} on {device=} with {recon_workers=}")
    model = ChefNMR2(checkpoints, device=device, recon_workers=recon_workers)

    try:
        yield model
    finally:

        # If `python worker.py` is killed with SIGKILL, the finally block never runs.
        # If Compose service uses `init: true` then `docker-init` is PID 1 inside the container.
        # When its main child, the SDK supervisor, dies, docker-init exits.
        # Linux then kills any pool processes still running in that PID namespace.

        # `init: true` tells Docker to start a small program called docker-init first.
        # Linux assigns that first process PID 1 inside the container. docker-init starts the SDK supervisor, which starts python worker.py. The process tree looks like this:
        #
        # docker-init                 PID 1
        # └── SDK supervisor          owns the worker session
        #     └── python worker.py    serves inference requests
        #         └── ReconPool processes reconstruct molecules
        #
        # Docker forwards the stop signal through docker-init to the SDK supervisor.
        # The supervisor terminates and reaps worker.py during session retirement.
        # docker-init exits when the supervisor exits.
        # Linux then kills any processes left in the container’s PID namespace.

        if model.recon_pool is None:
            # This can happen in two cases:
            #
            # 1. With recon_workers=0, ChefNMR2 reconstructs structures in the
            #    calling process and never creates a pool.
            #    https://github.com/numpde/fork-of-chefnmr2/blob/29374e57952684533d266832288413ba87c5d809/chefnmr2/infer/api.py#L444-L446
            # 2. If collecting pool results raises an exception, ChefNMR2 tries
            #    to shut down the pool, sets recon_pool to None, and continues
            #    reconstruction in the calling process
            #    https://github.com/numpde/fork-of-chefnmr2/blob/29374e57952684533d266832288413ba87c5d809/chefnmr2/infer/api.py#L541-L566
            pass
        else:
            log.info("Shutting down reconstruction pool")
            model.recon_pool.shutdown()


@swallow_any(Exception)
def receive_message(connection: Connection, max_frame_length=2 ** 20) -> ElucidationRequest | None:
    """Read one inference request from a connected client and return an ElucidationRequest or None.

    This call blocks until the client sends a complete request or closes its
    sending side. There is no read timeout.

    Request frames are limited to max_frame_length bytes, which defaults to 1 MiB.

    If the client closes between requests, return None.

    If the initial socket check fails, log the error and return None to end the session.

    If a byte is available, the initial check leaves it in place so
    connection.recv_bytes() can read the length header and the complete message.

    If the client closes after sending part of the header or part of the
    payload, recv_bytes() raises OSError. If it sends the complete header
    for a nonempty payload and closes before sending any payload bytes,
    recv_bytes() raises EOFError. In either case, log the failed frame read
    and return None to end the session.

    Example JSON payload:

        {
            "formula": "CH4",
            "peaks": {
                "h": [[1.0, 4, "s"]]
            }
        }

    For this payload, receive_message() returns:

        ElucidationRequest(
            peaks=Peaks(c=[], h=[(1.0, 4, "s")]),
            formula="CH4",
            n_samples=100,
            seed=0,
        )
    """

    try:
        # Peek without consuming the first byte of the next frame. EOF here is clean.
        # Once a byte is present, EOF before recv_bytes() finishes that frame means truncation.
        with socket.fromfd(connection.fileno(), socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            if not sock.recv(1, socket.MSG_PEEK):
                log.info("Client closed its sending side between requests")
                return None
    except OSError as error:
        log.warning(f"Could not check for the next request; {error = }")
        return None

    try:
        request_frame = connection.recv_bytes(maxlength=max_frame_length)
        log.info(f"Received request frame of {len(request_frame)} bytes")
    except EOFError:
        log.warning("Client closed before sending the complete request frame")
        return None
    except OSError as error:
        log.warning(f"Could not read the request frame; {error = }")
        return None

    try:
        obj = json.loads(request_frame.decode("utf-8"))
    except (ValueError, RecursionError) as error:
        log.warning(f"Could not decode the request frame as UTF-8 JSON; {error = }")
        return None

    # Invalid request fields are logged by the constructor. Its None result
    # tells the serving loop to end this session.
    return ElucidationRequest.from_json_or_none(obj)


def process_request(model: ChefNMR2, request: ElucidationRequest) -> ElucidationResult | ModelFailure:
    """Run one request on the loaded model and return its typed result.

    Convert our peaks into upstream's Peaks and call model.elucidate() with
    the formula, sample count, and seed. An exception during inference or
    result construction becomes a ModelFailure carrying its type and message.
    """

    from chefnmr2.infer import Peaks, ChefNMR2

    assert type(model) == ChefNMR2
    assert type(request) == ElucidationRequest

    try:
        log.info(f"Starting `model.elucidate` with {request.n_samples=} and {request.seed=}")

        result = model.elucidate(
            Peaks(c=request.peaks.c, h=request.peaks.h),
            formula=request.formula,
            n_samples=request.n_samples,
            seed=request.seed,
        )

        return ElucidationResult(**result)
    except Exception as error:
        return ModelFailure(exception_type=type(error).__name__, message=str(error))


@swallow_any(Exception)
def serve_connection(connection: Connection, model: ChefNMR2) -> None:
    """Read requests sequentially and send each JSON reply on the same connection.

    A ModelFailure is a reply, so the client can submit another request after
    receiving one. If receive_message() returns None because the client
    disconnected, reading failed, or the request could not be decoded or
    constructed, stop the loop and return. Unexpected exceptions are logged
    by swallow_any and end the session, allowing main() to clean up.
    """
    log.info("Starting the inference-request loop")

    while (request := receive_message(connection)) is not None:
        started = monotonic()

        response = process_request(model, request)
        log.info(f"Got {type(response)=} after {monotonic() - started:.2f} seconds")

        msg_bytes = json.dumps(asdict(response), allow_nan=False).encode("utf-8")

        log.debug(f"Sending {len(msg_bytes)=} bytes")
        connection.send_bytes(msg_bytes)


def parse_args() -> argparse.Namespace:
    """Read the runner's session arguments and Compose's model configuration."""
    parser = argparse.ArgumentParser()

    # The SDK supervisor (nmr-provider-tools) supplies these when it launches worker.py,
    # passing both open descriptors through subprocess.Popen(pass_fds=...)

    # Connected socket for receiving inference requests and sending replies.
    parser.add_argument("--session-fd", required=True, type=int)
    # Descriptor carrying the supervisor's session lock; keep open through process exit.
    parser.add_argument("--lease-fd", required=True, type=int)

    # Unique session identifier generated by the supervisor and included in the startup log.
    parser.add_argument("--operation", required=True)

    parser.add_argument("--ckpt-dir", required=True, type=Path, action="append")  # collects repeated args
    parser.add_argument("--device", required=True)
    parser.add_argument("--recon-workers", type=int)

    return parser.parse_args()


def main() -> None:
    """Prepare checkpoints, load the model, and serve the inherited connection.

    After the request loop ends, leaving load_model() requests pool shutdown
    and leaving Connection closes the inference descriptor. The worker then
    exits, allowing the SDK runner to observe its exit and reap it.
    """

    args = vars(parse_args())

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s pid=%(process)d %(message)s")
    log.setLevel(logging.INFO)

    # ChefNMR2 takes each checkpoint's name, weights path, and configuration path.
    checkpoints = [
        # E.g., /weights/chefnmr2-epoch1399/{model.pt, config.yaml}
        (ckpt_dir.name, ckpt_dir / "model.pt", ckpt_dir / "config.yaml")
        for ckpt_dir in args.pop('ckpt_dir')
    ]

    # The SDK runner (provider_core.runner in nmr-provider-tools) creates a socket pair
    # for inference requests and replies. It passes one end to this worker
    # as --session-fd and transfers the other to the provider adapter as session.science.
    # It also passes the session's lifetime lease as --lease-fd.
    # We keep that descriptor open until process exit, including model cleanup.

    # Check that the inherited lease descriptor is open. Discard the metadata.
    # This raises OSError for an invalid descriptor. The lease stays open because
    # the worker never closes it; fstat adds no lifetime management.
    os.fstat(args.pop('lease_fd'))

    log.info(f"Starting inference session with {args.pop('operation')=}")

    with Connection(args.pop('session_fd')) as connection:
        with load_model(checkpoints, **args) as model:
            serve_connection(connection, model)

    log.info("Inference worker session closed")


if __name__ == "__main__":
    main()
