from __future__ import annotations

import argparse
import json
import logging
import os
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict
from functools import wraps
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

if __package__:
    from .wire import ElucidationResult, ElucidationRequest, ModelFailure
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
    """Load the model once, keep it available while the worker serves requests,
    and shut down any reconstruction workers when context exists.
    """
    from chefnmr2.infer import ChefNMR2

    model = ChefNMR2(checkpoints, device=device, recon_workers=recon_workers)

    try:
        yield model
    finally:
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
            model.recon_pool.shutdown()


@swallow_any(Exception)
def receive_message(connection: Connection) -> ElucidationRequest | None:
    """Read one inference request from a connected client and return an ElucidationRequest or None.

    This call blocks until the client sends a complete request or closes its
    sending side. If the client closes between requests, return None. If the
    initial socket check fails, log the error and return None to end the session.

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
                return None
    except OSError as error:
        log.warning(f"Could not check for the next request; {error = }")
        return None

    try:
        request_frame = connection.recv_bytes()
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

    return ElucidationRequest.from_json_or_none(obj)


def process_request(model: ChefNMR2, request: ElucidationRequest) -> ElucidationResult | ModelFailure:
    """Run one elucidation on the loaded model and return the typed result."""

    from chefnmr2.infer import Peaks, ChefNMR2

    assert type(model) == ChefNMR2
    assert type(request) == ElucidationRequest

    try:
        result = model.elucidate(
            Peaks(c=request.peaks.c, h=request.peaks.h),
            formula=request.formula,
            n_samples=request.n_samples,
            seed=request.seed,
        )
    except Exception as error:
        return ModelFailure(exception_type=type(error).__name__, message=str(error))

    return ElucidationResult(**result)


@swallow_any(Exception)
def serve_connection(connection: Connection, model: ChefNMR2) -> None:
    while (request := receive_message(connection)) is not None:
        response = process_request(model, request)

        msg_bytes = json.dumps(asdict(response), allow_nan=False).encode("utf-8")
        connection.send_bytes(msg_bytes)


def serve_to(listener: socket.socket, **kw) -> None:
    # Start/re-start queuing incoming connections for the bound Unix socket.
    # Non-blocking.
    listener.listen(kw.pop("max_pending_connections"))

    with load_model(**kw) as model:
        while True:
            # Blocks until a client connects.
            # Returns a separate socket for talking to it.
            (client_socket, _) = listener.accept()

            # `detach` leaves the descriptor open and relinquishes it (client_socket can no longer close it).
            # `Connection` takes ownership and uses it for recv_bytes() and send_bytes().
            with Connection(client_socket.detach()) as connection:
                serve_connection(connection, model=model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True, type=Path)
    parser.add_argument("--ckpt-dir", required=True, type=Path, action="append")  # collects repeated args
    parser.add_argument("--device", required=True)
    parser.add_argument("--recon-workers", type=int)
    parser.add_argument("--max-pending-connections", type=int, default=socket.SOMAXCONN)
    return parser.parse_args()


def main() -> None:
    args = vars(parse_args())

    checkpoints = [
        # E.g., /weights/chefnmr2-epoch1399/{model.pt, config.yaml}
        (ckpt_dir.name, ckpt_dir / "model.pt", ckpt_dir / "config.yaml")
        for ckpt_dir in args.pop('ckpt_dir')
    ]

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        socket_path: Path = args.pop('socket_path')

        # Creates a Unix socket entry at socket_path and assigns that address to listener
        listener.bind(os.fspath(socket_path))

        # During cleanup below, os.path.samestat() compares the saved device and inode numbers
        # with those currently at socket_path (bound_entry). If they match, unlink the path.
        # The check and unlink are separate operations, so a replacement between them is still possible.
        bound_entry = socket_path.lstat()

        try:
            serve_to(listener, checkpoints=checkpoints, **args)
        finally:
            with suppress(FileNotFoundError):
                if os.path.samestat(bound_entry, socket_path.lstat()):
                    socket_path.unlink()


if __name__ == "__main__":
    main()
