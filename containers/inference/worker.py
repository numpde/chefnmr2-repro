from __future__ import annotations

import argparse
import json
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING

if __package__:
    from .wire import ElucidateRequest, Peaks
else:
    from wire import ElucidateRequest, Peaks

if TYPE_CHECKING:
    from chefnmr2.infer import ChefNMR2


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


MAX_MESSAGE_BYTES = 2 * 1024 * 1024


def receive_message(frames: Connection) -> ElucidateRequest | None:
    """Read one inference request from a connected client and return an ElucidateRequest.

    This call blocks until the client sends a complete request or closes its
    sending side. If the client closes between requests, return None.

    If a byte is available, the initial check leaves it in place so
    frames.recv_bytes() can read the length header and the complete message.

    If the client closes after sending part of the header or part of the
    payload, recv_bytes() raises OSError. If it sends the complete header
    for a nonempty payload and closes before sending any payload bytes,
    recv_bytes() raises EOFError.

    Example JSON payload:

        {
            "formula": "CH4",
            "peaks": {
                "h": [[1.0, 4, "s"]]
            }
        }

    For this payload, receive_message() returns:

        ElucidateRequest(
            peaks=Peaks(c=[], h=[(1.0, 4, "s")]),
            formula="CH4",
            n_samples=100,
            seed=0,
        )
    """

    # Leave the first header byte for recv_bytes. Once it exists, any EOF is truncation.
    with socket.fromfd(frames.fileno(), socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        if not sock.recv(1, socket.MSG_PEEK):
            return None

    payload = frames.recv_bytes()
    message = json.loads(payload.decode("utf-8"))

    if not isinstance(message, dict):
        raise ValueError(f"Expected a JSON object, received {type(message).__name__}")

    peaks = Peaks(**message.pop("peaks"))
    peaks.h = [(ppm, integration, multiplicity) for (ppm, integration, multiplicity) in peaks.h]
    return ElucidateRequest(peaks=peaks, **message)


def send_message(connection: socket.socket, message: dict[str, object]) -> None:
    ...


def execute_request(model: ChefNMR2, request: ElucidateRequest) -> dict[str, object]:
    ...


def serve_connection(connection: socket.socket, model: ChefNMR2) -> None:
    # Read-only multiprocessing.connection.Connection named `frames` for the same connected Unix socket
    with Connection(os.dup(connection.fileno()), writable=False) as frames:
        while (request := receive_message(frames)) is not None:
            response = execute_request(model, request)
            send_message(connection, response)


def serve_to(listener: socket.socket, **kw) -> None:
    # Start/re-start queuing incoming connections for the bound Unix socket.
    # Non-blocking.
    listener.listen(kw.pop("max_pending_connections"))

    with load_model(**kw) as model:
        while True:
            # Blocks until a client connects.
            # Returns a separate socket for talking to it.
            (connection, _) = listener.accept()

            with connection:
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
