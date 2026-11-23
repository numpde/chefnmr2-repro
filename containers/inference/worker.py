from __future__ import annotations

import argparse
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

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


def receive_message(connection: socket.socket) -> dict[str, object] | None:
    """
    Block for a complete request; return None on clean EOF.
    """
    ...


def send_message(connection: socket.socket, message: dict[str, object]) -> None:
    ...


def execute_request(model: ChefNMR2, request: dict[str, object]) -> dict[str, object]:
    ...


def serve_connection(connection: socket.socket, model: ChefNMR2) -> None:
    while (request := receive_message(connection)) is not None:
        response = execute_request(model, request)
        send_message(connection, response)


def serve(listener: socket.socket, **kw) -> None:
    # Start/re-start queuing incoming connections for the bound Unix socket.
    # Non-blocking.
    listener.listen(kw.pop("max_pending_connections"))

    with load_model(**kw) as model:
        while True:
            # Blocks until a client connects.
            # Returns a separate socket for talking to it.
            (connection, _) = listener.accept()

            with connection:
                serve_connection(connection, model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True, type=Path)
    parser.add_argument("--max-pending-connections", type=int, default=socket.SOMAXCONN)
    parser.add_argument("--ckpt-dir", required=True, type=Path, action="append")  # collects repeated args
    parser.add_argument("--device", required=True)
    parser.add_argument("--recon-workers", type=int)
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
            serve(listener, checkpoints=checkpoints, **args)
        finally:
            with suppress(FileNotFoundError):
                if os.path.samestat(bound_entry, socket_path.lstat()):
                    socket_path.unlink()


if __name__ == "__main__":
    main()
