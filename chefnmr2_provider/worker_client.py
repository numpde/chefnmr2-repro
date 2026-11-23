"""Exchange typed requests with one SDK-owned ChefNMR2 worker session.

The SDK runner owns child startup and retirement. This synchronous client runs
inside the companion's thread execution backend. Callers finish each request
before starting another or closing the client.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import stat
import tomllib

from multiprocessing.connection import Connection

from provider_core.failure_collection import FailureCollector
from provider_core.runner_protocol import RetirementUnconfirmed
from provider_core.runner_session import RunnerSession

from docker.containers.inference.wire import ElucidationRequest, ElucidationResult, ModelFailure

log = logging.getLogger(__name__)


class WorkerClient:
    """Retain one native session and its framed connection across sequential requests."""

    def __init__(self, session: RunnerSession, *, input_bytes: int, result_bytes: int) -> None:
        self.session = session
        self.input_bytes = input_bytes
        self.result_bytes = result_bytes
        self.connection: Connection | None = None
        self.failure: BaseException | None = None

    def connect(self) -> None:
        """Connect once. The greeting confirms launch, while model loading follows."""
        self.session.connect()
        self.connection = Connection(os.dup(self.session.science.fileno()))
        log.info(f"Opened worker session {self.session.operation}")

    def request(self, request: ElucidationRequest) -> ElucidationResult | ModelFailure:
        """Send once and read one typed reply, retaining failures without replay.

        A ModelFailure is a complete reply and permits another request. A broken
        exchange makes this client unusable until retired. Size limits apply to
        encoded payloads. Blocking I/O has no independent timeout here.
        """
        if self.connection is None or self.failure is not None:
            raise RuntimeError(f"Worker client requires a connected, healthy session; {self.failure = }")

        payload = json.dumps(asdict(request), allow_nan=False).encode("utf-8")
        if len(payload) > self.input_bytes:
            raise ValueError(f"Request is {len(payload)} bytes; maximum is {self.input_bytes}")

        try:
            self.connection.send_bytes(payload)
            response = json.loads(self.connection.recv_bytes(maxlength=self.result_bytes))
            if not isinstance(response, dict):
                raise ValueError(f"Expected a reply object, got {type(response).__name__}")
            if set(response) == {"exception_type", "message"}:
                return ModelFailure(**response)
            return ElucidationResult(**response)
        except BaseException as error:
            self.failure = error
            raise

    def close(self) -> None:
        """Request stop and confirm worker retirement before closing descriptors.

        A failed retirement keeps the session available for another explicit
        close attempt. Caller-side stop is recorded by RunnerSession.retire().
        """
        if self.session.control_connected:
            self.session.retire()
            if not self.session.retirement.scratch_released:
                raise RetirementUnconfirmed(f"Worker scratch remains after retiring {self.session.operation}")
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.session.close()


@contextmanager
def open_worker(config_path: Path) -> Iterator[WorkerClient]:
    """Read deployment settings, open one worker and settle it when the block exits.

    Like NMRPeak's native-session admission, inspect the owner-only lease before
    constructing RunnerSession with its device/inode identity. Preserve request
    and retirement failures independently through the SDK FailureCollector.
    """
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    runner = config["runner"]
    descriptor = os.open(runner["lease_file"], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        lease = os.fstat(descriptor)
        if not (stat.S_ISREG(lease.st_mode) and lease.st_uid == os.geteuid()
                and stat.S_IMODE(lease.st_mode) == 0o600):
            raise PermissionError(f"Lease should be an owner-only regular file owned by UID {os.geteuid()}; {lease = }")
    finally:
        os.close(descriptor)

    session = RunnerSession(
        runner["socket"], role=runner["role"],
        expected_lock_identity=(lease.st_dev, lease.st_ino),
        startup_seconds=runner["startup_seconds"],
        retirement_seconds=runner["retirement_seconds"],
    )
    policy = config["execution"]["operation_policy"]
    client = WorkerClient(session, input_bytes=policy["reject_encoded_input_over_bytes"],
                          result_bytes=policy["reject_encoded_result_over_bytes"])
    failures = FailureCollector(message="Worker session and retirement failed independently")
    with failures.capture():
        client.connect()
        yield client
    with failures.capture():
        client.close()
    failures.raise_if_any()
