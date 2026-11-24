"""Declare the inference operation in both containers; load science only in the worker."""

import hashlib
import json
import logging
import os
import pathlib
from contextlib import contextmanager, ExitStack
from glob import glob
from itertools import chain
from typing import TYPE_CHECKING

import provider_core.scientific as sdk
from provider_core.attempt_record import MAX_EVIDENCE_BYTES
from provider_core.diagnostics import describe_core_failure, failure_evidence_json

if __package__:
    from .wire import ElucidationBatch, ElucidationBatchResult, ElucidationResult, ModelFailure
else:
    from wire import ElucidationBatch, ElucidationBatchResult, ElucidationResult, ModelFailure

if TYPE_CHECKING:
    from chefnmr2.infer import ChefNMR2

log = logging.getLogger(__name__)


@contextmanager
def verified_weights(path, companions=()):
    """Verify selected checkpoint inputs and retain those exact files for loading."""
    path = pathlib.Path(path)
    paths = [path, *(pathlib.Path(companion) for companion in companions)]
    marker = path.with_name(path.name + ".sanitized.json")

    with ExitStack() as opened:
        try:
            receipt_stream = opened.enter_context(marker.open())
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Cannot load {path}: its sanitization receipt is missing; run make weights/sanitize."
            ) from error

        receipt = json.load(receipt_stream)

        if (
            (receipt["schema"] != 1) or
            (receipt["policy"] != "torch-weights-only-v1") or
            not {item.name for item in paths}.issubset(receipt["files"])
        ):
            raise ValueError(f"Cannot load {path}: its sanitization receipt does not cover the selected files and policy")

        streams = {item: opened.enter_context(item.open("rb")) for item in paths}

        for (item, stream) in streams.items():
            digest = hashlib.file_digest(stream, "sha256").hexdigest()

            if digest != receipt["files"][item.name]:
                raise ValueError(f"Cannot load {item}: SHA-256 {digest} differs from its sanitization receipt")

            stream.seek(0)

        # Managed writers invalidate the receipt before replacing covered files.
        # Recheck after opening and hashing; the retained descriptors then stay fixed.
        if not os.path.samestat(os.fstat(receipt_stream.fileno()), marker.stat()):
            raise ValueError(f"Cannot load {path}: sanitization changed while opening the checkpoint")

        yield streams


class Inference(sdk.Contribution):
    def __init__(self, *, checkpoint_dirs: list[str], device: str, recon_workers: int) -> None:
        super().__init__()

        if recon_workers < 1:
            raise ValueError(f"Native inference requires an owned reconstruction pool; {recon_workers=}")

        self.checkpoint_dirs = checkpoint_dirs
        self.device = device
        self.recon_workers = recon_workers
        self.model: ChefNMR2 | None = None

    def initialize(self, *, assets: sdk.AssetAccess) -> None:
        """Load the checkpoints before the SDK acknowledges model readiness."""

        with ExitStack() as opened:
            checkpoints = []

            paths = sorted(set(chain.from_iterable(map(glob, self.checkpoint_dirs))))
            for directory in map(pathlib.Path, paths):
                (weights, config) = (directory / "model.pt", directory / "config.yaml")
                streams = opened.enter_context(verified_weights(weights, companions=(config,)))

                checkpoints.append((directory.name, streams[weights], streams[config]))

            log.info(f"Loading {len(checkpoints)=} on {self.device=} with {self.recon_workers=}")
            from chefnmr2.infer import ChefNMR2

            self.model = ChefNMR2(checkpoints, device=self.device, recon_workers=self.recon_workers)

        log.info("Model ready")

    @sdk.operation()
    def elucidate_batch(self, *, batch: ElucidationBatch) -> ElucidationBatchResult:
        """Run entries sequentially and retain each completed result or model exception."""
        from chefnmr2.infer import Peaks

        replies = []

        for (index, elucidation) in enumerate(batch.requests):
            log.info(f"Starting batch entry {index=} with {elucidation.n_samples=} and {elucidation.seed=}")

            try:
                elucidation_result = self.model.elucidate(
                    Peaks(c=elucidation.peaks.c, h=elucidation.peaks.h),
                    formula=elucidation.formula,
                    n_samples=elucidation.n_samples,
                    seed=elucidation.seed,
                )

                elucidation_reply = ElucidationResult(**elucidation_result)
            except Exception as error:
                # todo: Review exception text for partial scientific-input echoes;
                # credentials and the formula are redacted, and locals are never captured.
                diagnostic = json.loads(
                    failure_evidence_json(
                        {"failure": error},
                        secrets=(
                            elucidation.formula,
                            str(elucidation.peaks),
                            str(elucidation.peaks.c),
                            str(elucidation.peaks.h),
                            json.dumps(elucidation.peaks.c),
                            json.dumps(elucidation.peaks.h),
                        ),
                        maximum_bytes=MAX_EVIDENCE_BYTES // 2,
                        describe_owner=lambda failure,
                        text,
                        window: {
                            **describe_core_failure(failure, text, window),
                            "message": str(failure),
                        },
                    ),
                )["failure"]

                log.error(f"Batch entry failed; {index=}; diagnostic={json.dumps(diagnostic)}")

                elucidation_reply = ModelFailure(
                    exception_type=diagnostic["exception_type"],
                    message=diagnostic["message"],
                    diagnostic=diagnostic,
                )
            else:
                log.info(f"Batch entry completed: {index=}, {elucidation_reply.n_valid=}, {elucidation_reply.n_samples=}")

            replies.append(elucidation_reply)

        return ElucidationBatchResult(replies=replies)

    def close(self) -> None:
        """Join reconstruction workers before releasing the initialized model."""

        if self.model is not None:
            if self.model.recon_pool is not None:
                log.info("Joining reconstruction workers")
                self.model.recon_pool.shutdown()

            self.model = None

            log.info("Model closed")


def construct(*, startup: dict) -> Inference:
    return Inference(**startup)
