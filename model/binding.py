"""Declare the inference operation in both containers; load science only in the worker."""

import logging
import pathlib
from typing import TYPE_CHECKING

import provider_core.scientific as sdk

if __package__:
    from .wire import ElucidationBatch, ElucidationBatchResult, ElucidationResult, ModelFailure
else:
    from wire import ElucidationBatch, ElucidationBatchResult, ElucidationResult, ModelFailure

if TYPE_CHECKING:
    from chefnmr2.infer import ChefNMR2

log = logging.getLogger(__name__)


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
        from chefnmr2.infer import ChefNMR2

        checkpoints = [
            (directory.name, directory / "model.pt", directory / "config.yaml")
            for directory in map(pathlib.Path, self.checkpoint_dirs)
        ]

        log.info(f"Loading {len(checkpoints)=} on {self.device=} with {self.recon_workers=}")

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
                log.exception(f"Batch entry failed; {index=}")
                elucidation_reply = ModelFailure(exception_type=type(error).__name__, message=str(error))
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
