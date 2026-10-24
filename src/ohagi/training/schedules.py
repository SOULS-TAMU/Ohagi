"""Host-side schedules selecting among already compiled executables."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionSchedule:
    """Policy controlling when projection is attempted during training."""

    min_pretrain_epochs: int = 0
    max_pretrain_epochs: int = 100
    eta: float | None = None
    readiness_fraction: float = 1.0
    on_not_ready: str = "continue"

    def __post_init__(self) -> None:
        if self.min_pretrain_epochs < 0:
            raise ValueError("min_pretrain_epochs must be nonnegative.")
        if self.max_pretrain_epochs < self.min_pretrain_epochs:
            raise ValueError("max_pretrain_epochs must not precede the minimum.")
        if not 0.0 <= self.readiness_fraction <= 1.0:
            raise ValueError("readiness_fraction must be in [0, 1].")
        if self.on_not_ready not in {"continue", "raise", "activate"}:
            raise ValueError("invalid on_not_ready policy.")

    def should_attempt(self, epoch: int, metric: float | None = None) -> bool:
        """Determine whether projection is enabled at the given training step."""
        if epoch < self.min_pretrain_epochs:
            return False
        if epoch >= self.max_pretrain_epochs:
            return True
        return self.eta is not None and metric is not None and metric <= self.eta
