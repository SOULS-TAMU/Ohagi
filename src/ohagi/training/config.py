"""Static training configuration and policy normalization."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Literal

GradientPolicy = Literal["strict", "residual_checked", "always"]
InvalidBatchPolicy = Literal["skip", "use_valid"]


def normalize_gradient_policy(value: str) -> GradientPolicy:
    """Validate and canonicalize the projection-gradient policy name."""
    selected = str(value).strip().lower()
    if selected not in {"strict", "residual_checked", "always"}:
        raise ValueError("gradient_policy must be strict, residual_checked, or always.")
    return selected  # type: ignore[return-value]


def normalize_invalid_batch_policy(value: str) -> InvalidBatchPolicy:
    """Validate how nonfinite or ineligible training updates are handled."""
    selected = str(value).strip().lower()
    if selected not in {"skip", "use_valid"}:
        raise ValueError("invalid_batch_policy must be skip or use_valid.")
    return selected  # type: ignore[return-value]


@dataclass(frozen=True)
class DualPretrainConfig:
    """Latent-dual initialization policy across the two training phases.

    During raw pretraining, latent dual values are viewed through
    ``min(-min(0, -z), gamma)``. At projection activation the cap is removed
    and the predicted duals are used immediately; no separate zero-dual
    projection executable is compiled.
    """

    gamma: float = 1e-3
    zero_initialize_dual_weights: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.gamma)) or float(self.gamma) <= 0.0:
            raise ValueError("dual pretraining gamma must be finite and positive.")


@dataclass(frozen=True)
class TrainingConfig:
    """Common host-side training controls."""

    epochs: int = 100
    batch_size: int = 32
    gradient_policy: GradientPolicy = "strict"
    invalid_batch_policy: InvalidBatchPolicy = "skip"
    print_every: int = 1
    dual_pretraining: DualPretrainConfig = field(default_factory=DualPretrainConfig)

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        normalize_gradient_policy(self.gradient_policy)
        normalize_invalid_batch_policy(self.invalid_batch_policy)
        if self.print_every < 0:
            raise ValueError("print_every must be nonnegative.")
