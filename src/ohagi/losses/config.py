"""User-facing loss specifications."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Loss:
    """Task and consistency-loss configuration.

    ``kind`` may be ``mse`` for surrogate fitting, ``observation`` for
    parameter-estimation data fitting, or ``objective`` for optimization.
    Dual consistency targets are final converged solver multipliers and are
    treated as latent, stop-gradient teacher targets.
    """

    kind: str = "mse"
    primal_consistency_weight: float = 0.0
    equality_dual_consistency_weight: float = 0.0
    inequality_dual_consistency_weight: float = 0.0
    observed_variables: tuple[str, ...] | None = None

    def __post_init__(self):
        selected = str(self.kind).strip().lower()
        if selected not in {"mse", "observation", "objective"}:
            raise ValueError("loss kind must be mse, observation, or objective.")
        for name in (
            "primal_consistency_weight",
            "equality_dual_consistency_weight",
            "inequality_dual_consistency_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        if self.observed_variables is not None and not self.observed_variables:
            raise ValueError("observed_variables cannot be empty.")

    @classmethod
    def mse(
        cls,
        *,
        primal_consistency_weight: float = 0.0,
        equality_dual_consistency_weight: float = 0.0,
        inequality_dual_consistency_weight: float = 0.0,
    ) -> "Loss":
        """Create a mean-squared-error loss specification for supervised outputs."""
        return cls(
            kind="mse",
            primal_consistency_weight=primal_consistency_weight,
            equality_dual_consistency_weight=equality_dual_consistency_weight,
            inequality_dual_consistency_weight=inequality_dual_consistency_weight,
        )

    @classmethod
    def objective(
        cls,
        *,
        primal_consistency_weight: float = 0.0,
        equality_dual_consistency_weight: float = 0.0,
        inequality_dual_consistency_weight: float = 0.0,
    ) -> "Loss":
        """Create a loss specification for minimizing the problem objective."""
        return cls(
            kind="objective",
            primal_consistency_weight=primal_consistency_weight,
            equality_dual_consistency_weight=equality_dual_consistency_weight,
            inequality_dual_consistency_weight=inequality_dual_consistency_weight,
        )

    @classmethod
    def parameter_estimation(
        cls,
        *,
        observed_variables: tuple[str, ...] | list[str] | None = None,
        primal_consistency_weight: float = 0.0,
        equality_dual_consistency_weight: float = 0.0,
        inequality_dual_consistency_weight: float = 0.0,
    ) -> "Loss":
        """Create a loss specification for supervised parameter estimation."""
        return cls(
            kind="observation",
            observed_variables=(
                None if observed_variables is None else tuple(observed_variables)
            ),
            primal_consistency_weight=primal_consistency_weight,
            equality_dual_consistency_weight=equality_dual_consistency_weight,
            inequality_dual_consistency_weight=inequality_dual_consistency_weight,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the loss configuration as a JSON-compatible mapping."""
        payload = asdict(self)
        if self.observed_variables is not None:
            payload["observed_variables"] = list(self.observed_variables)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Loss":
        """Reconstruct a loss specification from its configuration mapping."""
        values = dict(payload)
        if values.get("observed_variables") is not None:
            values["observed_variables"] = tuple(values["observed_variables"])
        return cls(**values)
