"""JAX-compatible dynamic state and immutable build reports."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple
import jax

Array = jax.Array
PyTree = Any


class AdamState(NamedTuple):
    """Adam step count and first- and second-moment parameter pytrees."""

    count: Array
    first_moment: PyTree
    second_moment: PyTree


class ProjectedMetrics(NamedTuple):
    """Loss and projection diagnostics produced by a projected training step."""

    loss: Array
    task_loss: Array
    raw_mse: Array
    consistency_loss: Array
    equality_dual_consistency: Array
    inequality_dual_consistency: Array
    inverse_regularization: Array
    projection_distance: Array
    forward_converged_fraction: Array
    forward_finite_fraction: Array
    gradient_eligible_fraction: Array
    update_accepted: Array
    max_forward_residual: Array
    mean_forward_iterations: Array


class PredictionOutput(NamedTuple):
    """Structured model outputs and projection diagnostics for prediction."""

    y_hat: Array
    y_tilde: Array
    forward_status_codes: Array
    forward_iterations: Array
    forward_residual_inf_norm: Array


@dataclass(frozen=True)
class BuildReport:
    """Description of the prepared network, problem and projection dimensions."""

    mode: str
    backend: str
    device: str
    batch_size: int
    input_dim: int
    output_dim: int
    physical_parameter_count: int
    projection_prepare_seconds: float
    backbone_initialize_seconds: float
    graph_construct_seconds: float
    raw_train_compile_seconds: float
    raw_eval_compile_seconds: float
    train_compile_seconds: float
    eval_compile_seconds: float
    predict_compile_seconds: float
    validation_seconds: float
    total_build_seconds: float
    persistent_cache_directory: str | None
    validated: bool
    predict_duals: bool = False
    dual_pretrain_gamma: float = 1e-3
    network_output_dim: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the build report as a plain mapping for host-side reporting."""
        return asdict(self)


@dataclass(frozen=True)
class CompiledProjectedGraphs:
    """Compiled callables shared by projected training and inference."""

    raw_train_step: Any
    raw_eval_step: Any
    train_step: Any
    eval_step: Any
    predict_step: Any
    raw_predict_step: Any
