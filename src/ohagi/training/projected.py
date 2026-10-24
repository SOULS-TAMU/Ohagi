"""End-to-end projected neural training and reporting."""

from __future__ import annotations
import contextlib
import io
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from ohagi.nn.outputs import split_pretraining_outputs as split_pretraining_outputs
from ohagi.nn.outputs import split_projection_outputs as split_projection_outputs
from ohagi.nn.outputs import split_raw_heads as split_raw_heads
from purin.solvers.implicit import ImplicitProjectionLayer
from purin.solvers.implicit import make_implicit_projection_layer
from purin.solvers.primal_dual_interior_point.solver import BATCH_STATUS_CONVERGED
from purin.solvers.primal_dual_interior_point.solver import (
    PreparedSparseProjectionBatchSolver,
)
from purin.execution.runtime import configure_cache as configure_compilation_cache
from ohagi.training.config import (
    normalize_gradient_policy as _normalize_gradient_policy,
)
from ohagi.training.config import (
    normalize_invalid_batch_policy as _normalize_invalid_batch_policy,
)
from ohagi.training.logging import KKTTableLogger
from ohagi.training.logging import print_postprocessing_finish
from ohagi.training.logging import print_postprocessing_start
from ohagi.training.logging import print_research_citation
from ohagi.training.state import BuildReport
from ohagi.training.state import CompiledProjectedGraphs
from ohagi.training.state import PredictionOutput
from ohagi.training.state import ProjectedMetrics
from ohagi.training.steps import Adam
from ohagi.training.steps import apply_updates
from ohagi.training.steps import optimizer_step as optimizer_step
from ohagi.models import ProjectedModel
from ohagi.nn.backbone import FunctionalBackbone
from ohagi.training.runner import train_epoch as execute_train_epoch  # noqa: F401 - supported compatibility export
from ohagi.training.runner import parameter_step
from ohagi.training.engine import Engine, EpochPlan
from ohagi.training.runner import EpochTransition
from purin.projections.adapters import ImplicitDistance
from ohagi.training.steps import copy_tree_to_device as _copy_tree_to_device
from ohagi.training.steps import masked_mean as _masked_mean
from ohagi.training.steps import per_sample as _per_sample

Array = jax.Array
PyTree = Any
BackboneInit = Callable[[Array, int, int, Any], PyTree]
BackboneApply = Callable[[PyTree, Array], Array]
ProjectionFactory = Callable[[int], PreparedSparseProjectionBatchSolver]


class ProjectedTrainerCore:
    """Projected neural model with build-time AOT compilation.

    ``projection`` may be a prepared fixed-batch solver or a factory receiving
    ``batch_size``.  A factory is preferred because compilation-cache setup can
    occur before the projection solver is prepared.
    """

    def __init__(
        self,
        *,
        projection: PreparedSparseProjectionBatchSolver | ProjectionFactory,
        backbone_init: BackboneInit,
        backbone_apply: BackboneApply,
        input_dim: int,
        backbone_input_dim: int | None = None,
        mode: str = "supervised",
        optimizer: Any | None = None,
        data_loss_fn: Callable[[Array, Array, Array], Array] | None = None,
        objective_fn: Callable[[Array, Array], Array] | None = None,
        problem_parameter_fn: Callable[[Array, Array | None], Array] | None = None,
        backbone_input_fn: Callable[[Array, Array | None], Array] | None = None,
        inverse_init: Any | None = None,
        inverse_regularizer: Callable[[Array], Array] | None = None,
        consistency_weight: float = 0.0,
        equality_dual_consistency_weight: float = 0.0,
        inequality_dual_consistency_weight: float = 0.0,
        predict_duals: bool = False,
        dual_pretrain_gamma: float = 1e-3,
        inequality_dual_floor: float = 1e-8,
        primal_lower_bounds: Any | None = None,
        primal_upper_bounds: Any | None = None,
        primal_bounds_fn: Callable[[Array], tuple[Array, Array]] | None = None,
        mu: float = 1e-4,
        optimizer_apply_updates: Callable[[PyTree, PyTree], PyTree] = apply_updates,
        gradient_policy: str = "residual_checked",
        inexact_gradient_tolerance: float = 1e-6,
        adjoint_gradient_tolerance: float = 1e-8,
        invalid_batch_policy: str = "skip",
        invalidate_failed_gradients: bool | None = None,
    ) -> None:
        """Store model, projection, loss and optimizer resources for shared training."""
        selected_mode = str(mode).strip().lower()
        if selected_mode not in {"supervised", "inverse", "optimization"}:
            raise ValueError("mode must be 'supervised', 'inverse', or 'optimization'.")
        if selected_mode == "optimization" and objective_fn is None:
            raise ValueError("optimization mode requires objective_fn.")
        if selected_mode == "inverse" and inverse_init is None:
            raise ValueError("inverse mode requires inverse_init.")
        if int(input_dim) < 1:
            raise ValueError("input_dim must be positive.")
        if float(consistency_weight) < 0.0:
            raise ValueError("consistency_weight must be nonnegative.")
        self.projection_definition = projection
        self.backbone_init = backbone_init
        self.backbone_apply = backbone_apply
        self.input_dim = int(input_dim)
        self.backbone_input_dim = (
            None if backbone_input_dim is None else int(backbone_input_dim)
        )
        if self.backbone_input_dim is not None and self.backbone_input_dim < 1:
            raise ValueError("backbone_input_dim must be positive when supplied.")
        self.mode = selected_mode
        self.optimizer = optimizer or Adam()
        self.optimizer_apply_updates = optimizer_apply_updates
        self.data_loss_fn = data_loss_fn
        self.objective_fn = objective_fn
        self.problem_parameter_fn = problem_parameter_fn
        self.backbone_input_fn = backbone_input_fn
        self.inverse_init = (
            None
            if inverse_init is None
            else np.asarray(inverse_init, dtype=np.float64).reshape(-1)
        )
        self.inverse_regularizer = inverse_regularizer
        self.consistency_weight = float(consistency_weight)
        self.equality_dual_consistency_weight = float(equality_dual_consistency_weight)
        self.inequality_dual_consistency_weight = float(
            inequality_dual_consistency_weight
        )
        self.predict_duals = bool(predict_duals)
        self.dual_pretrain_gamma = float(dual_pretrain_gamma)
        self.inequality_dual_floor = float(inequality_dual_floor)
        self.primal_lower_bounds = (
            None
            if primal_lower_bounds is None
            else np.asarray(primal_lower_bounds, dtype=np.float64).reshape(-1)
        )
        self.primal_upper_bounds = (
            None
            if primal_upper_bounds is None
            else np.asarray(primal_upper_bounds, dtype=np.float64).reshape(-1)
        )
        self.primal_bounds_fn = primal_bounds_fn
        for name, value in (
            ("equality_dual_consistency_weight", self.equality_dual_consistency_weight),
            (
                "inequality_dual_consistency_weight",
                self.inequality_dual_consistency_weight,
            ),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not np.isfinite(self.dual_pretrain_gamma) or self.dual_pretrain_gamma <= 0.0:
            raise ValueError("dual_pretrain_gamma must be finite and positive.")
        if (
            not np.isfinite(self.inequality_dual_floor)
            or self.inequality_dual_floor < 0.0
        ):
            raise ValueError("inequality_dual_floor must be finite and nonnegative.")
        if (
            self.predict_duals
            and self.dual_pretrain_gamma <= self.inequality_dual_floor
        ):
            raise ValueError(
                "dual_pretrain_gamma must exceed inequality_dual_floor when "
                "dual prediction is enabled."
            )
        self.mu = float(mu)
        self.gradient_policy = _normalize_gradient_policy(gradient_policy)
        self.invalid_batch_policy = _normalize_invalid_batch_policy(
            invalid_batch_policy
        )
        self.inexact_gradient_tolerance = float(inexact_gradient_tolerance)
        self.adjoint_gradient_tolerance = float(adjoint_gradient_tolerance)
        if (
            not np.isfinite(self.inexact_gradient_tolerance)
            or self.inexact_gradient_tolerance < 0.0
        ):
            raise ValueError(
                "inexact_gradient_tolerance must be finite and nonnegative."
            )
        if (
            not np.isfinite(self.adjoint_gradient_tolerance)
            or self.adjoint_gradient_tolerance < 0.0
        ):
            raise ValueError(
                "adjoint_gradient_tolerance must be finite and nonnegative."
            )
        # Deprecated compatibility argument. Failed gradients are never filled
        # with NaN; they are sanitized and controlled by gradient_policy.
        del invalidate_failed_gradients

        self.prepared_projection: PreparedSparseProjectionBatchSolver | None = None
        self.projection_layer: ImplicitProjectionLayer | None = None
        self.parameters: PyTree | None = None
        self.optimizer_state: PyTree | None = None
        self.graphs: CompiledProjectedGraphs | None = None
        self.build_report: BuildReport | None = None
        self.batch_size: int | None = None
        self.dtype = jnp.float64

    @property
    def is_built(self) -> bool:
        """Report whether training and prediction graphs have been prepared."""
        return self.graphs is not None

    def _resolve_projection(self, batch_size: int):
        definition = self.projection_definition
        if hasattr(definition, "solve_with_linearization_jax"):
            prepared = definition
        else:
            prepared = definition(int(batch_size))
        if prepared.batch_size != int(batch_size):
            raise ValueError(
                f"Prepared projection batch size {prepared.batch_size} does not "
                f"match requested model batch size {batch_size}."
            )
        return prepared

    def _default_problem_parameters(
        self, x_batch: Array, inverse_parameters: Array | None, physical_count: int
    ) -> Array:
        if inverse_parameters is None:
            values = x_batch
        else:
            inverse_batch = jnp.broadcast_to(
                inverse_parameters, (x_batch.shape[0], inverse_parameters.shape[0])
            )
            values = jnp.concatenate((x_batch, inverse_batch), axis=1)
        if values.shape[1] != physical_count:
            raise ValueError(
                "Default problem-parameter packing produced "
                f"{values.shape[1]} columns; the projection expects "
                f"{physical_count}. Supply problem_parameter_fn to customize "
                "the mapping."
            )
        return values

    def _default_backbone_inputs(
        self, x_batch: Array, inverse_parameters: Array | None
    ) -> Array:
        if inverse_parameters is None:
            return x_batch
        inverse_batch = jnp.broadcast_to(
            inverse_parameters, (x_batch.shape[0], inverse_parameters.shape[0])
        )
        return jnp.concatenate((x_batch, inverse_batch), axis=1)

    def build(
        self,
        *,
        batch_size: int,
        seed: int = 0,
        cache_dir: str | Path | None = None,
        validate: bool = True,
        compile_raw_predict: bool = True,
        example_x: Any | None = None,
        example_y: Any | None = None,
    ) -> BuildReport:
        """Prepare the solver and AOT-compile train/eval/predict entry points."""

        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive.")
        if self.is_built:
            raise RuntimeError("This model is already built.")
        total_start = time.perf_counter()
        configure_compilation_cache(cache_dir)

        projection_start = time.perf_counter()
        prepared = self._resolve_projection(int(batch_size))
        projection_seconds = time.perf_counter() - projection_start
        self.prepared_projection = prepared
        self.batch_size = int(batch_size)
        physical_count = int(
            prepared.parameter_count - len(prepared.compiled.variable_names)
        )
        output_dim = int(len(prepared.compiled.variable_names))
        equality_count = int(prepared.compiled.initial_state.y_c.shape[1])
        inequality_count = int(prepared.compiled.initial_state.y_d.shape[1])

        static_primal_lower, static_primal_upper = _prepare_projected_bounds(
            self, output_dim
        )

        network_output_dim = (
            output_dim + equality_count + inequality_count
            if self.predict_duals
            else output_dim
        )
        differentiate_problem_parameters = self.mode == "inverse"
        projection_layer = make_implicit_projection_layer(
            prepared,
            mu=self.mu,
            differentiate_problem_parameters=differentiate_problem_parameters,
            gradient_policy=self.gradient_policy,
            inexact_gradient_tolerance=self.inexact_gradient_tolerance,
            adjoint_gradient_tolerance=self.adjoint_gradient_tolerance,
        )
        self.projection_layer = projection_layer

        backbone_initialize_seconds, optimizer_state, trainable_parameters = (
            _initialize_projected_backbone(
                self,
                equality_count,
                inequality_count,
                network_output_dim,
                output_dim,
                seed,
            )
        )
        graph_construct_start = time.perf_counter()

        (
            batch_size_static,
            consistency_weight,
            data_loss_fn,
            dual_pretrain_gamma,
            equality_dual_consistency_weight,
            inequality_dual_consistency_weight,
            invalid_batch_policy,
            inverse_regularizer,
            mode,
            model,
            objective_fn,
            optimizer,
            optimizer_apply,
            predict_duals,
            raw_model,
            split_trainable,
        ) = _compose_projected_models(
            self,
            batch_size,
            equality_count,
            inequality_count,
            output_dim,
            physical_count,
            projection_layer,
            static_primal_lower,
            static_primal_upper,
        )

        forward_outputs, metrics_from_losses, raw_metrics_from_losses = (
            _prepare_projected_metrics(
                consistency_weight,
                equality_count,
                equality_dual_consistency_weight,
                inequality_count,
                inequality_dual_consistency_weight,
                invalid_batch_policy,
                inverse_regularizer,
                model,
                predict_duals,
            )
        )

        (
            eval_step,
            graph_construct_seconds,
            predict_step,
            raw_eval_step,
            raw_predict_step,
            raw_train_step,
            train_step,
        ) = _construct_projected_steps(
            self,
            batch_size_static,
            data_loss_fn,
            forward_outputs,
            graph_construct_start,
            invalid_batch_policy,
            metrics_from_losses,
            mode,
            objective_fn,
            optimizer,
            optimizer_apply,
            raw_metrics_from_losses,
            raw_model,
            split_trainable,
        )

        device = prepared.device
        parameters_device = _copy_tree_to_device(trainable_parameters, device)
        optimizer_state_device = _copy_tree_to_device(optimizer_state, device)

        (
            dummy_mask,
            dummy_x,
            dummy_y,
            raw_eval_compile_seconds,
            raw_eval_executable,
            raw_train_compile_seconds,
            raw_train_executable,
        ) = _prepare_projected_examples(
            self,
            batch_size_static,
            device,
            example_x,
            example_y,
            mode,
            optimizer_state_device,
            output_dim,
            parameters_device,
            raw_eval_step,
            raw_train_step,
        )

        (
            eval_compile_seconds,
            eval_executable,
            predict_compile_seconds,
            predict_executable,
            raw_predict_executable,
            train_compile_seconds,
            train_executable,
            validated,
            validation_seconds,
        ) = _compile_projected_graphs(
            compile_raw_predict,
            device,
            dummy_mask,
            dummy_x,
            dummy_y,
            eval_step,
            mode,
            optimizer_state_device,
            parameters_device,
            predict_step,
            raw_predict_step,
            train_step,
            validate,
        )

        self.parameters = parameters_device
        self.optimizer_state = optimizer_state_device
        self.graphs = CompiledProjectedGraphs(
            raw_train_step=raw_train_executable,
            raw_eval_step=raw_eval_executable,
            train_step=train_executable,
            eval_step=eval_executable,
            predict_step=predict_executable,
            raw_predict_step=raw_predict_executable,
        )
        report = BuildReport(
            mode=mode,
            backend=prepared.backend,
            device=prepared.device.device_kind,
            batch_size=batch_size_static,
            input_dim=self.input_dim,
            output_dim=output_dim,
            physical_parameter_count=physical_count,
            projection_prepare_seconds=projection_seconds,
            backbone_initialize_seconds=backbone_initialize_seconds,
            graph_construct_seconds=max(0.0, graph_construct_seconds),
            raw_train_compile_seconds=raw_train_compile_seconds,
            raw_eval_compile_seconds=raw_eval_compile_seconds,
            train_compile_seconds=train_compile_seconds,
            eval_compile_seconds=eval_compile_seconds,
            predict_compile_seconds=predict_compile_seconds,
            validation_seconds=validation_seconds,
            total_build_seconds=time.perf_counter() - total_start,
            persistent_cache_directory=(
                None
                if cache_dir is None
                else str(Path(cache_dir).expanduser().resolve())
            ),
            validated=validated,
            predict_duals=predict_duals,
            dual_pretrain_gamma=dual_pretrain_gamma,
            network_output_dim=network_output_dim,
        )
        self.build_report = report
        return report

    def _require_built(self):
        if not self.is_built:
            raise RuntimeError("Call model.build(...) before fit/evaluate/predict.")
        assert self.graphs is not None
        assert self.parameters is not None
        assert self.optimizer_state is not None
        assert self.prepared_projection is not None
        assert self.batch_size is not None

    def _fixed_batches(
        self, x: np.ndarray, y: np.ndarray | None, *, indices: np.ndarray
    ):
        assert self.batch_size is not None
        if not len(x):
            return
        from ohagi.data import fixed_batches

        data = {"inputs": x}
        if y is not None:
            data["targets"] = y
        for batch in fixed_batches(data, self.batch_size, indices=indices):
            yield (
                batch["inputs"],
                batch.get("targets"),
                batch["mask"],
                int(batch["mask"].sum()),
            )

    @staticmethod
    def _aggregate_metric_batches(
        metrics_with_counts: list[tuple[ProjectedMetrics, int]],
    ) -> dict[str, float]:
        if not metrics_with_counts:
            raise RuntimeError("No metric batches were produced.")
        host_metrics = jax.device_get(
            tuple(metrics for metrics, _count in metrics_with_counts)
        )
        counts = np.asarray(
            [count for _metrics, count in metrics_with_counts], dtype=np.float64
        )
        total = float(np.sum(counts))
        if total <= 0.0:
            raise RuntimeError("Metric aggregation received no valid samples.")

        def weighted(name: str) -> float:
            values = np.asarray(
                [float(getattr(item, name)) for item in host_metrics], dtype=np.float64
            )
            return float(np.sum(counts * values) / total)

        return {
            "loss": weighted("loss"),
            "task_loss": weighted("task_loss"),
            "raw_mse": weighted("raw_mse"),
            "consistency_loss": weighted("consistency_loss"),
            "inverse_regularization": weighted("inverse_regularization"),
            "projection_distance": weighted("projection_distance"),
            "forward_converged_fraction": weighted("forward_converged_fraction"),
            "forward_finite_fraction": weighted("forward_finite_fraction"),
            "gradient_eligible_fraction": weighted("gradient_eligible_fraction"),
            "update_accepted_fraction": weighted("update_accepted"),
            "skipped_update_fraction": 1.0 - weighted("update_accepted"),
            "max_forward_residual": float(
                np.max([float(item.max_forward_residual) for item in host_metrics])
            ),
            "mean_forward_iterations": weighted("mean_forward_iterations"),
        }

    def _evaluate_compiled_graph(
        self, graph: Any, x_host: np.ndarray, y_host: np.ndarray | None
    ) -> dict[str, float]:
        assert self.prepared_projection is not None
        indices = np.arange(x_host.shape[0])
        device = self.prepared_projection.device
        outputs: list[tuple[ProjectedMetrics, int]] = []
        for x_batch, y_batch, valid_mask, count in self._fixed_batches(
            x_host, y_host, indices=indices
        ):
            x_device = jax.device_put(jnp.asarray(x_batch), device)
            mask_device = jax.device_put(jnp.asarray(valid_mask), device)
            if self.mode in {"supervised", "inverse"}:
                if y_batch is None:
                    raise ValueError(f"{self.mode} evaluation requires y data.")
                metrics = graph(
                    self.parameters,
                    x_device,
                    jax.device_put(jnp.asarray(y_batch), device),
                    mask_device,
                )
            else:
                metrics = graph(self.parameters, x_device, mask_device)
            outputs.append((metrics, count))
        return self._aggregate_metric_batches(outputs)

    def fit(
        self,
        x: Any,
        y: Any | None = None,
        *,
        epochs: int,
        shuffle: bool = True,
        seed: int = 0,
        print_every: int = 0,
        pretrain_epochs: int = 0,
        eta: float | None = None,
        switch_metric: str = "raw_mse",
    ) -> list[dict[str, Any]]:
        """Train using only executables produced by :meth:`build`.

        Supervised and inverse models can begin with the compiled raw-backbone
        graph. Projection is activated for the following epoch when either the
        epoch-end raw metric reaches ``eta`` or ``pretrain_epochs`` is reached.
        The same parameter and optimizer-state buffers continue across phases.
        """

        self._require_built()
        assert self.graphs is not None
        assert self.prepared_projection is not None
        pretraining_requested, x_host, y_host = _validate_projected_fit(
            self, epochs, eta, pretrain_epochs, switch_metric, x, y
        )

        rng = np.random.default_rng(int(seed))
        history: list[dict[str, Any]] = []
        device = self.prepared_projection.device
        indices = np.arange(x_host.shape[0])
        projection_active = not pretraining_requested
        projection_start_epoch: int | None = 1 if projection_active else None
        switch_reason: str | None = None

        def epoch_policy(epoch, state):
            nonlocal projection_active, projection_start_epoch, switch_reason
            epoch_projection_active = projection_active
            if shuffle:
                rng.shuffle(indices)
            epoch_start = time.perf_counter()

            def prepare_batch(batch):
                x_batch, y_batch, valid_mask, count = batch
                x_device = jax.device_put(jnp.asarray(x_batch), device)
                mask_device = jax.device_put(jnp.asarray(valid_mask), device)
                if self.mode in {"supervised", "inverse"}:
                    assert y_batch is not None
                    y_device = jax.device_put(jnp.asarray(y_batch), device)
                    return (x_device, y_device, mask_device), count
                return (x_device, mask_device), count

            selected_step = (
                self.graphs.train_step
                if epoch_projection_active
                else self.graphs.raw_train_step
            )
            assert selected_step is not None
            result = yield EpochPlan(
                (self.parameters, self.optimizer_state),
                self._fixed_batches(x_host, y_host, indices=indices),
                parameter_step(selected_step),
                prepare=prepare_batch,
            )
            self.parameters, self.optimizer_state = result.state
            epoch_metrics = list(zip(result.metrics, result.counts))

            row = self._aggregate_metric_batches(epoch_metrics)
            if not epoch_projection_active:
                assert self.graphs.raw_eval_step is not None
                row = self._evaluate_compiled_graph(
                    self.graphs.raw_eval_step, x_host, y_host
                )
            row.update(
                {
                    "epoch": epoch,
                    "phase": "projected" if epoch_projection_active else "pretrain",
                    "projection_active": int(epoch_projection_active),
                    "epoch_seconds": float(time.perf_counter() - epoch_start),
                    "projection_start_epoch": (
                        -1 if projection_start_epoch is None else projection_start_epoch
                    ),
                    "switch_reason": "" if switch_reason is None else switch_reason,
                }
            )
            history.append(row)

            should_print = print_every and (
                epoch == 1 or epoch == int(epochs) or epoch % int(print_every) == 0
            )
            if should_print:
                print(
                    "projected_fit "
                    f"epoch={epoch} phase={row['phase']} "
                    f"loss={row['loss']:.6e} raw_mse={row['raw_mse']:.6e} "
                    f"projection_distance={row['projection_distance']:.3e} "
                    f"converged={row['forward_converged_fraction']:.3f} "
                    f"eligible={row['gradient_eligible_fraction']:.3f} "
                    f"updates={row['update_accepted_fraction']:.3f}"
                )

            if not epoch_projection_active:
                reached_eta = eta is not None and row[switch_metric] <= float(eta)
                reached_epoch_limit = int(pretrain_epochs) > 0 and epoch >= int(
                    pretrain_epochs
                )
                if reached_eta or reached_epoch_limit:
                    projection_active = True
                    if epoch < int(epochs):
                        projection_start_epoch = epoch + 1
                    reasons = []
                    if reached_eta:
                        reasons.append("eta")
                    if reached_epoch_limit:
                        reasons.append("pretrain_epochs")
                    switch_reason = "+".join(reasons)
                    if projection_start_epoch is not None:
                        print(
                            "Activating projection at "
                            f"epoch {projection_start_epoch} "
                            f"(trigger: {switch_reason})."
                        )
            return EpochTransition(state)

        Engine.run(None, int(epochs), epoch_policy)
        return history

    def evaluate(
        self, x: Any, y: Any | None = None, *, use_projection: bool = True
    ) -> dict[str, float]:
        """Evaluate the prepared model and aggregate its configured metrics."""
        self._require_built()
        assert self.graphs is not None
        assert self.prepared_projection is not None
        x_host = np.asarray(x, dtype=np.float64)
        if x_host.ndim != 2 or x_host.shape[1] != self.input_dim:
            raise ValueError(
                f"x must have shape (N, {self.input_dim}); got {x_host.shape}."
            )
        y_host = None if y is None else np.asarray(y, dtype=np.float64)
        if use_projection:
            graph = self.graphs.eval_step
        else:
            if self.graphs.raw_eval_step is None:
                raise RuntimeError("Raw evaluation is unavailable for this model mode.")
            graph = self.graphs.raw_eval_step
        return self._evaluate_compiled_graph(graph, x_host, y_host)

    def evaluate_raw(self, x: Any, y: Any) -> dict[str, float]:
        """Evaluate the backbone without invoking the projection solver."""

        return self.evaluate(x, y, use_projection=False)

    def predict(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return projected predictions from the prepared model."""
        self._require_built()
        assert self.graphs is not None
        assert self.prepared_projection is not None
        x_host = np.asarray(x, dtype=np.float64)
        squeeze = x_host.ndim == 1
        if squeeze:
            x_host = x_host.reshape(1, -1)
        if x_host.ndim != 2 or x_host.shape[1] != self.input_dim:
            raise ValueError(
                f"x must have shape (N, {self.input_dim}); got {x_host.shape}."
            )
        indices = np.arange(x_host.shape[0])
        device = self.prepared_projection.device
        hats = []
        projected = []
        for x_batch, _y_batch, _valid_mask, count in self._fixed_batches(
            x_host, None, indices=indices
        ):
            result = self.graphs.predict_step(
                self.parameters, jax.device_put(jnp.asarray(x_batch), device)
            )
            result_host = jax.device_get(result)
            hats.append(np.asarray(result_host.y_hat[:count]))
            projected.append(np.asarray(result_host.y_tilde[:count]))
        y_hat = np.concatenate(hats, axis=0)
        y_tilde = np.concatenate(projected, axis=0)
        if squeeze:
            return y_hat[0], y_tilde[0]
        return y_hat, y_tilde

    def predict_raw(self, x: Any) -> np.ndarray:
        """Return network predictions before projection."""
        self._require_built()
        assert self.graphs is not None
        if self.graphs.raw_predict_step is None:
            raise RuntimeError("Raw prediction was not compiled during build().")
        x_host = np.asarray(x, dtype=np.float64)
        squeeze = x_host.ndim == 1
        if squeeze:
            x_host = x_host.reshape(1, -1)
        indices = np.arange(x_host.shape[0])
        assert self.prepared_projection is not None
        device = self.prepared_projection.device
        outputs = []
        for x_batch, _y_batch, _mask, count in self._fixed_batches(
            x_host, None, indices=indices
        ):
            result = self.graphs.raw_predict_step(
                self.parameters, jax.device_put(jnp.asarray(x_batch), device)
            )
            outputs.append(np.asarray(jax.device_get(result[:count])))
        merged = np.concatenate(outputs, axis=0)
        return merged[0] if squeeze else merged


"""Public projected-neural model façade with additive logging/serialization."""

from typing import Any


_EPOCH_RE = re.compile(
    r"projected_fit epoch=(?P<epoch>\d+) phase=(?P<phase>\S+) "
    r"loss=(?P<loss>[-+0-9.eEnNaA]+) raw_mse=(?P<raw>[-+0-9.eEnNaA]+) "
    r"projection_distance=(?P<distance>[-+0-9.eEnNaA]+) "
    r"converged=(?P<converged>[-+0-9.eEnNaA]+) "
    r"eligible=(?P<eligible>[-+0-9.eEnNaA]+) "
    r"updates=(?P<updates>[-+0-9.eEnNaA]+)"
)
_SWITCH_RE = re.compile(
    r"Activating projection at epoch (?P<epoch>\d+) \(trigger: (?P<reason>.+)\)\."
)


class _TrainerLogStream(io.TextIOBase):
    """Text stream forwarding frontend progress to the configured trainer log."""

    def __init__(self, logger: KKTTableLogger, passthrough, validation_callback=None):
        self.logger = logger
        self.passthrough = passthrough
        self.validation_callback = validation_callback
        self.buffer = ""

    def writable(self):
        """Report that this training-log stream accepts text writes."""
        return True

    def write(self, text):
        """Forward text to the configured training log stream."""
        self.buffer += str(text)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._line(line)
        return len(text)

    def flush(self):
        """Flush pending writes to the underlying training log stream."""
        if self.buffer:
            self._line(self.buffer)
            self.buffer = ""

    def _line(self, line: str):
        match = _EPOCH_RE.fullmatch(line.strip())
        if match:
            values = match.groupdict()
            validation = None
            if self.validation_callback is not None:
                validation = self.validation_callback(values["phase"] == "projected")
            self.logger.epoch(
                int(values["epoch"]),
                {
                    "loss": float(values["loss"]),
                    "forward_converged_fraction": float(values["converged"]),
                    "gradient_eligible_fraction": float(values["eligible"]),
                },
                validation,
            )
            return
        match = _SWITCH_RE.fullmatch(line.strip())
        if match:
            self.logger.phase_switch(int(match.group("epoch")), match.group("reason"))
            return
        if line.strip():
            print(line, file=self.passthrough)


class ProjectedTrainer(ProjectedTrainerCore):
    """The unchanged core model plus opt-in public conveniences.

    Numerical graph construction, training steps, sparse projection and custom
    VJP execution are inherited unchanged. This façade only formats already
    produced host-side log lines and adds post-training serialization helpers.
    """

    def save_checkpoint(self, path):
        """Save parameters and optimizer state using optional Orbax."""
        from ohagi.io.orbax import save_model_checkpoint

        return save_model_checkpoint(self, path)

    def restore_checkpoint(self, path):
        """Restore parameters and optimizer into an equivalent built model."""
        from ohagi.io.orbax import restore_model_checkpoint

        return restore_model_checkpoint(self, path)

    def fit(
        self,
        x: Any,
        y: Any | None = None,
        *,
        epochs: int,
        shuffle: bool = True,
        seed: int = 0,
        print_every: int = 0,
        pretrain_epochs: int = 0,
        eta: float | None = None,
        switch_metric: str = "raw_mse",
        validation_data: tuple[Any, Any | None] | None = None,
        log_style: str = "kkt",
    ):
        """Train the projected model using the shared epoch engine."""
        if log_style not in {"kkt", "compact", "none"}:
            raise ValueError("log_style must be kkt, compact, or none")
        if log_style == "compact":
            history = super().fit(
                x,
                y,
                epochs=epochs,
                shuffle=shuffle,
                seed=seed,
                print_every=print_every,
                pretrain_epochs=pretrain_epochs,
                eta=eta,
                switch_metric=switch_metric,
            )
        elif log_style == "none" or not print_every:
            history = super().fit(
                x,
                y,
                epochs=epochs,
                shuffle=shuffle,
                seed=seed,
                print_every=0,
                pretrain_epochs=pretrain_epochs,
                eta=eta,
                switch_metric=switch_metric,
            )
        else:
            self._require_built()
            assert self.prepared_projection is not None
            compiled = self.prepared_projection.compiled
            logger = KKTTableLogger(epochs=int(epochs), print_every=int(print_every))
            logger.start(
                dims={
                    "n_x": self.input_dim,
                    "n_y": len(compiled.variable_names),
                    "n_eq": int(compiled.initial_state.y_c.shape[1]),
                    "n_ineq": int(compiled.initial_state.y_d.shape[1]),
                },
                task=self.mode,
                mode="inverse" if self.mode == "inverse" else "forward",
                samples=len(x),
                validation_samples=(
                    0 if validation_data is None else len(validation_data[0])
                ),
                batch_size=self.batch_size,
                network=getattr(self.backbone_apply, "__qualname__", "functional"),
                eta=eta,
                pretrain_epochs=int(pretrain_epochs),
                consistency_weight=self.consistency_weight,
            )

            def validation_callback(use_projection):
                if validation_data is None:
                    return None
                return self.evaluate(
                    validation_data[0],
                    validation_data[1],
                    use_projection=use_projection,
                )

            stream = _TrainerLogStream(
                logger,
                sys.stdout,
                validation_callback=validation_callback,
            )
            with contextlib.redirect_stdout(stream):
                history = super().fit(
                    x,
                    y,
                    epochs=epochs,
                    shuffle=shuffle,
                    seed=seed,
                    print_every=print_every,
                    pretrain_epochs=pretrain_epochs,
                    eta=eta,
                    switch_metric=switch_metric,
                )
            stream.flush()
            logger.finish()

        if validation_data is not None and history:
            validation = self.evaluate(validation_data[0], validation_data[1])
            history[-1].update(
                {f"final_validation_{key}": value for key, value in validation.items()}
            )
        return history

    def _checkpoint_payload(self):
        if self.parameters is None:
            raise RuntimeError("Build the DAEHN model first.")
        return {
            "parameters": self.parameters,
            "optimizer_state": self.optimizer_state,
        }, {"framework": getattr(self, "framework", "custom")}

    def _restore_checkpoint_payload(self, tree, metadata):
        self.parameters, self.optimizer_state = (
            tree["parameters"],
            tree["optimizer_state"],
        )


# Host-side reporting faÃƒÂ§ade.


_EPOCH_RE = re.compile(
    r"projected_fit epoch=(?P<epoch>\d+) phase=(?P<phase>\S+) "
    r"loss=(?P<loss>[-+0-9.eEnNaA]+) raw_mse=(?P<raw>[-+0-9.eEnNaA]+) "
    r"projection_distance=(?P<distance>[-+0-9.eEnNaA]+) "
    r"converged=(?P<converged>[-+0-9.eEnNaA]+) "
    r"eligible=(?P<eligible>[-+0-9.eEnNaA]+) "
    r"updates=(?P<updates>[-+0-9.eEnNaA]+)"
)
_SWITCH_RE = re.compile(
    r"Activating projection at epoch (?P<epoch>\d+) \(trigger: (?P<reason>.+)\)\."
)


class _ReportedLogStream(io.TextIOBase):
    """Convert compact core rows into the public KKT-HardNet table.

    The core emits one compact line per epoch when the faÃƒÂ§ade requests it.  An
    optional callback augments that line with constraint violations and
    validation metrics.  The logger still decides which epochs are displayed,
    while ``records`` retains metrics for every epoch for history plotting.
    """

    def __init__(
        self,
        logger: KKTTableLogger,
        passthrough,
        epoch_callback=None,
        validation_callback=None,
    ):
        self.logger = logger
        self.passthrough = passthrough
        self.epoch_callback = epoch_callback
        self.validation_callback = validation_callback
        self.buffer = ""
        self.records: dict[int, dict[str, Any]] = {}

    def writable(self):
        """Report that this training-log stream accepts text writes."""
        return True

    def write(self, text):
        """Forward text to the configured training log stream."""
        self.buffer += str(text)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._line(line)
        return len(text)

    def flush(self):
        """Flush pending writes to the underlying training log stream."""
        if self.buffer:
            self._line(self.buffer)
            self.buffer = ""

    def _line(self, line: str):
        """Translate one completed log line into the frontend reporting format."""
        match = _EPOCH_RE.fullmatch(line.strip())
        if match:
            values = match.groupdict()
            epoch = int(values["epoch"])
            phase = values["phase"]
            default_loss = (
                float(values["raw"])
                if phase in {"pretrain", "projected"}
                else float(values["loss"])
            )
            train = {"loss": default_loss, "violation": 0.0}
            validation = None
            extras: dict[str, Any] = {}
            if self.epoch_callback is not None:
                callback_result = self.epoch_callback(epoch, phase, values)
                if callback_result is not None:
                    train, validation, extras = callback_result
            elif self.validation_callback is not None:
                validation = self.validation_callback(phase == "projected")
                train = {
                    "loss": float(values["loss"]),
                    "violation": float(values.get("distance", 0.0)),
                }
                if validation is not None:
                    validation = {
                        "loss": float(validation.get("loss", 0.0)),
                        "violation": float(
                            validation.get(
                                "max_constraint_violation",
                                validation.get("max_forward_residual", 0.0),
                            )
                        ),
                    }
            record = {
                "train_loss": float(train["loss"]),
                "train_violation": float(train["violation"]),
            }
            if validation is not None:
                record.update(
                    {
                        "validation_loss": float(validation["loss"]),
                        "validation_violation": float(validation["violation"]),
                    }
                )
            record.update(extras)
            self.records[epoch] = record
            self.logger.epoch(epoch, train, validation)
            return
        match = _SWITCH_RE.fullmatch(line.strip())
        if match:
            self.logger.phase_switch(int(match.group("epoch")), match.group("reason"))
            return
        if line.strip():
            print(line, file=self.passthrough)


class ReportedProjectedModel(ProjectedTrainerCore):
    """The preserved numerical model plus reporting and serialization helpers.

    Sparse projection, condensed KKT solves, implicit differentiation and
    optimizer updates remain inherited from the core implementation.  This
    faÃƒÂ§ade adds host-side feasibility reporting, history plots, timing,
    summaries and user-facing completion messages.
    """

    def __init__(self, *args, model_name: str = "KKTHN", **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = str(model_name)
        self.history: list[dict[str, Any]] = []
        self.training_time_seconds: float | None = None
        self.train_samples: int = 0
        self.validation_samples: int = 0
        self._training_inputs: np.ndarray | None = None
        self._validation_inputs: np.ndarray | None = None
        self.final_train_violation: float | None = None
        self.final_validation_violation: float | None = None

    def _display_loss(self, metrics: dict[str, Any]) -> float:
        if self.mode in {"supervised", "inverse"}:
            return float(metrics.get("raw_mse", metrics["loss"]))
        return float(metrics["loss"])

    def _current_inverse_parameters(self):
        if self.mode != "inverse":
            return None
        if not isinstance(self.parameters, dict) or "inverse" not in self.parameters:
            raise RuntimeError("Inverse model parameters are unavailable.")
        return self.parameters["inverse"]

    def _maximum_constraint_violation(self, x: Any, *, use_projection: bool) -> float:
        """Return max(abs(equalities), positive-part(inequalities))."""

        self._require_built()
        assert self.graphs is not None
        assert self.prepared_projection is not None
        assert self.projection_layer is not None
        x_host = np.asarray(x, dtype=np.float64)
        if x_host.ndim != 2 or x_host.shape[1] != self.input_dim:
            raise ValueError(
                f"x must have shape (N, {self.input_dim}); got {x_host.shape}."
            )
        if x_host.shape[0] == 0:
            return 0.0

        compiled = self.prepared_projection.compiled
        device = self.prepared_projection.device
        inverse = self._current_inverse_parameters()
        indices = np.arange(x_host.shape[0])
        maximum = 0.0

        for x_batch, _y_batch, _valid_mask, count in self._fixed_batches(
            x_host, None, indices=indices
        ):
            x_device = jax.device_put(jnp.asarray(x_batch), device)
            if use_projection:
                prediction = self.graphs.predict_step(self.parameters, x_device)
                y_hat = prediction.y_hat
                candidate = prediction.y_tilde
            else:
                if self.graphs.raw_predict_step is None:
                    raise RuntimeError(
                        "Raw constraint evaluation is unavailable because raw "
                        "prediction was not compiled."
                    )
                y_hat = self.graphs.raw_predict_step(self.parameters, x_device)
                candidate = y_hat

            if self.problem_parameter_fn is None:
                problem_parameters = self._default_problem_parameters(
                    x_device, inverse, self.projection_layer.physical_parameter_count
                )
            else:
                problem_parameters = jnp.asarray(
                    self.problem_parameter_fn(x_device, inverse), dtype=jnp.float64
                )
            packed = self.projection_layer.pack_parameters(problem_parameters, y_hat)
            equality_values = np.asarray(
                jax.device_get(compiled.equality_batch_function(candidate, packed)),
                dtype=np.float64,
            )[:count]
            inequality_values = np.asarray(
                jax.device_get(compiled.inequality_batch_function(candidate, packed)),
                dtype=np.float64,
            )[:count]

            if equality_values.size:
                equality_violation = np.abs(equality_values)
                equality_violation = np.nan_to_num(
                    equality_violation, nan=np.inf, posinf=np.inf, neginf=np.inf
                )
                maximum = max(maximum, float(np.max(equality_violation)))
            if inequality_values.size:
                inequality_violation = np.maximum(inequality_values, 0.0)
                inequality_violation = np.nan_to_num(
                    inequality_violation, nan=np.inf, posinf=np.inf, neginf=np.inf
                )
                maximum = max(maximum, float(np.max(inequality_violation)))
        return maximum

    def fit(
        self,
        x: Any,
        y: Any | None = None,
        *,
        epochs: int,
        shuffle: bool = True,
        seed: int = 0,
        print_every: int = 0,
        pretrain_epochs: int = 0,
        eta: float | None = None,
        switch_metric: str = "raw_mse",
        validation_data: tuple[Any, Any | None] | None = None,
        log_style: str = "kkt",
    ):
        """Train the model while collecting frontend progress and artifact reports."""
        if log_style not in {"kkt", "compact", "none"}:
            raise ValueError("log_style must be kkt, compact, or none")

        x_host = np.asarray(x, dtype=np.float64)
        validation_x = (
            None
            if validation_data is None
            else np.asarray(validation_data[0], dtype=np.float64)
        )
        fit_start = time.perf_counter()
        stream = None
        logger = None

        if log_style == "compact":
            history = super().fit(
                x,
                y,
                epochs=epochs,
                shuffle=shuffle,
                seed=seed,
                print_every=print_every,
                pretrain_epochs=pretrain_epochs,
                eta=eta,
                switch_metric=switch_metric,
            )
        elif log_style == "none" or not print_every:
            history = super().fit(
                x,
                y,
                epochs=epochs,
                shuffle=shuffle,
                seed=seed,
                print_every=0,
                pretrain_epochs=pretrain_epochs,
                eta=eta,
                switch_metric=switch_metric,
            )
        else:
            self._require_built()
            assert self.prepared_projection is not None
            compiled = self.prepared_projection.compiled
            logger = KKTTableLogger(epochs=int(epochs), print_every=int(print_every))
            logger.start(
                dims={
                    "n_x": self.input_dim,
                    "n_y": len(compiled.variable_names),
                    "n_eq": int(compiled.initial_state.y_c.shape[1]),
                    "n_ineq": int(compiled.initial_state.y_d.shape[1]),
                },
                task=self.mode,
                mode="inverse" if self.mode == "inverse" else "forward",
                samples=len(x_host),
                validation_samples=(0 if validation_x is None else len(validation_x)),
                batch_size=self.batch_size,
                network=getattr(self.backbone_apply, "__qualname__", "functional"),
                eta=eta,
                pretrain_epochs=int(pretrain_epochs),
                consistency_weight=self.consistency_weight,
                dual_pretrain_gamma=(
                    self.dual_pretrain_gamma if self.predict_duals else None
                ),
                predicted_dual_projection=self.predict_duals,
            )

            def epoch_callback(epoch, phase, values):
                del epoch
                use_projection = phase == "projected"
                train_loss = (
                    float(values["raw"])
                    if self.mode in {"supervised", "inverse"}
                    else float(values["loss"])
                )
                train_violation = self._maximum_constraint_violation(
                    x_host, use_projection=use_projection
                )
                train_metrics = {"loss": train_loss, "violation": train_violation}
                validation_metrics = None
                extras = {"max_constraint_violation": train_violation}
                if validation_data is not None:
                    evaluated = self.evaluate(
                        validation_data[0],
                        validation_data[1],
                        use_projection=use_projection,
                    )
                    validation_violation = self._maximum_constraint_violation(
                        validation_data[0], use_projection=use_projection
                    )
                    validation_metrics = {
                        "loss": self._display_loss(evaluated),
                        "violation": validation_violation,
                    }
                    extras.update(
                        {
                            (
                                "validation_objective_loss"
                                if key == "loss"
                                else f"validation_{key}"
                            ): float(value)
                            for key, value in evaluated.items()
                        }
                    )
                    extras["validation_max_constraint_violation"] = validation_violation
                return train_metrics, validation_metrics, extras

            stream = _ReportedLogStream(
                logger, sys.stdout, epoch_callback=epoch_callback
            )
            # Capture every compact row so history contains train/validation
            # feasibility at every epoch. KKTTableLogger controls display rate.
            with contextlib.redirect_stdout(stream):
                history = super().fit(
                    x,
                    y,
                    epochs=epochs,
                    shuffle=shuffle,
                    seed=seed,
                    print_every=1,
                    pretrain_epochs=pretrain_epochs,
                    eta=eta,
                    switch_metric=switch_metric,
                )
            stream.flush()
            for row in history:
                row.update(stream.records.get(int(row["epoch"]), {}))

        self.train_samples = int(len(x_host))
        self.validation_samples = 0 if validation_x is None else int(len(validation_x))
        representative_count = min(int(self.batch_size or 1), int(x_host.shape[0]))
        self._training_inputs = np.array(x_host[:representative_count], copy=True)
        self._validation_inputs = (
            None
            if validation_x is None
            else np.array(
                validation_x[: min(int(self.batch_size or 1), len(validation_x))],
                copy=True,
            )
        )
        self.history = history

        final_is_projected = bool(history and history[-1].get("phase") == "projected")
        if history and final_is_projected and "train_violation" in history[-1]:
            self.final_train_violation = float(history[-1]["train_violation"])
        elif history:
            self.final_train_violation = self._maximum_constraint_violation(
                x_host, use_projection=True
            )
        else:
            self.final_train_violation = None
        if history and self.final_train_violation is not None:
            history[-1]["final_train_max_constraint_violation"] = (
                self.final_train_violation
            )

        if validation_data is not None and history:
            if final_is_projected and "validation_violation" in history[-1]:
                self.final_validation_violation = float(
                    history[-1]["validation_violation"]
                )
                for key, value in tuple(history[-1].items()):
                    if key.startswith("validation_") and not key.startswith(
                        "validation_violation"
                    ):
                        history[-1][f"final_{key}"] = value
            else:
                final_validation = self.evaluate(validation_data[0], validation_data[1])
                history[-1].update(
                    {
                        f"final_validation_{key}": value
                        for key, value in final_validation.items()
                    }
                )
                self.final_validation_violation = self._maximum_constraint_violation(
                    validation_data[0], use_projection=True
                )
            history[-1]["final_validation_max_constraint_violation"] = (
                self.final_validation_violation
            )
        else:
            self.final_validation_violation = None

        training_seconds = float(time.perf_counter() - fit_start)
        self.training_time_seconds = training_seconds
        if logger is not None:
            logger.finish(training_seconds)
        elif log_style == "compact" and print_every:
            print(f"Model training time (wall time) = {training_seconds:.4f} seconds")
            print("Model training finished!")
        return history

    def _benchmark_inference(self, *, single: bool, warmup: int, repeats: int) -> float:
        self._require_built()
        assert self.graphs is not None
        assert self.prepared_projection is not None
        assert self.batch_size is not None
        source = self._training_inputs
        if source is None or source.shape[0] == 0:
            source = np.zeros((1, self.input_dim), dtype=np.float64)
        count = 1 if single else min(self.batch_size, source.shape[0])
        fixed = np.repeat(source[count - 1 : count], self.batch_size, axis=0)
        fixed[:count] = source[:count]
        x_device = jax.device_put(
            jnp.asarray(fixed, dtype=jnp.float64), self.prepared_projection.device
        )
        for _ in range(max(0, int(warmup))):
            result = self.graphs.predict_step(self.parameters, x_device)
            jax.block_until_ready(result.y_tilde)
        timings = []
        for _ in range(max(1, int(repeats))):
            started = time.perf_counter()
            result = self.graphs.predict_step(self.parameters, x_device)
            jax.block_until_ready(result.y_tilde)
            timings.append(time.perf_counter() - started)
        return float(np.median(np.asarray(timings)) * 1e3)

    def summary(
        self,
        *,
        model_name: str | None = None,
        inference_warmup: int = 2,
        inference_repeats: int = 10,
    ) -> dict[str, Any]:
        """Print and return a KKT-HardNet-style model summary."""

        self._require_built()
        assert self.prepared_projection is not None
        compiled = self.prepared_projection.compiled
        output_dim = len(compiled.variable_names)
        physical_parameters = self.prepared_projection.parameter_count - output_dim
        final_row = self.history[-1] if self.history else {}
        violations = [
            float(value)
            for value in (self.final_train_violation, self.final_validation_violation)
            if value is not None and np.isfinite(float(value))
        ]
        if not violations:
            violations = [
                float(final_row[key])
                for key in (
                    "train_violation",
                    "validation_violation",
                    "max_constraint_violation",
                    "final_train_max_constraint_violation",
                    "final_validation_max_constraint_violation",
                )
                if key in final_row and np.isfinite(float(final_row[key]))
            ]
        maximum_violation = max(violations, default=0.0)
        single_ms = self._benchmark_inference(
            single=True, warmup=inference_warmup, repeats=inference_repeats
        )
        batch_ms = self._benchmark_inference(
            single=False, warmup=inference_warmup, repeats=inference_repeats
        )
        training_seconds = float(self.training_time_seconds or 0.0)
        selected_name = self.model_name if model_name is None else str(model_name)
        report = {
            "model_name": selected_name,
            "parameter_count": int(physical_parameters),
            "variable_count": int(output_dim),
            "equality_count": int(compiled.initial_state.y_c.shape[1]),
            "inequality_count": int(compiled.initial_state.y_d.shape[1]),
            "train_samples": int(self.train_samples),
            "validation_samples": int(self.validation_samples),
            "maximum_constraint_violation": float(maximum_violation),
            "training_time_seconds": training_seconds,
            "jax_single_inference_ms": single_ms,
            "jax_batch_inference_ms": batch_ms,
        }
        print("KKT-HardNet Summary")
        print("-" * 60)
        print(f"Model Name                     : {selected_name}")
        print(f"No. of Parameters              : {physical_parameters}")
        print(f"No. of Variables               : {output_dim}")
        print(f"No. of Equalities              : {compiled.initial_state.y_c.shape[1]}")
        print(f"No. of Inequalities            : {compiled.initial_state.y_d.shape[1]}")
        print(f"No. of Train Samples           : {self.train_samples}")
        print(f"No. of Validation Samples      : {self.validation_samples}")
        print(f"Maximum Constraint Violation   : {maximum_violation:.4e}")
        print(f"Training Time                  : {training_seconds:.2f} s")
        print(f"Est. JAX Single Inference Time : {single_ms:.2f} ms")
        print(f"Est. JAX Batch Inference Time  : {batch_ms:.2f} ms")
        print("-" * 60)
        print(
            "Note: Inference time estimations are based on microbenchmarking "
            "on the hardware used during training and may vary across "
            "different hardware and runtime conditions."
        )
        return report

    def plot_history(
        self,
        history: list[dict[str, Any]] | None = None,
        *,
        save_path: str | Path | None = None,
        show: bool = True,
        figsize: tuple[float, float] = (18.0, 5.0),
    ):
        """Plot train/validation MSE and maximum constraint violation."""

        selected = self.history if history is None else history
        if not selected:
            raise RuntimeError("No training history is available to plot.")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "plot_history requires matplotlib. Install kkthn[plotting] or matplotlib directly."
            ) from exc

        epochs = np.asarray(
            [int(row.get("epoch", index + 1)) for index, row in enumerate(selected)]
        )
        train_loss = np.asarray(
            [
                float(
                    row.get(
                        "train_loss",
                        row.get(
                            "raw_mse"
                            if self.mode in {"supervised", "inverse"}
                            else "loss",
                            row["loss"],
                        ),
                    )
                )
                for row in selected
            ]
        )
        if not all("train_violation" in row for row in selected):
            raise RuntimeError(
                "Constraint-violation history is unavailable. Train with "
                "log_style='kkt' and print_every > 0."
            )
        train_violation = np.asarray(
            [float(row["train_violation"]) for row in selected]
        )
        has_validation = all(
            "validation_loss" in row and "validation_violation" in row
            for row in selected
        )
        if has_validation:
            validation_loss = np.asarray(
                [float(row["validation_loss"]) for row in selected]
            )
            validation_violation = np.asarray(
                [float(row["validation_violation"]) for row in selected]
            )

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        loss_axis, violation_axis = axes
        loss_axis.plot(epochs, train_loss, label="Train MSE")
        if has_validation:
            loss_axis.plot(
                epochs, validation_loss, linestyle="--", label="Validation MSE"
            )
        loss_axis.set_title("Loss", fontsize=30)
        loss_axis.set_xlabel("Epoch", fontsize=22)
        loss_axis.set_ylabel("MSE", fontsize=22)
        loss_axis.tick_params(axis="both", labelsize=16)
        loss_axis.grid(True, alpha=0.3)
        loss_axis.legend(fontsize=20)

        violation_series = [train_violation]
        if has_validation:
            violation_series.append(validation_violation)
        combined_violation = np.concatenate(violation_series)
        positive_violation = combined_violation[
            np.isfinite(combined_violation) & (combined_violation > 0.0)
        ]
        violation_floor = (
            max(float(np.min(positive_violation)) * 0.1, 1e-16)
            if positive_violation.size
            else 1e-16
        )
        violation_axis.plot(
            epochs,
            np.maximum(train_violation, violation_floor),
            label="Train Violation",
        )
        if has_validation:
            violation_axis.plot(
                epochs,
                np.maximum(validation_violation, violation_floor),
                linestyle="--",
                label="Validation Violation",
            )
        violation_axis.set_yscale("log")
        violation_axis.set_title("Violation", fontsize=30)
        violation_axis.set_xlabel("Epoch", fontsize=22)
        violation_axis.set_ylabel("Max. Violation", fontsize=22)
        violation_axis.tick_params(axis="both", labelsize=16)
        violation_axis.grid(True, alpha=0.3)
        violation_axis.legend(fontsize=20)
        fig.tight_layout()

        if save_path is not None:
            destination = Path(save_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return fig, axes

    @staticmethod
    def print_postprocessing_start() -> None:
        """Announce the start of model postprocessing."""
        print_postprocessing_start()

    @staticmethod
    def print_postprocessing_finish(*, include_citation: bool = True) -> None:
        """Report completion of model postprocessing."""
        print_postprocessing_finish(include_citation=include_citation)

    @staticmethod
    def print_citation() -> None:
        """Print the research citation associated with the frontend."""
        print_research_citation()

    def save_predictor(
        self,
        directory,
        *,
        network_spec: dict[str, Any] | None = None,
        projection_spec: dict[str, Any] | None = None,
        problem=None,
        input_transform: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        predictor_factory: str | None = None,
        factory_config: dict[str, Any] | None = None,
        verbose: bool = True,
    ):
        """Write a prepared JAX predictor in the shared OHAGI format."""
        self._require_built()
        assert self.prepared_projection is not None
        from ohagi.serialization.predictor import save_predictor, model_graph

        serialized_network_spec = dict(network_spec or {"type": "jax_graph"})
        if isinstance(self.parameters, dict) and "backbone" in self.parameters:
            serialized_network_spec.setdefault("parameter_path", ["backbone"])

        predictor = save_predictor(
            directory,
            network_spec=serialized_network_spec,
            params=self.parameters,
            input_size=self.input_dim,
            output_size=len(self.prepared_projection.compiled.variable_names),
            batch_size=self.batch_size,
            projection_spec=projection_spec,
            problem=problem,
            input_transform=input_transform,
            dtype="float64",
            backend=self.prepared_projection.backend,
            metadata=metadata,
            predictor_factory=predictor_factory,
            factory_config=factory_config,
            _inference_spec=model_graph(
                self, framework=getattr(self, "framework", "custom")
            ),
            framework=getattr(self, "framework", "custom"),
        )
        if verbose:
            print("Model saved!")
        return predictor


def _compile_projected_graphs(
    compile_raw_predict,
    device,
    dummy_mask,
    dummy_x,
    dummy_y,
    eval_step,
    mode,
    optimizer_state_device,
    parameters_device,
    predict_step,
    raw_predict_step,
    train_step,
    validate,
):
    """Compile and validate the prepared train, evaluation and prediction graphs."""
    train_compile_start = time.perf_counter()
    train_jit = jax.jit(train_step, donate_argnums=(0, 1))
    if mode in {"supervised", "inverse"}:
        train_executable = train_jit.lower(
            parameters_device, optimizer_state_device, dummy_x, dummy_y, dummy_mask
        ).compile()
    else:
        train_executable = train_jit.lower(
            parameters_device, optimizer_state_device, dummy_x, dummy_mask
        ).compile()
    train_compile_seconds = time.perf_counter() - train_compile_start

    eval_compile_start = time.perf_counter()
    eval_jit = jax.jit(eval_step)
    if mode in {"supervised", "inverse"}:
        eval_executable = eval_jit.lower(
            parameters_device, dummy_x, dummy_y, dummy_mask
        ).compile()
    else:
        eval_executable = eval_jit.lower(
            parameters_device, dummy_x, dummy_mask
        ).compile()
    eval_compile_seconds = time.perf_counter() - eval_compile_start

    predict_compile_start = time.perf_counter()
    predict_executable = (
        jax.jit(predict_step).lower(parameters_device, dummy_x).compile()
    )
    if compile_raw_predict:
        raw_predict_executable = (
            jax.jit(raw_predict_step).lower(parameters_device, dummy_x).compile()
        )
    else:
        raw_predict_executable = None
    predict_compile_seconds = time.perf_counter() - predict_compile_start

    validation_seconds = 0.0
    validated = False
    if validate:
        validation_start = time.perf_counter()
        prediction = predict_executable(parameters_device, dummy_x)
        jax.block_until_ready(prediction.y_tilde)
        validation_parameters = _copy_tree_to_device(parameters_device, device)
        validation_optimizer_state = _copy_tree_to_device(
            optimizer_state_device, device
        )
        if mode in {"supervised", "inverse"}:
            validation_result = train_executable(
                validation_parameters,
                validation_optimizer_state,
                dummy_x,
                dummy_y,
                dummy_mask,
            )
        else:
            validation_result = train_executable(
                validation_parameters,
                validation_optimizer_state,
                dummy_x,
                dummy_mask,
            )
        jax.block_until_ready(validation_result[2].loss)
        validation_metrics = jax.device_get(validation_result[2])
        if not np.isfinite(float(validation_metrics.loss)):
            raise RuntimeError("Build validation produced a nonfinite loss.")
        if float(validation_metrics.forward_converged_fraction) < 1.0:
            raise RuntimeError(
                "Build validation projection did not converge for all "
                "synthetic rows. Provide a projection factory whose default "
                "zero batch is feasible or call build(validate=False)."
            )
        validation_seconds = time.perf_counter() - validation_start
        validated = True
    return (
        eval_compile_seconds,
        eval_executable,
        predict_compile_seconds,
        predict_executable,
        raw_predict_executable,
        train_compile_seconds,
        train_executable,
        validated,
        validation_seconds,
    )


def _prepare_projected_examples(
    self,
    batch_size_static,
    device,
    example_x,
    example_y,
    mode,
    optimizer_state_device,
    output_dim,
    parameters_device,
    raw_eval_step,
    raw_train_step,
):
    """Prepare fixed-shape examples and optional raw training executables."""

    def fixed_example(values, width):
        if values is None:
            host = np.zeros((batch_size_static, width), dtype=np.float64)
        else:
            raw = np.asarray(values, dtype=np.float64)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            if raw.ndim != 2 or raw.shape[1] != width or raw.shape[0] < 1:
                raise ValueError(
                    f"Build example must have shape (N, {width}) with N >= 1; got {raw.shape}."
                )
            host = np.repeat(raw[-1:, :], batch_size_static, axis=0)
            count = min(batch_size_static, raw.shape[0])
            host[:count] = raw[:count]
        return jax.device_put(jnp.asarray(host, dtype=self.dtype), device)

    dummy_x = fixed_example(example_x, self.input_dim)
    dummy_y = fixed_example(example_y, output_dim)
    dummy_mask = jax.device_put(jnp.ones((batch_size_static,), dtype=jnp.bool_), device)

    raw_train_compile_seconds = 0.0
    raw_eval_compile_seconds = 0.0
    raw_train_executable = None
    raw_eval_executable = None
    if mode in {"supervised", "inverse"}:
        assert raw_train_step is not None
        assert raw_eval_step is not None
        raw_train_compile_start = time.perf_counter()
        raw_train_executable = (
            jax.jit(raw_train_step, donate_argnums=(0, 1))
            .lower(
                parameters_device,
                optimizer_state_device,
                dummy_x,
                dummy_y,
                dummy_mask,
            )
            .compile()
        )
        raw_train_compile_seconds = time.perf_counter() - raw_train_compile_start

        raw_eval_compile_start = time.perf_counter()
        raw_eval_executable = (
            jax.jit(raw_eval_step)
            .lower(parameters_device, dummy_x, dummy_y, dummy_mask)
            .compile()
        )
        raw_eval_compile_seconds = time.perf_counter() - raw_eval_compile_start
    return (
        dummy_mask,
        dummy_x,
        dummy_y,
        raw_eval_compile_seconds,
        raw_eval_executable,
        raw_train_compile_seconds,
        raw_train_executable,
    )


def _construct_projected_steps(
    self,
    batch_size_static,
    data_loss_fn,
    forward_outputs,
    graph_construct_start,
    invalid_batch_policy,
    metrics_from_losses,
    mode,
    objective_fn,
    optimizer,
    optimizer_apply,
    raw_metrics_from_losses,
    raw_model,
    split_trainable,
):
    """Construct pure training, evaluation and prediction transitions."""
    from ohagi.training.steps import parameter_train_step, StepPolicy

    step_policy = StepPolicy(sanitize_gradients=True, check_candidate_finite=True)

    def eligible(metrics):
        fraction = metrics.gradient_eligible_fraction
        return (
            fraction >= (1.0 - 1e-15)
            if invalid_batch_policy == "skip"
            else fraction > 0.0
        )

    def make_step(objective):
        return parameter_train_step(
            objective,
            optimizer,
            policy=step_policy,
            eligible=eligible,
            optimizer_apply_updates=optimizer_apply,
            format_metrics=lambda loss, metrics, valid: metrics._replace(
                update_accepted=valid.astype(jnp.float64)
            ),
        )

    if mode in {"supervised", "inverse"}:

        def raw_loss_with_aux(parameters, x_batch, y_data, valid_mask):
            network_parameters, inverse = split_trainable(parameters)
            outputs, _ = raw_model.apply(parameters, {}, x_batch)
            y_hat = outputs["raw"]
            if data_loss_fn is None:
                task_values = jnp.mean(jnp.square(y_hat - y_data), axis=1)
            else:
                task_values = data_loss_fn(y_hat, y_data, x_batch)
            task_per_sample = _per_sample(task_values, batch_size_static)
            raw_mse_per_sample = jnp.mean(jnp.square(y_hat - y_data), axis=1)
            return raw_metrics_from_losses(
                task_per_sample, raw_mse_per_sample, valid_mask, inverse
            )

        raw_train_step = make_step(raw_loss_with_aux)

        def raw_eval_step(parameters, x_batch, y_data, valid_mask):
            _loss, metrics = raw_loss_with_aux(parameters, x_batch, y_data, valid_mask)
            return metrics

        def loss_with_aux(parameters, x_batch, y_data, valid_mask):
            _network_parameters, inverse = split_trainable(parameters)
            (y_hat, lambda_predicted, mu_predicted, projection_output) = (
                forward_outputs(parameters, x_batch)
            )
            if data_loss_fn is None:
                task_values = jnp.mean(
                    jnp.square(projection_output.y_tilde - y_data), axis=1
                )
            else:
                task_values = data_loss_fn(projection_output.y_tilde, y_data, x_batch)
            task_per_sample = _per_sample(task_values, batch_size_static)
            raw_mse_per_sample = jnp.mean(
                jnp.square(projection_output.y_tilde - y_data), axis=1
            )
            return metrics_from_losses(
                task_per_sample,
                raw_mse_per_sample,
                y_hat,
                lambda_predicted,
                mu_predicted,
                projection_output,
                valid_mask,
                inverse,
            )

        train_step = make_step(loss_with_aux)

        def eval_step(parameters, x_batch, y_data, valid_mask):
            _loss, metrics = loss_with_aux(parameters, x_batch, y_data, valid_mask)
            return metrics

    else:
        raw_train_step = None
        raw_eval_step = None

        def loss_with_aux(parameters, x_batch, valid_mask):
            _network_parameters, inverse = split_trainable(parameters)
            (y_hat, lambda_predicted, mu_predicted, projection_output) = (
                forward_outputs(parameters, x_batch)
            )
            assert objective_fn is not None
            task_values = objective_fn(projection_output.y_tilde, x_batch)
            task_per_sample = _per_sample(task_values, batch_size_static)
            return metrics_from_losses(
                task_per_sample,
                task_per_sample,
                y_hat,
                lambda_predicted,
                mu_predicted,
                projection_output,
                valid_mask,
                inverse,
            )

        train_step = make_step(loss_with_aux)

        def eval_step(parameters, x_batch, valid_mask):
            _loss, metrics = loss_with_aux(parameters, x_batch, valid_mask)
            return metrics

    def predict_step(parameters, x_batch):
        (y_hat, _lambda_predicted, _mu_predicted, projection_output) = forward_outputs(
            parameters, x_batch
        )
        return PredictionOutput(
            y_hat=y_hat,
            y_tilde=projection_output.y_tilde,
            forward_status_codes=projection_output.forward_status_codes,
            forward_iterations=projection_output.forward_iterations,
            forward_residual_inf_norm=projection_output.forward_residual_inf_norm,
        )

    def raw_predict_step(parameters, x_batch):
        outputs, _ = raw_model.apply(parameters, {}, x_batch)
        return outputs["raw"]

    self._inference_forward = predict_step
    graph_construct_seconds = time.perf_counter() - graph_construct_start
    return (
        eval_step,
        graph_construct_seconds,
        predict_step,
        raw_eval_step,
        raw_predict_step,
        raw_train_step,
        train_step,
    )


def _prepare_projected_metrics(
    consistency_weight,
    equality_count,
    equality_dual_consistency_weight,
    inequality_count,
    inequality_dual_consistency_weight,
    invalid_batch_policy,
    inverse_regularizer,
    model,
    predict_duals,
):
    """Prepare forward evaluation and task-specific metric callables."""

    def forward_outputs(parameters, x_batch):
        outputs, _ = model.apply(parameters, {}, x_batch)
        return (
            outputs["raw"],
            outputs["lambda_raw"],
            outputs["mu_raw"],
            outputs["projection_result"],
        )

    def inverse_regularization_value(inverse):
        if inverse is not None and inverse_regularizer is not None:
            return jnp.asarray(inverse_regularizer(inverse), dtype=jnp.float64)
        return jnp.asarray(0.0, dtype=jnp.float64)

    def metrics_from_losses(
        task_per_sample,
        raw_mse_per_sample,
        y_hat,
        lambda_predicted,
        mu_predicted,
        projection_output,
        valid_mask,
        inverse,
    ):
        finite_outputs = (
            projection_output.forward_state_finite
            & jnp.all(jnp.isfinite(y_hat), axis=1)
            & jnp.all(jnp.isfinite(projection_output.y_tilde), axis=1)
        )
        gradient_eligible = (
            valid_mask & finite_outputs & projection_output.forward_gradient_eligible
        )
        safe_task = jnp.nan_to_num(task_per_sample, nan=0.0, posinf=0.0, neginf=0.0)
        safe_raw_mse = jnp.nan_to_num(
            raw_mse_per_sample, nan=0.0, posinf=0.0, neginf=0.0
        )
        report_mask = valid_mask & finite_outputs
        task_loss = _masked_mean(safe_task, report_mask)
        raw_mse = _masked_mean(safe_raw_mse, report_mask)
        projection_distance_per_sample = jnp.mean(
            jnp.square(jnp.nan_to_num(y_hat - projection_output.y_tilde)), axis=1
        )
        projection_distance = _masked_mean(projection_distance_per_sample, report_mask)
        consistency_loss = consistency_weight * projection_distance

        if predict_duals and equality_count:
            equality_dual_per_sample = jnp.mean(
                jnp.square(lambda_predicted - projection_output.multipliers_eq),
                axis=1,
            )
        else:
            equality_dual_per_sample = jnp.zeros_like(task_per_sample)
        if predict_duals and inequality_count:
            inequality_dual_per_sample = jnp.mean(
                jnp.square(mu_predicted - projection_output.multipliers_ineq),
                axis=1,
            )
        else:
            inequality_dual_per_sample = jnp.zeros_like(task_per_sample)

        equality_dual_consistency = equality_dual_consistency_weight * _masked_mean(
            equality_dual_per_sample, report_mask
        )
        inequality_dual_consistency = inequality_dual_consistency_weight * _masked_mean(
            inequality_dual_per_sample, report_mask
        )
        inverse_reg = inverse_regularization_value(inverse)
        reported_total_loss = (
            task_loss
            + consistency_loss
            + equality_dual_consistency
            + inequality_dual_consistency
            + inverse_reg
        )

        # The differentiable scalar uses only rows accepted by the selected
        # inexact-gradient policy. Metrics continue to report all finite
        # final iterates so skipped batches never appear as zero error.
        training_task_loss = _masked_mean(safe_task, gradient_eligible)
        training_projection_distance = _masked_mean(
            projection_distance_per_sample, gradient_eligible
        )
        training_equality_dual_consistency = (
            equality_dual_consistency_weight
            * _masked_mean(equality_dual_per_sample, gradient_eligible)
        )
        training_inequality_dual_consistency = (
            inequality_dual_consistency_weight
            * _masked_mean(inequality_dual_per_sample, gradient_eligible)
        )
        training_total_loss = (
            training_task_loss
            + consistency_weight * training_projection_distance
            + training_equality_dual_consistency
            + training_inequality_dual_consistency
            + inverse_reg
        )
        valid_weight = valid_mask.astype(jnp.float64)
        denominator = jnp.maximum(jnp.sum(valid_weight), 1.0)
        converged = jnp.logical_and(
            valid_mask,
            projection_output.forward_status_codes == BATCH_STATUS_CONVERGED,
        )
        converged_fraction = jnp.sum(converged.astype(jnp.float64)) / denominator
        finite_fraction = (
            jnp.sum((valid_mask & finite_outputs).astype(jnp.float64)) / denominator
        )
        eligible_fraction = jnp.sum(gradient_eligible.astype(jnp.float64)) / denominator
        masked_residual = jnp.where(
            valid_mask, projection_output.forward_residual_inf_norm, -jnp.inf
        )
        max_residual = jnp.max(masked_residual)
        mean_iterations = (
            jnp.sum(
                valid_weight * projection_output.forward_iterations.astype(jnp.float64)
            )
            / denominator
        )
        if invalid_batch_policy == "skip":
            default_update_accepted = eligible_fraction >= (1.0 - 1e-15)
        else:
            default_update_accepted = eligible_fraction > 0.0
        return training_total_loss, ProjectedMetrics(
            loss=reported_total_loss,
            task_loss=task_loss,
            raw_mse=raw_mse,
            consistency_loss=consistency_loss,
            equality_dual_consistency=equality_dual_consistency,
            inequality_dual_consistency=inequality_dual_consistency,
            inverse_regularization=inverse_reg,
            projection_distance=projection_distance,
            forward_converged_fraction=converged_fraction,
            forward_finite_fraction=finite_fraction,
            gradient_eligible_fraction=eligible_fraction,
            update_accepted=default_update_accepted.astype(jnp.float64),
            max_forward_residual=max_residual,
            mean_forward_iterations=mean_iterations,
        )

    def raw_metrics_from_losses(
        task_per_sample, raw_mse_per_sample, valid_mask, inverse
    ):
        safe_task = jnp.nan_to_num(task_per_sample, nan=0.0, posinf=0.0, neginf=0.0)
        safe_raw = jnp.nan_to_num(raw_mse_per_sample, nan=0.0, posinf=0.0, neginf=0.0)
        task_loss = _masked_mean(safe_task, valid_mask)
        raw_mse = _masked_mean(safe_raw, valid_mask)
        inverse_reg = inverse_regularization_value(inverse)
        total_loss = task_loss + inverse_reg
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        one = jnp.asarray(1.0, dtype=jnp.float64)
        return total_loss, ProjectedMetrics(
            loss=total_loss,
            task_loss=task_loss,
            raw_mse=raw_mse,
            consistency_loss=zero,
            equality_dual_consistency=zero,
            inequality_dual_consistency=zero,
            inverse_regularization=inverse_reg,
            projection_distance=zero,
            forward_converged_fraction=one,
            forward_finite_fraction=one,
            gradient_eligible_fraction=one,
            update_accepted=one,
            max_forward_residual=zero,
            mean_forward_iterations=zero,
        )

    return forward_outputs, metrics_from_losses, raw_metrics_from_losses


def _compose_projected_models(
    self,
    batch_size,
    equality_count,
    inequality_count,
    output_dim,
    physical_count,
    projection_layer,
    static_primal_lower,
    static_primal_upper,
):
    """Compose raw and projected models with frozen transformation resources."""
    batch_size_static = int(batch_size)
    consistency_weight = self.consistency_weight
    equality_dual_consistency_weight = self.equality_dual_consistency_weight
    inequality_dual_consistency_weight = self.inequality_dual_consistency_weight
    predict_duals = self.predict_duals
    dual_pretrain_gamma = self.dual_pretrain_gamma
    inequality_dual_floor = self.inequality_dual_floor
    data_loss_fn = self.data_loss_fn
    objective_fn = self.objective_fn
    inverse_regularizer = self.inverse_regularizer
    problem_parameter_fn = self.problem_parameter_fn
    backbone_input_fn = self.backbone_input_fn
    backbone_apply = self.backbone_apply
    optimizer = self.optimizer
    optimizer_apply = self.optimizer_apply_updates
    mode = self.mode
    invalid_batch_policy = self.invalid_batch_policy
    primal_bounds_fn = self.primal_bounds_fn
    static_primal_lower_jax = jnp.asarray(static_primal_lower, dtype=self.dtype)
    static_primal_upper_jax = jnp.asarray(static_primal_upper, dtype=self.dtype)

    def split_trainable(parameters):
        inverse = parameters.get("inverse") if mode == "inverse" else None
        return parameters["backbone"], inverse

    def build_problem_parameters(x_batch, inverse):
        if problem_parameter_fn is not None:
            return problem_parameter_fn(x_batch, inverse)
        return self._default_problem_parameters(x_batch, inverse, physical_count)

    def build_backbone_inputs(x_batch, inverse):
        if backbone_input_fn is not None:
            return backbone_input_fn(x_batch, inverse)
        return self._default_backbone_inputs(x_batch, inverse)

    def build_primal_bounds(problem_parameters):
        target_shape = (*problem_parameters.shape[:-1], output_dim)
        if primal_bounds_fn is None:
            lower = jnp.broadcast_to(static_primal_lower_jax, target_shape)
            upper = jnp.broadcast_to(static_primal_upper_jax, target_shape)
        else:
            lower, upper = primal_bounds_fn(problem_parameters)
            lower = jnp.broadcast_to(jnp.asarray(lower, dtype=self.dtype), target_shape)
            upper = jnp.broadcast_to(jnp.asarray(upper, dtype=self.dtype), target_shape)
        return lower, upper

    def prepare_model_inputs(parameters, x_batch, context):
        network_parameters, inverse = split_trainable(parameters)
        network_inputs = build_backbone_inputs(x_batch, inverse)
        problem_parameters = build_problem_parameters(x_batch, inverse)
        return network_parameters, network_inputs, problem_parameters

    def process_model_outputs(network_values, problem_parameters):
        primal_lower, primal_upper = build_primal_bounds(problem_parameters)
        return split_projection_outputs(
            network_values,
            n_y=output_dim,
            n_eq=equality_count,
            n_ineq=inequality_count,
            predict_duals=predict_duals,
            inequality_dual_floor=inequality_dual_floor,
            primal_lower_bounds=primal_lower,
            primal_upper_bounds=primal_upper,
        )

    model = ProjectedModel(
        FunctionalBackbone(None, backbone_apply),
        ImplicitDistance(projection_layer, predict_duals),
        prepare_inputs=prepare_model_inputs,
        output_transform=process_model_outputs,
        return_projection=True,
    )
    self._shared_model = model

    def process_raw_outputs(network_values, problem_parameters):
        primal_lower, primal_upper = build_primal_bounds(problem_parameters)
        return split_pretraining_outputs(
            network_values,
            n_y=output_dim,
            n_eq=equality_count,
            n_ineq=inequality_count,
            predict_duals=predict_duals,
            dual_pretrain_gamma=dual_pretrain_gamma,
            primal_lower_bounds=primal_lower,
            primal_upper_bounds=primal_upper,
        )

    raw_model = ProjectedModel(
        FunctionalBackbone(None, backbone_apply),
        prepare_inputs=prepare_model_inputs,
        output_transform=process_raw_outputs,
    )
    return (
        batch_size_static,
        consistency_weight,
        data_loss_fn,
        dual_pretrain_gamma,
        equality_dual_consistency_weight,
        inequality_dual_consistency_weight,
        invalid_batch_policy,
        inverse_regularizer,
        mode,
        model,
        objective_fn,
        optimizer,
        optimizer_apply,
        predict_duals,
        raw_model,
        split_trainable,
    )


def _initialize_projected_backbone(
    self, equality_count, inequality_count, network_output_dim, output_dim, seed
):
    """Initialize the selected backbone, inverse variables and optimizer state."""
    inverse_size = 0 if self.inverse_init is None else int(self.inverse_init.size)
    default_backbone_input_dim = (
        self.input_dim + inverse_size
        if self.backbone_input_dim is None
        else self.backbone_input_dim
    )
    backbone_initialize_start = time.perf_counter()
    key = jax.random.PRNGKey(int(seed))
    try:
        backbone_parameters = self.backbone_init(
            key,
            default_backbone_input_dim,
            network_output_dim,
            self.dtype,
            n_primal_outputs=output_dim if self.predict_duals else None,
            n_equality_duals=equality_count if self.predict_duals else 0,
            n_inequality_duals=inequality_count if self.predict_duals else 0,
            dual_initial_gamma=self.dual_pretrain_gamma,
            inequality_dual_floor=self.inequality_dual_floor,
        )
    except TypeError:
        # Preserve compatibility with custom initializers that implement
        # only the historical four-argument callable contract.
        backbone_parameters = self.backbone_init(
            key, default_backbone_input_dim, network_output_dim, self.dtype
        )
    if self.mode == "inverse":
        trainable_parameters = {
            "backbone": backbone_parameters,
            "inverse": jnp.asarray(self.inverse_init, dtype=self.dtype),
        }
    else:
        trainable_parameters = {"backbone": backbone_parameters}
    optimizer_state = self.optimizer.init(trainable_parameters)
    backbone_initialize_seconds = time.perf_counter() - backbone_initialize_start
    return backbone_initialize_seconds, optimizer_state, trainable_parameters


def _prepare_projected_bounds(self, output_dim):
    """Normalize static primal bounds before constructing model graphs."""

    def normalize_static_bound(values, fill_value, name):
        if values is None:
            return np.full((output_dim,), fill_value, dtype=np.float64)
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != output_dim:
            raise ValueError(
                f"{name} must contain {output_dim} entries; got {array.size}."
            )
        if np.any(np.isnan(array)):
            raise ValueError(f"{name} may contain infinities but not NaN values.")
        return array

    static_primal_lower = normalize_static_bound(
        self.primal_lower_bounds, -np.inf, "primal_lower_bounds"
    )
    static_primal_upper = normalize_static_bound(
        self.primal_upper_bounds, np.inf, "primal_upper_bounds"
    )
    if np.any(static_primal_lower > static_primal_upper):
        raise ValueError("primal lower bounds must not exceed upper bounds.")
    return static_primal_lower, static_primal_upper


def _validate_projected_fit(self, epochs, eta, pretrain_epochs, switch_metric, x, y):
    """Normalize observations and validate the existing training-phase options."""
    x_host = np.asarray(x, dtype=np.float64)
    if x_host.ndim != 2 or x_host.shape[1] != self.input_dim:
        raise ValueError(
            f"x must have shape (N, {self.input_dim}); got {x_host.shape}."
        )
    if self.mode in {"supervised", "inverse"}:
        if y is None:
            raise ValueError(f"{self.mode} mode requires y data.")
        y_host = np.asarray(y, dtype=np.float64)
        output_dim = len(self.prepared_projection.compiled.variable_names)
        if y_host.shape != (x_host.shape[0], output_dim):
            raise ValueError(
                f"y must have shape {(x_host.shape[0], output_dim)}; got {y_host.shape}."
            )
    else:
        y_host = None
    if int(epochs) < 1:
        raise ValueError("epochs must be positive.")
    if int(pretrain_epochs) < 0:
        raise ValueError("pretrain_epochs must be nonnegative.")
    if eta is not None and (not np.isfinite(float(eta)) or float(eta) < 0.0):
        raise ValueError("eta must be finite and nonnegative when supplied.")
    allowed_switch_metrics = {"raw_mse", "task_loss", "loss"}
    if switch_metric not in allowed_switch_metrics:
        raise ValueError("switch_metric must be one of raw_mse, task_loss, or loss.")

    pretraining_requested = int(pretrain_epochs) > 0 or eta is not None
    if pretraining_requested and self.mode == "optimization":
        raise ValueError(
            "Backbone-only pretraining currently requires supervised or "
            "inverse data. Optimization mode has no raw-label loss."
        )
    if pretraining_requested:
        if self.graphs.raw_train_step is None or self.graphs.raw_eval_step is None:
            raise RuntimeError(
                "Raw pretraining graphs were not compiled during build()."
            )
    return pretraining_requested, x_host, y_host
