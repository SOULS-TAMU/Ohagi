"""Pure optimizer helpers and JIT-compatible train/evaluation step builders."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import jax
import jax.numpy as jnp
from ohagi._utils.tree import tree_all_finite
from ohagi._utils.tree import tree_select
from ohagi.training.state import AdamState

PyTree = Any


@dataclass(frozen=True)
class Adam:
    """Small dependency-free Adam optimizer with an Optax-like interface."""

    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def init(self, parameters: PyTree) -> AdamState:
        """Initialize zero moment pytrees and a scalar Adam step counter."""
        zeros = jax.tree_util.tree_map(jnp.zeros_like, parameters)
        return AdamState(
            count=jnp.asarray(0, dtype=jnp.int32),
            first_moment=zeros,
            second_moment=zeros,
        )

    def update(
        self, gradients: PyTree, state: AdamState, parameters: PyTree | None = None
    ) -> tuple[PyTree, AdamState]:
        """Compute Adam parameter updates and the next optimizer state."""
        del parameters
        count = state.count + jnp.asarray(1, dtype=state.count.dtype)
        first = jax.tree_util.tree_map(
            lambda moment, gradient: (
                self.beta1 * moment + (1.0 - self.beta1) * gradient
            ),
            state.first_moment,
            gradients,
        )
        second = jax.tree_util.tree_map(
            lambda moment, gradient: (
                self.beta2 * moment + (1.0 - self.beta2) * jnp.square(gradient)
            ),
            state.second_moment,
            gradients,
        )
        count_float = count.astype(jnp.float64)
        first_scale = 1.0 - self.beta1**count_float
        second_scale = 1.0 - self.beta2**count_float
        updates = jax.tree_util.tree_map(
            lambda first_moment, second_moment: (
                -self.learning_rate
                * (first_moment / first_scale)
                / (jnp.sqrt(second_moment / second_scale) + self.epsilon)
            ),
            first,
            second,
        )
        return updates, AdamState(count, first, second)


def apply_updates(parameters: PyTree, updates: PyTree) -> PyTree:
    """Apply additive optimizer updates to a parameter pytree."""

    return jax.tree_util.tree_map(
        lambda parameter, update: parameter + update, parameters, updates
    )


def copy_tree_to_device(tree: PyTree, device: Any) -> PyTree:
    """Place every array leaf of a pytree on the requested JAX device."""
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(jnp.array(value), device), tree
    )


def call_optimizer_update(optimizer, gradients, state, parameters):
    """Call an optimizer update while accommodating its supported signature."""
    try:
        return optimizer.update(gradients, state, parameters)
    except TypeError:
        return optimizer.update(gradients, state)


def per_sample(values: jax.Array, batch_size: int) -> jax.Array:
    """Reduce trailing value dimensions to one scalar per batch instance."""
    values = jnp.asarray(values, dtype=jnp.float64)
    if values.ndim == 0:
        return jnp.broadcast_to(values, (batch_size,))
    if values.shape[0] != batch_size:
        raise ValueError(
            "A loss/objective function must return a scalar or an array whose "
            f"first dimension is the fixed batch size {batch_size}."
        )
    if values.ndim == 1:
        return values
    return jnp.mean(values, axis=tuple(range(1, values.ndim)))


def masked_mean(values: jax.Array, valid_mask: jax.Array) -> jax.Array:
    """Average selected batch values without counting masked observations."""
    weights = valid_mask.astype(values.dtype)
    return jnp.sum(weights * values) / jnp.maximum(jnp.sum(weights), 1.0)


def optimizer_step(
    parameters,
    optimizer_state,
    gradients,
    optimizer,
    *,
    valid,
    update_fn=apply_updates,
    reject_invalid=True,
    sanitize_gradients=False,
    check_candidate_finite=False,
    check_optimizer_state=True,
):
    """Single optimizer transition used by all model/trainer interfaces."""
    if sanitize_gradients:
        gradients = jax.tree_util.tree_map(
            lambda value: jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
            gradients,
        )
    updates, candidate_state = call_optimizer_update(
        optimizer, gradients, optimizer_state, parameters
    )
    candidate_parameters = update_fn(parameters, updates)
    if check_candidate_finite:
        valid = valid & tree_all_finite(candidate_parameters)
        if check_optimizer_state:
            valid = valid & tree_all_finite(candidate_state)
    if reject_invalid:
        return (
            tree_select(valid, candidate_parameters, parameters),
            tree_select(valid, candidate_state, optimizer_state),
            valid,
        )
    return candidate_parameters, candidate_state, valid


@dataclass(frozen=True)
class StepPolicy:
    """Selection of gradient and invalid-update behavior for a pure training step."""

    reject_invalid: bool = True
    sanitize_gradients: bool = False
    check_candidate_finite: bool = False
    check_optimizer_state: bool = True


def parameter_train_step(
    loss_fn,
    optimizer,
    *,
    policy=StepPolicy(),
    eligible=None,
    format_metrics=None,
    optimizer_apply_updates=apply_updates,
):
    """Differentiate and update once; task adapters supply objective and policy."""
    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    def step(parameters, optimizer_state, *batch):
        (loss, metrics), gradients = value_and_grad(parameters, *batch)
        valid = jnp.isfinite(loss) & tree_all_finite(gradients)
        if eligible is not None:
            valid = valid & eligible(metrics)
        new_parameters, new_state, valid = optimizer_step(
            parameters,
            optimizer_state,
            gradients,
            optimizer,
            valid=valid,
            update_fn=optimizer_apply_updates,
            reject_invalid=policy.reject_invalid,
            sanitize_gradients=policy.sanitize_gradients,
            check_candidate_finite=policy.check_candidate_finite,
            check_optimizer_state=policy.check_optimizer_state,
        )
        if format_metrics is None:
            metrics = {**metrics, "loss": loss, "update_accepted": valid}
        else:
            metrics = format_metrics(loss, metrics, valid)
        return new_parameters, new_state, metrics

    return step


def build_train_step(
    loss_fn,
    optimizer,
    *,
    invalid_batch_policy="skip",
    optimizer_apply_updates=apply_updates,
):
    """Construct the shared parameter-and-optimizer transition for one batch."""
    if invalid_batch_policy not in {"skip", "use_valid", "propagate"}:
        raise ValueError("invalid_batch_policy must be skip, use_valid, or propagate.")
    return parameter_train_step(
        loss_fn,
        optimizer,
        policy=StepPolicy(reject_invalid=invalid_batch_policy == "skip"),
        eligible=lambda metrics: jnp.asarray(metrics.get("gradient_valid", True)),
        optimizer_apply_updates=optimizer_apply_updates,
    )


def make_train_step(loss_fn, optimizer, *, invalid_batch_policy: str = "skip"):
    """Historical functional API over the shared train-step builder."""
    return build_train_step(
        loss_fn, optimizer, invalid_batch_policy=invalid_batch_policy
    )


def make_eval_step(loss_fn):
    """Build one pure evaluation step; callers decide when to apply ``jax.jit``."""

    def step(parameters, batch):
        loss, metrics = loss_fn(parameters, batch)
        return {**metrics, "loss": loss}

    return step
