"""Pure JAX loss functions for projected-neural training."""

from __future__ import annotations
from collections.abc import Mapping
import jax
import jax.numpy as jnp


def masked_mean(values, mask=None):
    """Average values, optionally excluding invalid samples."""
    values = jnp.asarray(values)
    if mask is None:
        return jnp.mean(values)
    weights = jnp.asarray(mask, dtype=values.dtype)
    return jnp.sum(values * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def mse(prediction, target, mask=None):
    """Mean squared error reduced per sample before masking."""
    prediction = jnp.asarray(prediction)
    target = jnp.asarray(target)
    per_sample = jnp.mean(jnp.square(prediction - target), axis=-1)
    return masked_mean(per_sample, mask)


def raw_output_mse(raw, target, mask=None):
    """Measure mean squared error on the unprojected network outputs."""
    return mse(raw, target, mask)


def projected_output_mse(projected, target, mask=None):
    """Measure mean squared error on the projected network outputs."""
    return mse(projected, target, mask)


def projection_consistency(raw, projected, mask=None):
    """Penalize disagreement between raw and projected primal outputs."""
    return mse(projected, raw, mask)


# Historical short name.
consistency = projection_consistency


def supervised_mse(
    *,
    prediction_key: str = "projected",
    target_key: str = "solution",
    consistency_weight: float = 0.0,
):
    """Build a supervised projected-output loss with optional consistency."""
    weight = float(consistency_weight)

    def loss(outputs: Mapping, batch: Mapping):
        mask = batch.get("mask")
        task = projected_output_mse(outputs[prediction_key], batch[target_key], mask)
        consistency_value = (
            projection_consistency(outputs["raw"], outputs["projected"], mask)
            if weight
            else jnp.asarray(0.0, task.dtype)
        )
        return task + weight * consistency_value, {
            "task_loss": task,
            "consistency_loss": consistency_value,
        }

    return loss


def problem_objective(objective):
    """Build a loss that evaluates the supplied problem objective on projected outputs."""
    return objective_loss(objective)


def objective_loss(objective, *, respect_mask=True):
    """Build a loss from a per-sample optimization objective."""
    evaluate_batch = jax.vmap(objective)

    def loss(outputs: Mapping, batch: Mapping):
        values = evaluate_batch(batch["theta"], outputs["projected"])
        value = masked_mean(values, batch.get("mask") if respect_mask else None)
        return value, {"objective": value}

    return loss


def inverse_observation(observation_fn=lambda y: y):
    """Build an inverse-model observation loss."""

    def loss(outputs: Mapping, batch: Mapping):
        value = mse(
            observation_fn(outputs["projected"]),
            batch["observations"],
            batch.get("mask"),
        )
        return value, {"observation_loss": value}

    return loss


def weighted_sum(terms):
    """Build a loss combining named component losses with configured scalar weights."""
    return combine_losses(terms)


def combine_losses(terms, *, initial_dtype="outputs"):
    """Combine named ``(weight, loss_function)`` terms."""
    terms = dict(terms)

    def loss(outputs: Mapping, batch: Mapping):
        dtype = (
            jnp.asarray(next(iter(outputs.values()))).dtype
            if initial_dtype == "outputs"
            else initial_dtype
        )
        total = jnp.asarray(0.0, dtype=dtype)
        metrics = {}
        for name, (weight, function) in terms.items():
            value, auxiliary = function(outputs, batch)
            total = total + float(weight) * value
            metrics[name] = value
            metrics.update({f"{name}/{key}": item for key, item in auxiliary.items()})
        return total, metrics

    return loss


def latent_dual_consistency(predicted, target, valid_mask=None, sample_mask=None):
    """MSE to a stop-gradient latent dual target on valid solver rows only."""
    predicted = jnp.asarray(predicted)
    target = jax.lax.stop_gradient(jnp.asarray(target))
    if predicted.shape[-1] == 0:
        return jnp.asarray(0.0, dtype=predicted.dtype)
    per_sample = jnp.mean(jnp.square(predicted - target), axis=-1)
    mask = None
    if valid_mask is not None:
        mask = jnp.asarray(valid_mask, dtype=bool)
    if sample_mask is not None:
        sample_mask = jnp.asarray(sample_mask, dtype=bool)
        mask = sample_mask if mask is None else (mask & sample_mask)
    return masked_mean(per_sample, mask)


def configured_training_loss(config, *, objective=None, observation_indices=None):
    """Build the loss selected by :class:`kkthn.losses.Loss`.

    Expected output keys are ``raw``, ``projected``, ``lambda_raw``,
    ``lambda_projected``, ``mu_raw``, ``mu_projected`` and ``dual_valid``.
    """
    kind = str(config.kind).strip().lower()
    observation_indices = (
        None if observation_indices is None else jnp.asarray(observation_indices)
    )

    def loss(outputs: Mapping, batch: Mapping):
        sample_mask = batch.get("mask")
        projection_valid = outputs.get("dual_valid")
        projection_mask = sample_mask
        if projection_valid is not None:
            projection_valid = jnp.asarray(projection_valid, dtype=bool)
            if sample_mask is None:
                projection_mask = projection_valid
            else:
                projection_mask = (
                    jnp.asarray(sample_mask, dtype=bool) & projection_valid
                )

        if kind == "objective":
            if objective is None:
                raise ValueError("An optimization objective is required.")
            values = jax.vmap(objective)(batch["theta"], outputs["projected"])
            task = masked_mean(values, projection_mask)
        else:
            prediction = outputs["projected"]
            if kind == "observation" and observation_indices is not None:
                prediction = prediction[:, observation_indices]
            task = mse(prediction, batch["solution"], projection_mask)

        primal = projection_consistency(
            outputs["raw"], outputs["projected"], projection_mask
        )
        lambda_value = latent_dual_consistency(
            outputs["lambda_raw"],
            outputs["lambda_projected"],
            projection_valid,
            sample_mask,
        )
        mu_value = latent_dual_consistency(
            outputs["mu_raw"], outputs["mu_projected"], projection_valid, sample_mask
        )
        total = (
            task
            + float(config.primal_consistency_weight) * primal
            + float(config.equality_dual_consistency_weight) * lambda_value
            + float(config.inequality_dual_consistency_weight) * mu_value
        )
        return total, {
            "task_loss": task,
            "primal_consistency_loss": primal,
            "equality_dual_consistency_loss": lambda_value,
            "inequality_dual_consistency_loss": mu_value,
        }

    return loss
