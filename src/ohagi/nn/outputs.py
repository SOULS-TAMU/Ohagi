"""Shared primal and dual output transformations for both training APIs."""

import jax
import jax.numpy as jnp
from ohagi.nn.config import boxed_relu


def split_raw_heads(
    network_values, *, n_y: int, n_eq: int, n_ineq: int, predict_duals: bool
):
    """Split a raw network vector into primal/equality/inequality heads."""

    values = jnp.asarray(network_values)
    y_hat = values[..., :n_y]
    shape = values.shape[:-1]
    if not predict_duals:
        lambda_raw = jnp.zeros((*shape, n_eq), dtype=values.dtype)
        mu_raw = jnp.zeros((*shape, n_ineq), dtype=values.dtype)
        return y_hat, lambda_raw, mu_raw

    equality_end = n_y + n_eq
    lambda_raw = values[..., n_y:equality_end]
    mu_raw = values[..., equality_end : equality_end + n_ineq]
    return y_hat, lambda_raw, mu_raw


def split_pretraining_outputs(
    network_values,
    *,
    n_y: int,
    n_eq: int,
    n_ineq: int,
    predict_duals: bool,
    dual_pretrain_gamma: float,
    primal_lower_bounds=-jnp.inf,
    primal_upper_bounds=jnp.inf,
):
    """Return boxed primal and boxed latent dual outputs for pretraining."""

    y_hat, lambda_raw, mu_raw = split_raw_heads(
        network_values, n_y=n_y, n_eq=n_eq, n_ineq=n_ineq, predict_duals=predict_duals
    )
    y_hat = boxed_relu(y_hat, primal_lower_bounds, primal_upper_bounds)
    if not predict_duals:
        return y_hat, lambda_raw, mu_raw
    return (
        y_hat,
        boxed_relu(lambda_raw, 0.0, dual_pretrain_gamma),
        boxed_relu(mu_raw, 0.0, dual_pretrain_gamma),
    )


def split_projection_outputs(
    network_values,
    *,
    n_y,
    n_eq,
    n_ineq,
    predict_duals,
    inequality_dual_floor,
    primal_lower_bounds=-jnp.inf,
    primal_upper_bounds=jnp.inf,
    primal_transform_fn=None,
    problem_parameters=None,
):
    """Split concatenated predictions into primal and constraint-multiplier heads."""
    y_hat, lambda_raw, mu_raw = split_raw_heads(
        network_values, n_y=n_y, n_eq=n_eq, n_ineq=n_ineq, predict_duals=predict_duals
    )

    if primal_transform_fn is not None:
        y_hat = primal_transform_fn(y_hat, problem_parameters)
    else:
        y_hat = boxed_relu(y_hat, primal_lower_bounds, primal_upper_bounds)

    if not predict_duals:
        return y_hat, lambda_raw, mu_raw

    floor = jnp.asarray(inequality_dual_floor, dtype=jnp.asarray(network_values).dtype)

    return (y_hat, lambda_raw, jax.nn.relu(mu_raw) + floor)
