"""Shared objective, multiplier-penalty and primal/dual consistency loss."""

import jax.numpy as jnp
from .core import masked_mean, mse


def modified_lagrangian_loss(
    objective, equalities, inequalities, *, alpha_consistency=10.0
):
    """Build the objective, residual-multiplier and projection-consistency loss."""
    if alpha_consistency < 0:
        raise ValueError("alpha_consistency must be nonnegative")

    def loss(outputs, batch):
        context, y = batch["theta"], outputs["projected"]
        mask = batch.get("mask")
        obj = masked_mean(objective(context, y), mask)
        eq, iq = equalities(context, y), inequalities(context, y)
        eq_penalty, eq_violation = jnp.asarray(0.0, y.dtype), jnp.asarray(0.0, y.dtype)
        iq_penalty, iq_violation = jnp.asarray(0.0, y.dtype), jnp.asarray(0.0, y.dtype)
        if eq.shape[1]:
            eq_penalty = masked_mean(
                jnp.square(jnp.sum(outputs["lambda_projected"] * eq, axis=1)), mask
            )
            eq_violation = (
                jnp.mean(jnp.abs(eq))
                if mask is None
                else masked_mean(jnp.mean(jnp.abs(eq), axis=1), mask)
            )
        if iq.shape[1]:
            positive = jnp.maximum(iq, 0.0)
            iq_penalty = masked_mean(
                jnp.sum(outputs["mu_projected"] * positive, axis=1), mask
            )
            iq_violation = (
                jnp.mean(positive)
                if mask is None
                else masked_mean(jnp.mean(positive, axis=1), mask)
            )

        def consistency(raw, corrected):
            if raw.shape[1] == 0:
                return jnp.asarray(0.0, y.dtype)
            return (
                jnp.mean((raw - corrected) ** 2)
                if mask is None
                else mse(raw, corrected, mask)
            )

        primal = consistency(outputs["raw"], y)
        equality = consistency(outputs["lambda_raw"], outputs["lambda_projected"])
        inequality = consistency(outputs["mu_raw"], outputs["mu_projected"])
        cons = primal + equality + inequality
        total = (
            obj
            + eq_penalty
            + iq_penalty
            + jnp.asarray(alpha_consistency, y.dtype) * cons
        )
        return total, {
            "loss": total,
            "obj": obj,
            "consistency": cons,
            "eq_violation": eq_violation,
            "ineq_violation": iq_violation,
            "eq_penalty": eq_penalty,
            "ineq_penalty": iq_penalty,
            "mse_y": primal,
            "mse_lam": equality,
            "mse_mu": inequality,
        }

    return loss
