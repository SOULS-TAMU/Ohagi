"""Original-coordinate task and constraint metrics."""

import jax
import jax.numpy as jnp
from purin.compilation import compile_problem


def constraint_metrics(problem, context, values):
    """Summarize equality and inequality violations for a batch of primal values."""
    c = compile_problem(problem)
    eq = jax.vmap(c.equalities)(context, values)
    iq = jax.vmap(c.inequalities)(context, values)
    return {
        "equality_violation": jnp.max(abs(eq), axis=1, initial=0.0),
        "inequality_violation": jnp.max(jnp.maximum(iq, 0), axis=1, initial=0.0),
    }


def regression_metrics(prediction, target):
    """Compute scalar prediction-error statistics against reference targets."""
    error = jnp.asarray(prediction) - jnp.asarray(target)
    return {
        "mse": jnp.mean(error**2),
        "mae": jnp.mean(abs(error)),
        "max_error": jnp.max(abs(error)),
    }
