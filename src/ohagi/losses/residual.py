"""Residual losses independent of a solver's internal scaled residuals."""

import jax.numpy as jnp
from .core import masked_mean


def residual_mse(equalities, inequalities, mask=None):
    """Average squared residuals, respecting an optional observation mask."""
    blocks = jnp.concatenate((equalities, jnp.maximum(inequalities, 0)), axis=-1)
    if blocks.shape[-1] == 0:
        return jnp.asarray(0.0, blocks.dtype)
    return masked_mean(jnp.mean(blocks**2, axis=-1), mask)
