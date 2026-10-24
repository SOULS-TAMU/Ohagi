"""One optional Linen MLP construction for combined or separate output heads."""

import jax.numpy as jnp


def mlp_forward(
    inputs,
    *,
    hidden_sizes,
    primal_size,
    activation=jnp.tanh,
    equality_size=0,
    inequality_size=0,
    separate_heads=False,
    name_prefix=None,
    dtype=None,
    param_dtype=None,
):
    """Called inside a Linen compact scope; backend/defaults are explicit."""
    from flax import linen as nn

    def dense(values, width, index):
        options = {}
        if name_prefix is not None:
            options["name"] = f"{name_prefix}{index}"
        if dtype is not None:
            options["dtype"] = dtype
        if param_dtype is not None:
            options["param_dtype"] = param_dtype
        return nn.Dense(width, **options)(values)

    values = inputs
    for index, width in enumerate(hidden_sizes):
        values = activation(dense(values, width, index))
    index = len(hidden_sizes)
    if not separate_heads:
        return dense(values, primal_size + equality_size + inequality_size, index)
    primal = dense(values, primal_size, index)
    index += 1
    equality = (
        dense(values, equality_size, index)
        if equality_size
        else jnp.zeros((inputs.shape[0], 0), dtype=inputs.dtype)
    )
    index += bool(equality_size)
    inequality = (
        dense(values, inequality_size, index)
        if inequality_size
        else jnp.zeros((inputs.shape[0], 0), dtype=inputs.dtype)
    )
    return primal, equality, inequality
