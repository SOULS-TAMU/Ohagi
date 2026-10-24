"""Functional backbones preserve dtype, JIT, state and batch contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ohagi.nn import FunctionalBackbone


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_backbone_runtime_configuration(dtype, compiled, batch_size):
    backbone = FunctionalBackbone(
        lambda key, input_dim, output_dim, dtype: jnp.ones(
            (input_dim, output_dim), dtype
        ),
        lambda parameters, inputs: inputs @ parameters,
    )
    parameters, state = backbone.init(jax.random.PRNGKey(0), 2, 3, dtype)
    inputs = jnp.tile(jnp.array([[1.0, 1.5]], dtype=dtype), (batch_size, 1))
    if compiled:
        actual, next_state = jax.jit(backbone.apply)(parameters, state, inputs)
    else:
        with jax.disable_jit():
            actual, next_state = backbone.apply(parameters, state, inputs)
    assert actual.dtype == dtype
    assert next_state == state == {}
    np.testing.assert_allclose(actual, np.full((batch_size, 3), 2.5), atol=1e-6, rtol=0)
