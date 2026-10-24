"""Optional adapters and exact checkpoint-resume consistency."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ohagi import Adam
from ohagi.nn.adapters.flax_functional import FlaxFunctionalBackbone
from ohagi.io.orbax import load_orbax, save_orbax
from ohagi.training.steps import apply_updates


def test_custom_flax_backbone_matches_linen():
    nn = pytest.importorskip("flax.linen")

    class Network(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(2, dtype=x.dtype, param_dtype=x.dtype)(x)

    module = Network()
    backbone = FlaxFunctionalBackbone(module)
    parameters = backbone.init(jax.random.PRNGKey(4), 3, 2)
    inputs = jnp.ones((1, 3))
    np.testing.assert_array_equal(
        backbone.apply(parameters, inputs), module.apply({"params": parameters}, inputs)
    )


def test_adam_matches_optax():
    optax = pytest.importorskip("optax")
    parameters = {"x": jnp.array([2.0, -1.0])}
    own = Adam(learning_rate=1e-3)
    other = optax.adam(1e-3)
    own_state, other_state = own.init(parameters), other.init(parameters)
    gradient = {"x": jnp.array([0.3, -0.2])}
    for _ in range(5):
        ours, own_state = own.update(gradient, own_state, parameters)
        theirs, other_state = other.update(gradient, other_state, parameters)
        np.testing.assert_allclose(ours["x"], theirs["x"], atol=1e-12, rtol=1e-10)


def test_orbax_resumed_training_matches_uninterrupted(tmp_path):
    pytest.importorskip("orbax.checkpoint")
    optimizer = Adam(learning_rate=0.01)
    parameters = {"x": jnp.array([2.0, -1.0])}
    initial = {
        "params": parameters,
        "optimizer": optimizer.init(parameters),
        "key": jax.random.key(7),
    }

    def advance(tree, steps):
        for _ in range(steps):
            gradients = jax.grad(lambda p: jnp.sum(p["x"] ** 2))(tree["params"])
            updates, state = optimizer.update(
                gradients, tree["optimizer"], tree["params"]
            )
            tree = {
                "params": apply_updates(tree["params"], updates),
                "optimizer": state,
                "key": tree["key"],
            }
        return tree

    uninterrupted = advance(initial, 6)
    partial = advance(initial, 3)
    save_orbax(tmp_path / "checkpoint", partial, metadata={"epoch": 3})
    restored, metadata = load_orbax(tmp_path / "checkpoint", partial)
    assert metadata == {"epoch": 3}
    resumed = advance(restored, 3)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(resumed), jax.tree_util.tree_leaves(uninterrupted)
    ):
        if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
            actual, expected = (
                jax.random.key_data(actual),
                jax.random.key_data(expected),
            )
        np.testing.assert_array_equal(actual, expected)
