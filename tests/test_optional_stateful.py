"""Optional stateful Flax and Linen adapters preserve shared training semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ohagi.nn.adapters.flax import FlaxLinenBackbone
from ohagi.training.stateful import build_stateful_steps, warmup_stateful


@pytest.mark.parametrize("heads", [(0, 0), (1, 1)])
def test_projected_linen_heads(heads):
    nn = pytest.importorskip("flax.linen")
    from ohagi.nn.adapters.projected_mlp import ProjectedMLP
    from ohagi.nn.adapters.linen import mlp_forward

    module = ProjectedMLP((3,), 2, *heads)
    x = jnp.ones((2, 2))
    variables = module.init(jax.random.PRNGKey(0), x)
    outputs = module.apply(variables, x)
    assert tuple(o.shape for o in outputs) == ((2, 2), (2, heads[0]), (2, heads[1]))

    class Combined(nn.Module):
        @nn.compact
        def __call__(self, x):
            return mlp_forward(
                x,
                hidden_sizes=(3,),
                primal_size=2,
                equality_size=heads[0],
                inequality_size=heads[1],
                name_prefix="layer",
                dtype=x.dtype,
                param_dtype=x.dtype,
            )

    combined = Combined()
    values = combined.init(jax.random.PRNGKey(0), x)
    assert combined.apply(values, x).shape == (2, 2 + sum(heads))


@pytest.mark.parametrize("mode_keyword", [None, "training"])
def test_linen_mutable_state(mode_keyword):
    nn = pytest.importorskip("flax.linen")

    class Network(nn.Module):
        @nn.compact
        def __call__(self, x, training=False):
            x = nn.Dense(2, dtype=x.dtype, param_dtype=x.dtype)(x)
            return nn.BatchNorm(
                use_running_average=not training, dtype=x.dtype, param_dtype=x.dtype
            )(x)

    module = Network()
    adapter = FlaxLinenBackbone(module, mode_keyword=mode_keyword)
    params, state = adapter.init(jax.random.PRNGKey(1), 2, 2, jnp.float64)
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    actual, new_state = adapter.apply(
        params, state, x, key=jax.random.PRNGKey(2), training=True
    )
    assert actual.shape == (2, 2)
    assert jnp.all(jnp.isfinite(actual))
    prediction, unchanged = adapter.apply(params, new_state, x)
    np.testing.assert_allclose(
        prediction, module.apply({"params": params, **new_state}, x)
    )
    assert unchanged is new_state


@pytest.mark.parametrize("include", [False, True])
def test_stateful_training_and_warmup(include):
    nn = pytest.importorskip("flax.linen")
    optax = pytest.importorskip("optax")
    module = nn.Dense(1, dtype=jnp.float64, param_dtype=jnp.float64)

    def loss(params, x):
        value = jnp.mean((module.apply({"params": params}, x) - 1) ** 2)
        return value, {"loss": value}

    init, train, evaluate = build_stateful_steps(
        loss,
        module,
        batch_size=2,
        input_size=1,
        dtype=jnp.float64,
        learning_rate=0.01,
        include_update_accepted=include,
        optimizer=optax.adam(0.01) if include else None,
    )
    state = init(jax.random.PRNGKey(5))
    before = evaluate(state.params, jnp.ones((2, 1)))["loss"]
    updated, metrics = train(state, jnp.ones((2, 1)))
    assert int(updated.step) == 1
    assert ("update_accepted" in metrics) == include
    assert evaluate(updated.params, jnp.ones((2, 1)))["loss"] < before
    result = warmup_stateful(
        state=updated,
        train_step_fn=train,
        eval_step_fn=evaluate,
        input_size=1,
        batch_size=2,
        seed=2,
        device=jax.devices("cpu")[0],
        commit_update=False,
    )
    assert int(result.step) == 1


def test_flax_mlp_matches_functional_dual_initialization():
    pytest.importorskip("flax.linen")
    from ohagi.nn.config import MLP
    from ohagi.nn.mlp import MLPBackbone

    key = jax.random.PRNGKey(5)
    x = jnp.array([[1.0, 2.0]])
    flax = MLP(hidden_sizes=(3,), backend="flax").build()
    functional = MLPBackbone((3,), jnp.tanh)
    kwargs = dict(n_primal_outputs=2, n_equality_duals=1, n_inequality_duals=1)
    a = flax.init(key, 2, 4, **kwargs)
    b = functional.init(key, 2, 4, **kwargs)
    np.testing.assert_allclose(flax.apply(a, x), functional.apply(b, x), atol=1e-12)
    assert MLP(backend="flax").to_dict()["backend"] == "flax"
