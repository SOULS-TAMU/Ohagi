"""Functional layers checked against explicit algebra, including invalid contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ohagi.nn.layers.core import Identity, Lambda, Elementwise, Dropout, LayerNorm
from ohagi.nn.layers.linear import Dense
from ohagi.nn.layers.combinators import Serial, Residual, Parallel, FanOut, FanInSum
from ohagi.nn.config import MLP, boxed_relu, resolve_activation
from ohagi.nn.outputs import (
    split_raw_heads,
    split_pretraining_outputs,
    split_projection_outputs,
)

KEY = jax.random.PRNGKey(12)
X = jnp.array([[1.0, 2.0, 4.0], [-2.0, 0.0, 3.0]])


@pytest.mark.parametrize("bias", [False, True])
def test_dense_values_and_input_gradient(bias):
    layer = Dense(2, use_bias=bias)
    shape, params = layer.init(KEY, X.shape)
    assert shape == (2, 2)
    expected = X @ params["kernel"] + (params["bias"] if bias else 0)
    np.testing.assert_allclose(layer.apply(params, X), expected)
    np.testing.assert_allclose(
        jax.grad(lambda x: layer.apply(params, x).sum())(X),
        jnp.broadcast_to(params["kernel"].sum(1), X.shape),
    )
    with pytest.raises(ValueError):
        layer.init(KEY, ())
    with pytest.raises(ValueError):
        layer.init(KEY, (2, 0))


def test_composition_shapes_and_values():
    identity = Identity()
    assert identity.init(KEY, X.shape) == (X.shape, ())
    assert identity.apply((), X) is X
    empty = Serial()
    assert empty.init(KEY, X.shape) == (X.shape, ())
    np.testing.assert_array_equal(empty.apply((), X), X)
    twice = Lambda(lambda x: 2 * x)
    layer = Serial(Residual(twice), Elementwise(jnp.square))
    shape, params = layer.init(KEY, X.shape)
    assert shape == X.shape
    np.testing.assert_array_equal(layer.apply(params, X, training=True), 9 * X**2)
    parallel = Parallel(identity, twice)
    shapes, params = parallel.init(KEY, X.shape)
    assert shapes == (X.shape, X.shape)
    branches = parallel.apply(params, X)
    np.testing.assert_array_equal(branches[1], 2 * X)
    fan = FanOut(3)
    shapes, params = fan.init(KEY, X.shape)
    add = FanInSum()
    assert add.init(KEY, shapes) == (X.shape, ())
    np.testing.assert_array_equal(add.apply((), fan.apply(params, X)), 3 * X)
    kwargs = Lambda(lambda x, scale=1: scale * x)
    np.testing.assert_array_equal(kwargs.apply((), X, scale=4), 4 * X)
    with pytest.raises(ValueError):
        Residual(Dense(2)).init(KEY, X.shape)
    with pytest.raises(ValueError):
        add.init(KEY, ())
    with pytest.raises(ValueError):
        add.init(KEY, ((2, 3), (2, 4)))
    with pytest.raises(ValueError):
        layer.apply((), X)


def test_dropout_and_normalization():
    layer = Dropout(0.5)
    assert layer.init(KEY, X.shape) == (X.shape, ())
    np.testing.assert_array_equal(layer.apply((), X), X)
    np.testing.assert_array_equal(Dropout(0).apply((), X, training=True), X)
    with pytest.raises(ValueError):
        layer.apply((), X, training=True)
    expected = jnp.where(jax.random.bernoulli(KEY, 0.5, X.shape), 2 * X, 0)
    np.testing.assert_array_equal(layer.apply((), X, training=True, rng=KEY), expected)
    norm = LayerNorm(epsilon=0.01)
    shape, params = norm.init(KEY, X.shape)
    assert shape == X.shape
    expected = (np.asarray(X) - np.mean(X, axis=-1, keepdims=True)) / np.sqrt(
        np.var(X, axis=-1, keepdims=True) + 0.01
    )
    np.testing.assert_allclose(norm.apply(params, X), expected)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Dense(0),
        lambda: Lambda(3),
        lambda: FanOut(0),
        lambda: Dropout(-1),
        lambda: Dropout(1),
        lambda: LayerNorm(epsilon=0),
    ],
)
def test_invalid_layers(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


@pytest.mark.parametrize(
    "name", ["tanh", "relu", "sigmoid", "softplus", "gelu", "silu", "swish"]
)
def test_activation_configuration(name):
    reference = getattr(jax.nn, "silu" if name == "swish" else name)
    np.testing.assert_allclose(resolve_activation(name)(X), reference(X))
    config = MLP(hidden_sizes=(3,), activation=name, predict_duals=True)
    assert MLP.from_dict(config.to_dict()) == config
    assert config.output_size(n_y=2, n_eq=1, n_ineq=3) == 6
    assert config.build() is not None


def test_mlp_validation_and_boxed_heads():
    assert resolve_activation(jnp.sin) is jnp.sin
    for kwargs in [
        dict(backend="torch"),
        dict(hidden_sizes=()),
        dict(hidden_sizes=(0,)),
        dict(inequality_dual_floor=-1),
        dict(activation="bad"),
    ]:
        with pytest.raises(ValueError):
            MLP(**kwargs)
    with pytest.raises(ValueError):
        resolve_activation("bad")
    assert MLP(predict_duals=False).output_size(n_y=2, n_eq=1, n_ineq=3) == 2
    values = jnp.array([-2.0, 0.4, 2.0, 3.0])
    lo = jnp.array([0.0, -jnp.inf, 0.0, -jnp.inf])
    hi = jnp.array([1.0, 1.0, jnp.inf, jnp.inf])
    np.testing.assert_allclose(boxed_relu(values, lo, hi), [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(boxed_relu(values, 1.0), jnp.clip(values, 0, 1))
    assert jnp.all(
        jnp.isfinite(jax.grad(lambda v: boxed_relu(v, lo, hi).sum())(values))
    )
    values = jnp.array([[-2.0, 3.0, -0.5, 2.0]])
    sizes = dict(n_y=2, n_eq=1, n_ineq=1)
    for predict in [False, True]:
        y, eq, iq = split_raw_heads(values, **sizes, predict_duals=predict)
        np.testing.assert_array_equal(y, values[:, :2])
        np.testing.assert_array_equal(eq, values[:, 2:3] if predict else [[0.0]])
        y, eq, iq = split_pretraining_outputs(
            values,
            **sizes,
            predict_duals=predict,
            dual_pretrain_gamma=0.1,
            primal_lower_bounds=0.0,
            primal_upper_bounds=1.0,
        )
        np.testing.assert_array_equal(y, [[0.0, 1.0]])
        np.testing.assert_allclose(iq, [[0.1 if predict else 0.0]])
        y, eq, iq = split_projection_outputs(
            values, **sizes, predict_duals=predict, inequality_dual_floor=0.01
        )
        np.testing.assert_allclose(iq, [[2.01 if predict else 0.0]])
    y, _, _ = split_projection_outputs(
        values,
        **sizes,
        predict_duals=True,
        inequality_dual_floor=0,
        primal_transform_fn=lambda y, p: y + p,
        problem_parameters=1.0,
    )
    np.testing.assert_array_equal(y, values[:, :2] + 1)
