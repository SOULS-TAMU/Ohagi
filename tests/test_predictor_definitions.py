"""Portable predictor definitions preserve algebra and all supported transforms."""

import copy
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from purin.problems.qp import ParametricQP
from ohagi.serialization.problem import save_problem, load_problem, _load_matrix
from ohagi.serialization.predictor import (
    make_network_apply,
    save_predictor,
    load_predictor,
    _activation,
    _prepare_projection,
    _load_factory,
    _load_transform,
    _projection_config_from_dict,
)


@pytest.mark.parametrize("sparse", [False, True])
def test_problem_roundtrip(tmp_path, sparse):
    convert = csr_matrix if sparse else np.asarray
    problem = ParametricQP(
        P=convert(np.diag([2.0, 3.0])),
        q0=np.array([1.0, 2.0]),
        q_parameter_map=convert([[1.0], [2.0]]),
        A_eq=convert([[1.0, 1.0]]),
        b_eq0=[3.0],
        b_eq_parameter_map=convert([[1.0]]),
        A_ineq=convert([[-1.0, 0.0]]),
        b_ineq0=[0.0],
        b_ineq_parameter_map=convert([[0.0]]),
    )
    file = tmp_path / "problem.npz"
    manifest = save_problem(file, problem)
    restored = load_problem(file, manifest)
    for field in manifest["fields"]:
        actual = getattr(restored, field)
        expected = getattr(problem, field)
        np.testing.assert_array_equal(
            actual.toarray() if hasattr(actual, "toarray") else actual,
            expected.toarray() if hasattr(expected, "toarray") else expected,
        )
    with pytest.raises(TypeError):
        save_problem(file, object())
    for key in ["format", "type"]:
        bad = copy.deepcopy(manifest)
        bad[key] = "wrong"
        with pytest.raises(ValueError):
            load_problem(file, bad)
    with pytest.raises(ValueError):
        _load_matrix({"kind": "bad"}, {})
    with pytest.raises(ValueError):
        _load_matrix({"kind": "dense", "key": "x", "shape": [2]}, {"x": np.ones(3)})


@pytest.mark.parametrize(
    "activation", ["relu", "tanh", "gelu", "sigmoid", "softplus", "silu", "swish"]
)
@pytest.mark.parametrize("layout", ["legacy", "kernel", "nested", "backbone"])
def test_network_architecture_reconstruction(activation, layout):
    first = {"W": jnp.array([[1.0, -1.0]]), "b": jnp.array([0.2, 0.3])}
    second = {"W": jnp.array([[2.0], [3.0]]), "b": jnp.array([0.4])}
    params = (first, second)
    spec = {"activation": activation}
    if layout == "kernel":
        params = tuple(
            {"kernel": layer["W"], **({"bias": layer["b"]} if i == 0 else {})}
            for i, layer in enumerate(params)
        )
    elif layout == "nested":
        params = {"model": params}
        spec["parameter_path"] = ["model"]
    elif layout == "backbone":
        params = {"backbone": params}
    else:
        spec["parameter_format"] = "legacy"
    value = make_network_apply(spec)(params, jnp.array([[0.8]]))
    expected = _activation(activation)(jnp.array([[1.0, -0.5]])) @ second["W"] + (
        0 if layout == "kernel" else 0.4
    )
    np.testing.assert_allclose(value, expected)


@pytest.mark.parametrize("transform", ["identity", "standard"])
@pytest.mark.parametrize("projection", ["none", "box", "linear_equality"])
def test_predictor_metadata_roundtrip(tmp_path, transform, projection):
    params = ({"W": jnp.array([[2.0, -1.0]]), "b": jnp.array([0.5, 0.5])},)
    transform_spec = {"type": transform}
    if transform == "standard":
        transform_spec.update(mean=np.array([1.0]), scale=np.array([2.0]))
    spec = {"type": projection}
    if projection == "box":
        spec.update(lower=np.array([0.0, 0.0]), upper=np.array([1.0, 1.0]))
    if projection == "linear_equality":
        spec.update(A=np.array([[1.0, 1.0]]), b=np.array([1.0]))
    model = save_predictor(
        tmp_path,
        network_spec={"type": "mlp"},
        params=params,
        input_size=1,
        output_size=2,
        batch_size=2,
        projection_spec=spec,
        input_transform=transform_spec,
    )
    expected = (
        np.array([4.5, -1.5]) if transform == "identity" else np.array([1.5, 0.0])
    )
    if projection == "box":
        expected = np.clip(expected, 0, 1)
    if projection == "linear_equality":
        expected = expected - (expected.sum() - 1) / 2
    for predictor in [model, load_predictor(tmp_path / "metadata.json")]:
        np.testing.assert_allclose(predictor(np.array([2.0])), expected, atol=1e-10)


def test_rejects_unknown_definition(tmp_path):
    for spec in [{"type": "bad"}, {"activation": "bad"}]:
        with pytest.raises(ValueError):
            make_network_apply(spec)
    for spec in [
        {"type": "bad"},
        {
            "type": "standard",
            "mean": [0.0],
            "scale": [1.0],
            "arrays_file": "missing.npz",
        },
    ]:
        with pytest.raises((ValueError, FileNotFoundError)):
            _load_transform(tmp_path, spec)
    for problem in [None, object()]:
        with pytest.raises(ValueError):
            _prepare_projection(
                {"type": "bad"},
                problem=problem,
                root=tmp_path,
                batch_size=1,
                dtype=jnp.float64,
                backend="cpu",
            )
    with pytest.raises(ValueError):
        _load_factory("not-a-path")
    with pytest.raises(TypeError):
        _load_factory("math:pi")
    assert _load_factory(jnp.sin) is jnp.sin
    assert _load_factory("jax.numpy:sin") is jnp.sin
    config = _projection_config_from_dict(
        {
            "mu": 0.01,
            "linear_solver": {"tolerance": 1e-8},
            "implicit_gradient": {"policy": "strict"},
        }
    )
    assert config.mu == 0.01 and config.linear_solver.tolerance == 1e-8
