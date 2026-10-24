"""Metrics, normalization, scheduling and history reporting contracts."""

import csv
import jax.numpy as jnp
import numpy as np
import pytest
from ohagi.transforms.scaling import Standardizer, DiagonalCoordinates
from ohagi.training.schedules import ProjectionSchedule
from ohagi.metrics import regression_metrics, constraint_metrics
from ohagi.reporting.history import save_history, plot_history
from ohagi.losses.lagrangian import modified_lagrangian_loss
from purin import Projection


def test_coordinate_scaling_and_standardizer():
    data = jnp.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
    normal = Standardizer.fit(data)
    np.testing.assert_allclose(normal.inverse(normal.transform(data)), data)
    np.testing.assert_allclose(normal.transform(data).mean(0), [0.0, 0.0])
    coordinates = DiagonalCoordinates(jnp.array([2.0, 3.0]), jnp.array([1.0, -1.0]))
    np.testing.assert_allclose(coordinates.physical(coordinates.normalize(data)), data)
    np.testing.assert_allclose(
        coordinates.physical_gradient(jnp.array([2.0, 6.0])), [1.0, 2.0]
    )
    np.testing.assert_allclose(
        coordinates.physical_multipliers(jnp.array([2.0, 3.0]), jnp.array([3.0, 4.0])),
        [6.0, 12.0],
    )
    for scale in [[0.0], [-1.0], [np.inf]]:
        with pytest.raises(ValueError):
            DiagonalCoordinates(scale)
    for bad in [[], np.ones(2), [[np.nan]]]:
        with pytest.raises(ValueError):
            Standardizer.fit(bad)


def test_schedules_and_metrics(tmp_path):
    schedule = ProjectionSchedule(min_pretrain_epochs=2, max_pretrain_epochs=5, eta=0.1)
    assert not schedule.should_attempt(1, 0)
    assert schedule.should_attempt(5)
    assert not schedule.should_attempt(2)
    assert schedule.should_attempt(2, 0.05)
    assert not schedule.should_attempt(2, 0.2)
    for kwargs in [
        dict(min_pretrain_epochs=-1),
        dict(min_pretrain_epochs=5, max_pretrain_epochs=2),
        dict(readiness_fraction=2),
        dict(on_not_ready="bad"),
    ]:
        with pytest.raises(ValueError):
            ProjectionSchedule(**kwargs)
    metrics = regression_metrics(jnp.array([1.0, -2.0]), jnp.zeros(2))
    assert metrics["mse"] == 2.5 and metrics["mae"] == 1.5 and metrics["max_error"] == 2
    problem = Projection()
    p = problem.add_parameter(["b"])
    y = problem.add_variable(["u", "v"])
    problem.objective = 0
    problem.constraints.add(y.u + y.v == p.b, y.u >= 0)
    metrics = constraint_metrics(problem, jnp.array([[1.0]]), jnp.array([[-1.0, 3.0]]))
    np.testing.assert_array_equal(metrics["equality_violation"], [1.0])
    np.testing.assert_array_equal(metrics["inequality_violation"], [1.0])
    history = [dict(epoch=1, loss=2.0), dict(epoch=2, loss=1.0, validation=0.5)]
    file = save_history(tmp_path / "history.csv", history)
    with file.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["validation"] == "0.5" and rows[0]["validation"] == ""
    with pytest.raises(ValueError):
        save_history(file, [])
    plt = pytest.importorskip("matplotlib.pyplot")
    ax = plot_history(history)
    assert len(ax.lines) == 1
    plot_history(history, ax=ax)
    assert len(ax.lines) == 2
    plt.close("all")


@pytest.mark.parametrize("mask", [None, jnp.array([True, False])])
@pytest.mark.parametrize("constraints", [False, True])
def test_modified_lagrangian_exact_components(mask, constraints):
    y = jnp.ones((2, 1))
    empty = jnp.empty((2, 0))
    dual = y if constraints else empty
    out = dict(
        raw=2 * y,
        projected=y,
        lambda_raw=2 * dual,
        lambda_projected=dual,
        mu_raw=2 * dual,
        mu_projected=dual,
    )
    batch = {"theta": y}
    if mask is not None:
        batch["mask"] = mask
    loss = modified_lagrangian_loss(
        lambda p, y: y[:, 0],
        lambda p, y: 2 * y if constraints else empty,
        lambda p, y: 3 * y if constraints else empty,
        alpha_consistency=2,
    )
    value, metrics = loss(out, batch)
    assert value == (14 if constraints else 3)
    assert metrics["eq_penalty"] == (4 if constraints else 0)
    assert metrics["ineq_penalty"] == (3 if constraints else 0)
    assert metrics["consistency"] == (3 if constraints else 1)
    with pytest.raises(ValueError):
        modified_lagrangian_loss(None, None, None, alpha_consistency=-1)


def test_model_types_losses_and_configuration():
    from ohagi.model_types import ModelType
    from ohagi.losses.residual import residual_mse
    from ohagi.training.config import TrainingConfig, DualPretrainConfig

    for value, expected in [
        ("simulation", ModelType.SURROGATE),
        ("estimate", ModelType.PARAMETER_ESTIMATION),
        ("optimize", ModelType.OPTIMIZATION),
        ("surrogate", ModelType.SURROGATE),
        (ModelType.SURROGATE, ModelType.SURROGATE),
    ]:
        assert ModelType.normalize(value) is expected
    with pytest.raises(ValueError):
        ModelType.normalize("bad")
    assert residual_mse(jnp.array([[2.0]]), jnp.array([[-3.0]])) == 2
    assert residual_mse(jnp.empty((1, 0)), jnp.empty((1, 0))) == 0
    assert TrainingConfig().batch_size == 32
    for kwargs in [dict(epochs=0), dict(batch_size=0), dict(print_every=-1)]:
        with pytest.raises(ValueError):
            TrainingConfig(**kwargs)
    with pytest.raises(ValueError):
        DualPretrainConfig(gamma=0)


def test_activation_layers_and_public_exports():
    import ohagi
    from ohagi.nn.layers import activations

    for name, reference in [
        ("Relu", "relu"),
        ("ReLU", "relu"),
        ("Gelu", "gelu"),
        ("GELU", "gelu"),
        ("Silu", "silu"),
        ("SiLU", "silu"),
        ("Sigmoid", "sigmoid"),
        ("Tanh", "tanh"),
        ("Softplus", "softplus"),
    ]:
        layer = getattr(activations, name)()
        import jax

        values = jnp.array([-1.0, 2.0])
        np.testing.assert_allclose(
            layer.apply((), values), getattr(jax.nn, reference)(values)
        )
    assert ohagi.MLP.__name__ == "MLP"
    assert ohagi.data is not None
    with pytest.raises(AttributeError):
        getattr(ohagi, "unknown_resource")


def test_bundle_parameters_and_utilities(tmp_path):
    from ohagi.serialization.bundle import save_model, load_parameters
    from ohagi._utils.serialization import json_safe, sha256, timestamp, resolve_path

    params = {"x": jnp.arange(3)}
    file = tmp_path / "metadata.json"
    save_model(
        file,
        framework="custom",
        parameters=params,
        inference_spec=dict(
            predict_fn=lambda x: 2 * x, input_size=1, output_size=1, dtype="float64"
        ),
    )
    np.testing.assert_array_equal(load_parameters(file)["x"], params["x"])
    import json

    data = json.loads(file.read_text())
    data["parameters"]["file"] = "../outside.npz"
    with pytest.raises(ValueError):
        load_parameters(file, data)
    with pytest.raises(ValueError):
        load_parameters(file, {"format": "bad"})
    with pytest.raises(ValueError):
        save_model(
            file,
            framework="custom",
            parameters=params,
            inference_spec={"framework": "other"},
        )
    assert json_safe(np.float64(2)) == 2
    assert json_safe(jnp.array([1.0, 2.0])) == [1.0, 2.0]
    assert len(sha256(file)) == 64
    assert len(timestamp()) == 15
    assert resolve_path(file) == file.resolve()
