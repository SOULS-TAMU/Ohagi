"""End-to-end projected trainer uses real PURIN solvers and compiled gradients."""

import jax.numpy as jnp
import numpy as np
import pytest
from purin import Projection
from purin.solvers.primal_dual_interior_point.reference.iteration import FixedMuOptions
from ohagi.nn.mlp import MLPBackbone
from ohagi.training.projected import ReportedProjectedModel


def make_model(*, inequalities=False, **kwargs):
    problem = Projection()
    p = problem.add_parameter(["b"])
    target = problem.add_parameter(["u0", "v0"])
    y = problem.add_variable(["u", "v"])
    problem.set_projection_objective(variables=y, target=[target.u0, target.v0])
    problem.constraints.add(y.u + y.v == p.b)
    if inequalities:
        problem.constraints.add(y.u >= 0, y.v >= 0)

    def prepare(batch_size):
        return problem.prepare_sparse_projection_batch_solver(
            parameters=np.zeros((batch_size, 3)),
            batch_size=batch_size,
            options=FixedMuOptions(max_iterations=20, residual_tolerance=1e-8),
            backend="cpu",
        )

    backbone = MLPBackbone(hidden_sizes=(3,))
    return ReportedProjectedModel(
        projection=prepare,
        backbone_init=backbone.init,
        backbone_apply=backbone.apply,
        input_dim=1,
        **kwargs,
    )


def test_train_evaluate_predict_and_export(tmp_path):
    model = make_model()
    assert not model.is_built
    with pytest.raises(RuntimeError):
        model.predict([[1.0]])
    model.build(batch_size=2)
    assert model.is_built
    with pytest.raises(RuntimeError):
        model.build(batch_size=2)
    x = np.array([[0.5], [1.0], [1.5]])
    y = np.repeat(x / 2, 2, axis=1)
    raw, projected = model.predict(x)
    assert raw.shape == projected.shape == (3, 2)
    np.testing.assert_allclose(projected.sum(1), x[:, 0], atol=1e-7)
    np.testing.assert_allclose(model.predict(x[0])[1], projected[0], atol=1e-12)
    np.testing.assert_allclose(model.predict_raw(x), raw)
    for projected_mode in [False, True]:
        metrics = model.evaluate(x, y, use_projection=projected_mode)
        assert np.isfinite(metrics["loss"])
    history = model.fit(
        x,
        y,
        epochs=2,
        pretrain_epochs=1,
        shuffle=False,
        print_every=1,
        validation_data=(x, y),
    )
    assert len(history) == 2
    assert history[0]["phase"] == "pretrain" and history[1]["phase"] == "projected"
    assert model.final_train_violation < 1e-7
    assert model.final_validation_violation < 1e-7
    assert model.evaluate_raw(x, y)["loss"] >= 0
    assert (
        model._maximum_constraint_violation(np.empty((0, 1)), use_projection=True) == 0
    )
    for bad in [np.zeros((2, 2)), np.zeros((2, 1, 1))]:
        with pytest.raises(ValueError):
            model.predict(bad)
        with pytest.raises(ValueError):
            model.evaluate(bad, y)
    for options in [
        dict(epochs=-1),
        dict(epochs=1, log_style="bad"),
        dict(epochs=1, pretrain_epochs=-1),
        dict(epochs=1, switch_metric="bad"),
    ]:
        with pytest.raises(ValueError):
            model.fit(x, y, **options)


@pytest.mark.parametrize("mode", ["supervised", "optimization", "inverse"])
def test_training_modes_duals_bounds_and_reporting(mode, tmp_path):
    kwargs = dict(
        mode=mode,
        predict_duals=True,
        consistency_weight=0.1,
        equality_dual_consistency_weight=0.1,
        inequality_dual_consistency_weight=0.1,
        invalid_batch_policy="use_valid",
        primal_lower_bounds=[0.0, 0.0],
        primal_upper_bounds=[2.0, 2.0],
    )
    if mode == "optimization":
        kwargs["objective_fn"] = lambda x, y: jnp.sum(y**2, axis=1)
    if mode == "inverse":
        kwargs.update(
            inverse_init=[0.2],
            inverse_regularizer=lambda p: 0.1 * jnp.sum(p**2),
            problem_parameter_fn=lambda x, p: x,
            backbone_input_fn=lambda x, p: jnp.concatenate(
                (x, jnp.broadcast_to(p, (x.shape[0], 1))), axis=1
            ),
            backbone_input_dim=2,
            primal_bounds_fn=lambda p: (
                jnp.zeros((p.shape[0], 2)),
                2 * jnp.ones((p.shape[0], 2)),
            ),
        )
    model = make_model(inequalities=True, **kwargs)
    x = np.array([[0.5], [1.0], [1.5]])
    y = None if mode == "optimization" else np.repeat(x / 2, 2, axis=1)
    model.build(
        batch_size=2,
        validate=False,
        example_x=x[:2],
        example_y=None if y is None else y[:2],
    )
    for style in ["none", "compact", "kkt"]:
        rows = model.fit(
            x,
            y,
            epochs=1,
            shuffle=True,
            seed=2,
            print_every=1,
            validation_data=(x, y),
            log_style=style,
        )
        assert len(rows) == 1 and np.isfinite(rows[0]["loss"])
        assert np.isfinite(model.evaluate(x, y)["loss"])
    assert model.summary(inference_warmup=0, inference_repeats=1)["variable_count"] == 2
    assert model._benchmark_inference(single=True, warmup=1, repeats=1) >= 0
    if mode != "optimization":
        rows = model.fit(x, y, epochs=2, pretrain_epochs=5, eta=1e9, print_every=1)
        assert rows[-1]["phase"] == "projected"
    else:
        with pytest.raises(ValueError):
            model.fit(x, epochs=1, pretrain_epochs=1)
    model.print_postprocessing_start()
    model.print_postprocessing_finish()
    model.print_citation()
    plt = pytest.importorskip("matplotlib.pyplot")
    model.plot_history(save_path=tmp_path / "history.png", show=False)
    assert (tmp_path / "history.png").is_file()
    plt.close("all")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bad"},
        {"mode": "optimization"},
        {"mode": "inverse"},
        {"input_dim": 0},
        {"consistency_weight": -1},
        {"backbone_input_dim": 0},
        {"equality_dual_consistency_weight": -1},
        {"inequality_dual_consistency_weight": np.nan},
        {"dual_pretrain_gamma": 0},
        {"inequality_dual_floor": -1},
        {
            "predict_duals": True,
            "dual_pretrain_gamma": 0.001,
            "inequality_dual_floor": 0.01,
        },
        {"inexact_gradient_tolerance": -1},
        {"adjoint_gradient_tolerance": np.nan},
        {"gradient_policy": "bad"},
        {"invalid_batch_policy": "bad"},
    ],
)
def test_invalid_trainer_configuration(kwargs):
    from ohagi.training.projected import ProjectedTrainerCore

    values = dict(
        projection=None,
        backbone_init=lambda *args: (),
        backbone_apply=lambda p, x: x,
        input_dim=1,
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        ProjectedTrainerCore(**values)


def test_raw_graph_disabled_and_input_validation():
    model = make_model()
    with pytest.raises(ValueError):
        model.build(batch_size=0)
    model.build(batch_size=1, compile_raw_predict=False)
    with pytest.raises(RuntimeError):
        model.predict_raw([[1.0]])
    with pytest.raises(RuntimeError):
        model._maximum_constraint_violation([[1.0]], use_projection=False)
    with pytest.raises(ValueError):
        model._maximum_constraint_violation([[1.0, 2.0]], use_projection=True)
    with pytest.raises(ValueError):
        model.evaluate([[1.0]])
    with pytest.raises(ValueError):
        model.fit([[1.0]], epochs=1)
    with pytest.raises(ValueError):
        model._default_problem_parameters(jnp.ones((1, 1)), None, 2)
    np.testing.assert_array_equal(
        model._default_problem_parameters(jnp.ones((1, 1)), jnp.array([2.0]), 2),
        [[1.0, 2.0]],
    )
    np.testing.assert_array_equal(
        model._default_backbone_inputs(jnp.ones((1, 1)), jnp.array([2.0])), [[1.0, 2.0]]
    )
    with pytest.raises(ValueError):
        model.fit([[1.0]], [[0.5, 0.5]], epochs=0)
