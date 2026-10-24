"""Loss reductions checked with independent scalar expectations and gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ohagi.losses import core
from ohagi.losses.config import Loss

RAW = jnp.array([[1.0, 2.0], [100.0, 100.0]])
PROJECTED = jnp.array([[2.0, 4.0], [200.0, 200.0]])
MASK = jnp.array([True, False])


def outputs():
    return dict(
        raw=RAW,
        projected=PROJECTED,
        lambda_raw=jnp.ones((2, 1)),
        lambda_projected=jnp.zeros((2, 1)),
        mu_raw=2 * jnp.ones((2, 1)),
        mu_projected=jnp.zeros((2, 1)),
        dual_valid=MASK,
    )


def test_mse_wrappers_and_mask():
    for fn in [
        core.mse,
        core.raw_output_mse,
        core.projected_output_mse,
        core.projection_consistency,
    ]:
        assert fn(RAW, PROJECTED, MASK) == 2.5
        assert fn(RAW, PROJECTED, jnp.zeros(2)) == 0
        np.testing.assert_allclose(
            fn(RAW, PROJECTED), np.mean(np.square(np.asarray(RAW - PROJECTED)))
        )
    assert core.masked_mean(jnp.array([2.0, 4.0])) == 3
    batch = dict(
        solution=jnp.zeros((2, 2)),
        theta=jnp.array([[1.0], [2.0]]),
        mask=MASK,
        observations=jnp.array([[2.0], [9.0]]),
    )
    for weight in [0.0, 0.5]:
        value, metrics = core.supervised_mse(consistency_weight=weight)(
            outputs(), batch
        )
        assert value == 10 + weight * 2.5
        assert metrics["task_loss"] == 10
    assert core.inverse_observation(lambda y: y[:, :1])(outputs(), batch)[0] == 0
    objective = lambda theta, y: theta[0] + jnp.sum(y**2)
    assert core.problem_objective(objective)(outputs(), batch)[0] == 21
    np.testing.assert_allclose(
        core.objective_loss(objective, respect_mask=False)(outputs(), batch)[0],
        (21 + 80002) / 2,
    )
    terms = {
        "fit": (2.0, core.supervised_mse()),
        "obs": (3.0, core.inverse_observation(lambda y: y[:, :1])),
    }
    for loss in [
        core.weighted_sum(terms),
        core.combine_losses(terms, initial_dtype=jnp.float64),
    ]:
        value, metrics = loss(outputs(), batch)
        assert value == 20
        assert metrics["fit/task_loss"] == 10


@pytest.mark.parametrize(
    "valid,sample", [(None, None), (MASK, None), (None, MASK), (MASK, MASK)]
)
def test_dual_teacher_is_stop_gradient(valid, sample):
    pred = jnp.array([[1.0], [5.0]])
    target = jnp.zeros((2, 1))
    expected = 13 if valid is None and sample is None else 1
    assert core.latent_dual_consistency(pred, target, valid, sample) == expected
    np.testing.assert_array_equal(
        jax.grad(lambda t: core.latent_dual_consistency(pred, t, valid, sample))(
            target
        ),
        target,
    )
    assert core.latent_dual_consistency(jnp.empty((2, 0)), jnp.empty((2, 0))) == 0


@pytest.mark.parametrize("kind", ["mse", "objective", "observation"])
@pytest.mark.parametrize("mask", [False, True])
def test_configured_loss(kind, mask):
    out = outputs()
    batch = dict(
        theta=jnp.ones((2, 1)),
        solution=jnp.zeros((2, 1 if kind == "observation" else 2)),
    )
    if mask:
        batch["mask"] = MASK
    spec = Loss(
        kind=kind,
        primal_consistency_weight=2,
        equality_dual_consistency_weight=3,
        inequality_dual_consistency_weight=4,
    )
    loss = core.configured_training_loss(
        spec, objective=lambda t, y: jnp.sum(y), observation_indices=[0]
    )
    value, metrics = loss(out, batch)
    task = {"mse": 10, "objective": 6, "observation": 4}[kind]
    assert value == task + 5 + 3 + 16
    assert metrics["task_loss"] == task
    out.pop("dual_valid")
    assert jnp.isfinite(loss(out, batch)[0])
    if kind == "objective":
        with pytest.raises(ValueError):
            core.configured_training_loss(spec)(out, batch)


def test_loss_configuration_roundtrip_and_validation():
    for spec in [
        Loss.mse(),
        Loss.objective(),
        Loss.parameter_estimation(observed_variables=["a"]),
    ]:
        assert Loss.from_dict(spec.to_dict()) == spec
    for kwargs in [
        dict(kind="bad"),
        dict(primal_consistency_weight=-1),
        dict(equality_dual_consistency_weight=-1),
        dict(inequality_dual_consistency_weight=-1),
        dict(observed_variables=()),
    ]:
        with pytest.raises(ValueError):
            Loss(**kwargs)
