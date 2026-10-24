"""Training, batching and optimizer behavior independent of the frontends."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ohagi import Adam
from ohagi.data.batching import fixed_batches
from ohagi.training.engine import Engine, EpochPlan
from ohagi.training.runner import EpochTransition, evaluate_epoch, train_epoch
from ohagi.training.steps import build_train_step


@pytest.mark.parametrize("execution", ["host", "scan"])
def test_epoch_execution_and_evaluation(execution):
    batches = jnp.array([1.0, 2.0, 3.0])
    result = train_epoch(
        0.0,
        batches,
        lambda state, batch: (state + batch, {"sum": state + batch}),
        execution=execution,
    )
    np.testing.assert_allclose(result.state, 6.0)
    result = evaluate_epoch(
        2.0, batches, lambda state, batch: state + batch, execution=execution
    )
    np.testing.assert_allclose(result.state, 2.0)


def test_resume_epoch_policy_matches_uninterrupted():
    def policy(epoch, state):
        result = yield EpochPlan(
            state, [epoch], step=lambda state, batch: (state + batch, {})
        )
        return EpochTransition(result.state, False)

    uninterrupted = Engine.run(0, 6, policy)
    first = Engine.run(0, 3, policy)
    resumed = Engine.run(first.state, 6, policy, start_epoch=4)
    assert uninterrupted == resumed
    assert resumed.state == 21


def test_invalid_epoch_policy():
    def invalid(epoch, state):
        yield "not an epoch plan"

    with pytest.raises(TypeError, match="EpochPlan"):
        Engine.run(0, 1, invalid)


def test_padded_batches_preserve_valid_samples():
    batches = list(fixed_batches({"x": np.arange(5)}, 3))
    np.testing.assert_array_equal(batches[1]["x"], [3, 4, 4])
    np.testing.assert_array_equal(batches[1]["mask"], [True, True, False])
    assert len(list(fixed_batches({"x": np.arange(5)}, 3, drop_last=True))) == 1


@pytest.mark.parametrize(
    "data, size",
    [({}, 2), ({"x": [1]}, 0), ({"x": [1], "y": [1, 2]}, 2), ({"mask": [True]}, 1)],
)
def test_invalid_batches(data, size):
    with pytest.raises(ValueError):
        list(fixed_batches(data, size))


def test_adam_training_reduces_loss_and_rejects_nonfinite_update():
    optimizer = Adam(learning_rate=0.05)
    parameters = {"w": jnp.array([2.0])}
    state = optimizer.init(parameters)
    step = jax.jit(
        build_train_step(
            lambda p, target: (jnp.sum((p["w"] - target) ** 2), {}), optimizer
        )
    )
    initial = float(jnp.sum(parameters["w"] ** 2))
    for _ in range(20):
        parameters, state, metrics = step(parameters, state, jnp.zeros(1))
    assert float(jnp.sum(parameters["w"] ** 2)) < initial / 2
    previous = parameters["w"]
    previous_count = state.count
    parameters, state, metrics = step(parameters, state, jnp.array([jnp.nan]))
    np.testing.assert_array_equal(parameters["w"], previous)
    np.testing.assert_array_equal(state.count, previous_count)
    assert not bool(metrics["update_accepted"])
