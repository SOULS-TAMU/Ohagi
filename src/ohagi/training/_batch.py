"""Shared batch execution for training and evaluation.

Host execution accommodates adaptive phases and padded datasets. Scan execution
keeps a fixed-shape epoch on device. Both consume the same state/step contract;
neither depends on a frontend or problem formulation.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class EpochResult:
    """Updated state and reduced metrics returned by a batch scan."""

    state: object
    metrics: object
    counts: object


@dataclass(frozen=True)
class EpochTransition:
    """State, report and stop decision returned by an epoch policy."""

    state: object
    stop: bool = False


@dataclass(frozen=True)
class RunResult:
    """Final state and accumulated records from the host epoch loop."""

    state: object
    completed_epoch: int


def train_epoch(
    state, batches, step, *, prepare=lambda batch: (batch, 1), execution="host"
):
    """Scan a pure training transition over the supplied epoch batches."""
    if execution == "scan":
        state, metrics = jax.lax.scan(step, state, batches)
        return EpochResult(state, metrics, None)
    if execution != "host":
        raise ValueError(f"Unknown epoch execution {execution!r}")
    rows, counts = [], []
    for batch in batches:
        payload, count = prepare(batch)
        state, metrics = step(state, payload)
        rows.append(metrics)
        counts.append(count)
    return EpochResult(state, rows, counts)


def evaluate_epoch(
    state, batches, step, *, prepare=lambda batch: (batch, 1), execution="host"
):
    """Scan evaluation batches without changing the supplied parameters."""
    if execution == "scan":
        return EpochResult(
            state, jax.vmap(lambda batch: step(state, batch))(batches), None
        )
    if execution != "host":
        raise ValueError(f"Unknown epoch execution {execution!r}")
    rows, counts = [], []
    for batch in batches:
        payload, count = prepare(batch)
        rows.append(step(state, payload))
        counts.append(count)
    return EpochResult(state, rows, counts)


def mean_metrics(rows):
    """Average each metric leaf across its leading batch dimension."""
    return jax.tree_util.tree_map(lambda values: jnp.mean(values), rows)


def parameter_step(step):
    """Adapt historical (parameters, optimizer, *batch) steps to state steps."""

    def update(state, batch):
        parameters, optimizer_state, metrics = step(*state, *batch)
        return (parameters, optimizer_state), metrics

    return update


def warmup(
    state,
    example,
    train_step,
    eval_step,
    *,
    parameters=lambda state: state.params,
    commit_update=False,
):
    """Compile and execute training/evaluation, optionally retaining the warm-up update."""
    candidate, _ = train_step(state, example)
    eval_step(parameters(candidate), example)
    jax.block_until_ready(parameters(candidate))
    return candidate if commit_update else state


# Preserve historical qualified identities for checkpoint and introspection users.
EpochResult.__module__ = "ohagi.training.runner"
EpochTransition.__module__ = "ohagi.training.runner"
RunResult.__module__ = "ohagi.training.runner"
train_epoch.__module__ = "ohagi.training.runner"
evaluate_epoch.__module__ = "ohagi.training.runner"
mean_metrics.__module__ = "ohagi.training.runner"
parameter_step.__module__ = "ohagi.training.runner"
warmup.__module__ = "ohagi.training.runner"
