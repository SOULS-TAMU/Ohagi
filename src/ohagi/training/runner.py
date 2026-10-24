"""Stable batch-runner interface over shared execution resources."""

from ._batch import (
    EpochResult as EpochResult,
    EpochTransition as EpochTransition,
    RunResult as RunResult,
    train_epoch as train_epoch,
    evaluate_epoch as evaluate_epoch,
    mean_metrics as mean_metrics,
    parameter_step as parameter_step,
    warmup as warmup,
)


def run_epochs(state, end_epoch, epoch_step, *, start_epoch=1):
    """Own epoch ordering and early stopping for all frontend policies.

    An epoch policy selects prepared graphs, batch order, validation, reporting,
    and phase changes. Numerical batch execution is provided by train_epoch.
    The explicit epoch range preserves cumulative checkpoint-resume semantics.
    """
    from ._engine import Engine

    return Engine.run(state, end_epoch, epoch_step, start_epoch=start_epoch)
