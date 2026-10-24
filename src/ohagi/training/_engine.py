"""The single shared epoch engine, separated from model/trainer preparation."""

from dataclasses import dataclass
from ._batch import train_epoch, EpochResult, RunResult


@dataclass(frozen=True)
class EpochPlan:
    """A task supplies data and prepared graphs; Engine owns their execution."""

    state: object
    batches: object
    step: object = None
    prepare: object = lambda batch: (batch, 1)
    execution: str = "host"
    kernel: object = None


class Engine:
    """One epoch and batch pathway for stateless or stateful JAX task adapters.

    Policies are generators: yield an EpochPlan, then use its EpochResult to
    report, validate, or select the next phase. Kernels may be precompiled by
    JAX, preserving donation and scan behavior without frontend epoch loops.
    """

    @staticmethod
    def execute(plan):
        """Execute one prepared epoch plan through the shared batch pathway.

        Args:
            plan: ``EpochPlan`` containing state, batches and either a compiled
                epoch kernel or a batch step with its preparation/execution policy.

        Returns:
            ``EpochResult`` containing the updated state and epoch metrics.
            A supplied kernel determines metric reduction and donation behavior.
        """
        if plan.kernel is not None:
            state, metrics = plan.kernel(plan.state, plan.batches)
            return EpochResult(state, metrics, None)
        return train_epoch(
            plan.state,
            plan.batches,
            plan.step,
            prepare=plan.prepare,
            execution=plan.execution,
        )

    @classmethod
    def run(cls, state, end_epoch, policy, *, start_epoch=1):
        """Drive frontend epoch policy through the shared host training loop.

        Args:
            state: Initial training state passed to the epoch policy.
            end_epoch: Inclusive final epoch number.
            policy: Callable ``policy(epoch, state)`` returning an
                ``EpochTransition``, or a generator that yields ``EpochPlan``
                objects and receives their ``EpochResult`` values.
            start_epoch: Inclusive first epoch number, including when resuming.

        Returns:
            ``RunResult`` containing the final state and last completed epoch.
            Execution stops early when the policy's transition requests it.

        Raises:
            TypeError: If a generator policy yields anything other than an epoch plan.
        """

        completed = start_epoch - 1
        for epoch in range(start_epoch, end_epoch + 1):
            task = policy(epoch, state)
            if hasattr(task, "send"):
                result = None
                while True:
                    try:
                        plan = task.send(result)
                    except StopIteration as done:
                        transition = done.value
                        break
                    if not isinstance(plan, EpochPlan):
                        raise TypeError(
                            "Training policies must yield EpochPlan objects."
                        )
                    result = cls.execute(plan)
            else:
                transition = task
            state, completed = transition.state, epoch
            if transition.stop:
                break
        return RunResult(state, completed)


EpochPlan.__module__ = "ohagi.training.engine"
Engine.__module__ = "ohagi.training.engine"
