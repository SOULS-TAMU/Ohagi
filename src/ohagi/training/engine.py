"""Pure stateful model steps and an optional host-side trainer."""

from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from ohagi._utils.tree import tree_all_finite, tree_select
from ohagi.training.steps import apply_updates as apply_updates, optimizer_step
from ohagi.data import fixed_batches
from .runner import train_epoch, evaluate_epoch, mean_metrics
from .runner import run_epochs, EpochTransition
from ._engine import Engine as Engine, EpochPlan


class TrainState(NamedTuple):
    """Backbone parameters, model state, optimizer state, PRNG key and step count."""

    parameters: object
    model_state: object
    optimizer_state: object
    key: object
    step: object


def create_train_state(
    backbone, optimizer, key, *, input_dim, output_dim, dtype=jnp.float64
):
    """Initialize a backbone and optimizer with independent initialization and run keys."""
    init_key, run_key = jax.random.split(key)
    params, state = backbone.init(init_key, input_dim, output_dim, dtype)
    return TrainState(
        params, state, optimizer.init(params), run_key, jnp.asarray(0, jnp.int32)
    )


def make_model_train_step(model, loss_fn, optimizer):
    """Build a model training step that commits only eligible finite updates."""

    def step(state, batch):
        next_key, apply_key = jax.random.split(state.key)

        def objective(params):
            outputs, model_state = model.apply(
                params,
                state.model_state,
                batch["inputs"],
                context=batch.get("theta"),
                key=apply_key,
                training=True,
            )
            loss, metrics = loss_fn(outputs, batch)
            mask = batch.get("mask", jnp.ones(outputs["projected"].shape[0], bool))
            eligible = jnp.any(outputs["dual_valid"] & mask)
            return loss, (metrics, model_state, eligible)

        (loss, (metrics, model_state, eligible)), grads = jax.value_and_grad(
            objective, has_aux=True
        )(state.parameters)
        valid = eligible & jnp.isfinite(loss) & tree_all_finite((grads, model_state))
        if "gradient_valid" in metrics:
            valid &= jnp.all(metrics["gradient_valid"])
        params, opt_state, valid = optimizer_step(
            state.parameters,
            state.optimizer_state,
            grads,
            optimizer,
            valid=valid,
            check_candidate_finite=True,
        )
        new = TrainState(
            params,
            tree_select(valid, model_state, state.model_state),
            opt_state,
            next_key,
            state.step + valid.astype(jnp.int32),
        )
        return new, {**metrics, "loss": loss, "update_accepted": valid}

    return step


def make_model_eval_step(model, loss_fn):
    """Build a model evaluation step without updating parameters or model state."""

    def step(state, batch):
        outputs, _ = model.apply(
            state.parameters,
            state.model_state,
            batch["inputs"],
            context=batch.get("theta"),
            key=state.key,
            training=False,
        )
        loss, metrics = loss_fn(outputs, batch)
        return {**metrics, "loss": loss}

    return step


@dataclass(frozen=True)
class TrainingResult:
    """Final training state and the ordered sequence of epoch records."""

    state: TrainState
    history: tuple


@dataclass(frozen=True)
class Trainer:
    """Host-side fitting interface over pure model training and evaluation steps."""

    model: object
    loss: object
    optimizer: object
    jit: bool = True

    def fit(
        self,
        state,
        data,
        *,
        epochs,
        batch_size,
        shuffle=True,
        seed=0,
        callbacks=(),
        validation=None,
    ):
        """Fit a model over fixed batches and return its final state and history."""
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be positive")
        train = make_model_train_step(self.model, self.loss, self.optimizer)
        evaluate = make_model_eval_step(self.model, self.loss)
        if self.jit:
            train, evaluate = jax.jit(train), jax.jit(evaluate)
        rng = np.random.default_rng(seed)
        size = len(next(iter(data.values())))
        history = []

        def aggregate(rows):
            total = sum(n for n, _ in rows)
            return {
                key: sum(n * float(np.asarray(values[key])) for n, values in rows)
                / total
                for key in rows[0][1]
            }

        def epoch_policy(epoch_number, state):
            epoch = epoch_number - 1
            indices = rng.permutation(size) if shuffle else np.arange(size)
            prepare = lambda batch: (batch, int(batch["mask"].sum()))
            result = yield EpochPlan(
                state,
                fixed_batches(data, batch_size, indices=indices),
                train,
                prepare=prepare,
            )
            state = result.state
            rows = list(zip(result.counts, jax.device_get(result.metrics)))
            record = {"epoch": epoch + 1, **aggregate(rows)}
            if validation is not None:
                evaluated = evaluate_epoch(
                    state,
                    fixed_batches(validation, batch_size),
                    evaluate,
                    prepare=prepare,
                )
                values = list(zip(evaluated.counts, jax.device_get(evaluated.metrics)))
                record.update(
                    {"validation_" + k: v for k, v in aggregate(values).items()}
                )
            history.append(record)
            stop = False
            for callback in callbacks:
                stop = bool(callback(epoch + 1, state, record)) or stop
            return EpochTransition(state, stop)

        state = run_epochs(state, epochs, epoch_policy).state
        return TrainingResult(state, tuple(history))


def build_compiled_epoch_fns(train_step, eval_step, metric_keys):
    """Build per-epoch train and evaluation reducers over fixed batches."""

    @jax.jit
    def train_kernel(state, batches):
        def body(st, xb):
            st, metrics = train_step(st, xb)
            return st, tuple(metrics[k] for k in metric_keys)

        result = train_epoch(state, batches, body, execution="scan")
        return result.state, mean_metrics(result.metrics)

    @jax.jit
    def eval_epoch(params, batches):
        def one_batch(parameters, xb):
            metrics = eval_step(parameters, xb)
            return tuple(metrics[k] for k in metric_keys)

        result = evaluate_epoch(params, batches, one_batch, execution="scan")
        return mean_metrics(result.metrics)

    return train_kernel, eval_epoch
