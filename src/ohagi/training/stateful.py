"""Flax/Optax state transport over OHAGI's common pure training step."""

from functools import partial
import jax
import jax.numpy as jnp
from .steps import build_train_step
from .runner import warmup


def build_stateful_steps(
    loss_fn,
    backbone,
    *,
    batch_size,
    input_size,
    dtype,
    learning_rate=None,
    optimizer=None,
    invalid_batch_policy="skip",
    include_update_accepted=True,
):
    """Adapt Flax TrainState to OHAGI's common parameter-training transition.

    Args:
        loss_fn: Callable ``(params, inputs) -> (scalar_loss, metrics)``.
        backbone: Flax module providing ``init`` and ``apply``.
        batch_size: Fixed initialization batch size.
        input_size: Input feature count.
        dtype: Initialization input dtype.
        learning_rate: Adam rate used when no optimizer is supplied.
        optimizer: Optional Optax optimizer overriding the default Adam choice.
        invalid_batch_policy: Shared update policy: ``skip``, ``use_valid``
            or ``propagate``.
        include_update_accepted: Retain the update eligibility metric when True.

    Returns:
        Tuple ``(init_state, train_step, eval_step)``. Training consumes a Flax
        TrainState and fixed-shape inputs; evaluation consumes params and inputs.

    Notes:
        The compiled train step donates its state argument. Do not reuse an
        old state's donated buffers after calling it. Flax and Optax are
        imported only when constructing these optional training resources.
    """
    import optax
    from flax.training.train_state import TrainState

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state, inputs):
        transition = build_train_step(
            loss_fn,
            state.tx,
            invalid_batch_policy=invalid_batch_policy,
            optimizer_apply_updates=optax.apply_updates,
        )
        params, opt_state, metrics = transition(state.params, state.opt_state, inputs)
        if not include_update_accepted:
            metrics = {
                name: value
                for name, value in metrics.items()
                if name != "update_accepted"
            }
        return state.replace(
            step=state.step + 1, params=params, opt_state=opt_state
        ), metrics

    @jax.jit
    def eval_step(params, inputs):
        _, metrics = loss_fn(params, inputs)
        return metrics

    def init_state(rng):
        params = backbone.init(rng, jnp.zeros((batch_size, input_size), dtype=dtype))[
            "params"
        ]
        return TrainState.create(
            apply_fn=backbone.apply,
            params=params,
            tx=optax.adam(learning_rate) if optimizer is None else optimizer,
        )

    return init_state, train_step, eval_step


def warmup_stateful(
    *,
    state,
    train_step_fn,
    eval_step_fn,
    input_size,
    batch_size,
    seed,
    dtype=jnp.float64,
    device=None,
    commit_update=False,
):
    """Warm compiled training/evaluation steps with deterministic synthetic inputs.

    Args:
        state: Initial training state.
        train_step_fn: Stateful training transition to warm.
        eval_step_fn: Parameter-only evaluation function to warm.
        input_size: Synthetic input feature count.
        batch_size: Synthetic batch size matching the compiled training signature.
        seed: PRNG seed for synthetic normal inputs.
        dtype: Synthetic input dtype.
        device: Optional device on which to place inputs and state.
        commit_update: Retain the synthetic training update when True; otherwise
            preserve the supplied training state through the shared warm-up policy.

    Returns:
        The result of the shared ``warmup`` runner, including its chosen state.
    """
    inputs = jax.random.normal(
        jax.random.PRNGKey(seed), (batch_size, input_size), dtype=dtype
    )
    if device is not None:
        inputs, state = jax.device_put(inputs, device), jax.device_put(state, device)
    return warmup(
        state, inputs, train_step_fn, eval_step_fn, commit_update=commit_update
    )
