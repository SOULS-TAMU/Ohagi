# Training state and checkpoints

## Explicit state, shared execution

`create_train_state` initializes backbone parameters/state and optimizer state.
`Trainer.fit` runs the composed model and loss. `TrainingResult` carries the
updated `state` and history. Preserve nontrainable model state as well as
parameters when using stateful layers.

Fixed-shape batching, update builders and epoch plans are common services.
Frontends choose task-specific preparation and phase policy rather than replacing
the engine. Custom losses must handle masks/padding and the supplied data contract.

## Checkpoint versus deployment

Orbax checkpoints store numeric training state for continuation. Reconstruct the
same problem, network, optimizer, data ordering and phase configuration before
restoring. High-level resume entry points depend on the frontend.

A deployment bundle freezes a prediction graph for repeated inference. It does
not replace optimizer state or resume training. See [artifacts](artifacts.md)
and [inference](INFERENCE.md).

Functional JAX suffices for the core engine. Optional Flax, Orbax and Optax
integrations do not change the deployment loader's metadata-path interface.
