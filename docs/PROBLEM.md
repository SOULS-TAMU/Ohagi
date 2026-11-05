# Models, backbones and losses

## The shared pathway

A frontend supplies a problem and its preparation policy; PURIN supplies the
projection. OHAGI composes a backbone, projection, loss and optimizer and runs
one shared learning engine. Its numeric state is not a framework-specific
problem implementation.

The [quick start](examples/quickstart.rst) trains a model with `u+v=b`.
`ProjectedModel` composes a backbone adapter and prepared projection. `Trainer`
runs the update loop. The example's observations are a mapping with a common
sample axis: network `inputs`, projection parameters `theta`, and target `solution`.

## Backbone contract

Stateful adapters expose:

```python
parameters, state = backbone.init(key, input_dim, output_dim, dtype)
outputs, next_state = backbone.apply(
    parameters, state, inputs, key=key, training=True,
)
```

`MLPAdapter` wraps functional `MLPBackbone`. `FunctionalBackbone` adapts custom
JAX-compatible initialization/application functions. Optional Flax adapters use
the same engine contract. Handle nontrainable state and RNG keys explicitly.
Changing shapes or dtypes can require another compilation.

## Loss and optimizer

`supervised_mse()` compares projected solutions with targets. Other loss builders
and frontend policies determine raw, projected, differential and consistency
terms. PURIN supplies projection derivatives; OHAGI composes them with the
network derivative and optimizer update. A low training loss does not itself
establish feasibility, solver convergence or valid gradients.
