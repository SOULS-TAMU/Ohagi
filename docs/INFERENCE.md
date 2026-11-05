# JAX export and repeated inference

An inference bundle contains `metadata.json` and the referenced `.jax_export`
program. Pass only the metadata path to the loader, but move all referenced files
together. Metadata alone is not a complete model.

## Export a prediction graph

The shared entry point accepts a JAX-compatible batched function with frozen
parameters and an explicit input/output signature:

```python
from ohagi.inference import export, load

export("deployment/metadata.json", predict_function,
       input_size=1, output_size=2, framework="ohagi")
predictor = load("deployment/metadata.json")
prediction = predictor.predict([0.7])
```

`predict_function` must accept shape `(1, input_size)` and return
`(1, output_size)`. Capture trained numeric parameters/state in that graph.
The [runnable export example](examples/deployment.rst) supplies the entire function.

## Reuse the prepared predictor

Load once outside the inference loop. The loader validates the graph/signature
and warms up a single-instance call. Inputs can be `(input_size,)` or
`(1, input_size)`; larger batches are a different interface.

`predict` returns synchronized NumPy values. `predict_jax` retains a device array;
synchronize it for completed-work timing. Reuse shape, precision and device.
Loading/warmup and steady-state prediction should be measured separately.

The originating frontend, Flax and Orbax are not required to execute a compatible
frozen graph. A full-model bundle and an Orbax optimizer checkpoint have different
purposes; see [artifact formats](artifacts.md). JAX is the deployment backend.
There is no separate native compilation or standalone StableHLO save option.
