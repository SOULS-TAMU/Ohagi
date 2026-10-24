# Model artifacts and checkpoints

## Deployment bundles

All three frontends use the same OHAGI JAX deployment format. Export from a
trained frontend model:

```python
model.export_inference("deployment/metadata.json")
```

Pass the metadata path to `ohagi.inference.load`. Keep its referenced
`.jax_export` file in the bundle; the JSON alone does not contain the complete
model. Loading validates the graph and prepares a fixed single-instance
signature. Reuse that predictor across calls.

Inputs may have shape `(input_size,)` or `(1, input_size)`. `predict` returns
synchronized NumPy output; `predict_jax` returns a JAX array. Synchronize device
results when measuring completed inference time.

Deployment bundles use `ohagi_model_v1`, `ohagi_jax_inference_v1` and
`ohagi_jax_program_v1`. The graph captures the parameters and operations needed
for prediction. Deployment requires OHAGI, PURIN and a compatible JAX runtime;
it does not reconstruct the original frontend problem or import Flax or Orbax.

## Full-model reload and training checkpoints

Frontend full-model saves also retain formulation data and a parameter archive
using `ohagi_pytree_npz_v1`. Frontend reload capabilities depend on whether the
problem definition is serializable. These resources serve a different purpose
from a prepared deployment predictor.

Optional Orbax checkpoints preserve numeric training state, including optimizer
state, for supported frontend resume workflows. They are separate from
deployment exports. The historical `hax.json` sidecar and `hax_orbax_v1`
identifier remain for checkpoint compatibility.

JAX is the supported deployment backend. JAX export uses StableHLO internally;
there is no separate native backend or standalone StableHLO export option.
