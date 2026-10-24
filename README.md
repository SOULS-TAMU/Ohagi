<p align="center">
  <img src="assets/ohagi-logo.png" alt="OHAGI logo" width="220">
</p>

# OHAGI

[![CI](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/ci.yml)
[![Coverage](https://raw.githubusercontent.com/SOULS-TAMU/Ohagi/codex/coverage/coverage.svg)](https://github.com/SOULS-TAMU/Ohagi/tree/codex/coverage)
[![CodeQL](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/codeql.yml)
[![Validation](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/validation.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/validation.yml)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Powered by JAX](https://img.shields.io/badge/JAX-powered-8A2BE2)](https://docs.jax.dev/)
[![License: Apache-2.0](https://img.shields.io/github/license/SOULS-TAMU/Ohagi?color=blue)](LICENSE)
[![Citation: BibTeX](https://img.shields.io/badge/Citation-BibTeX-6F4E37)](CITATION.bib)

**A shared JAX engine for hard-constrained neural networks.**

Compose a neural backbone with a differentiable projection, train the model,
and prepare it for repeated single-instance inference. OHAGI powers KKTHN,
DAEHN and NLPOPTNet, with projection algorithms and sparse kernels supplied by PURIN.

[Installation](#installation) · [Quick start](#quick-start) · [Architecture](#architecture) · [Inference](#inference) · [Contributing](CONTRIBUTING.md)

- **Flexible backbones:** functional JAX networks and optional Flax Linen modules.
- **Shared training:** projected models, losses, optimizers and optional Orbax checkpoints.
- **Portable inference:** one JAX export/load interface across all three frontends.

## Installation

From a `hardnet-stack` checkout:

```shell
python -m pip install ./purin ./ohagi
```

Python 3.10+ is required. For GPU support, follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Optional integrations: `python -m pip install "./ohagi[flax,orbax,optax]"`.

## Quick start

Train a small network whose projected outputs satisfy **u + v = b**:

```python
import jax
import jax.numpy as jnp
from purin import Distance, Projection
from ohagi import Adam, ProjectedModel, Trainer
from ohagi.nn import MLPAdapter, MLPBackbone
from ohagi.losses import supervised_mse
from ohagi.training.engine import create_train_state

jax.config.update("jax_enable_x64", True)

problem = Projection()
p = problem.add_parameter(["b"])
y = problem.add_variable(["u", "v"])
problem.objective = 0.0
problem.constraints.add(y.u + y.v == p.b)

projection = Distance(problem, "linear-equality").prepare(batch_size=4)
backbone = MLPAdapter(MLPBackbone(hidden_sizes=(16, 16)))
model = ProjectedModel(backbone, projection)
optimizer = Adam(learning_rate=1e-3)
state = create_train_state(
    backbone, optimizer, jax.random.PRNGKey(0), input_dim=1, output_dim=2
)
data = {
    "inputs": jnp.ones((8, 1)),
    "theta": jnp.ones((8, 1)),
    "solution": jnp.tile(jnp.array([0.8, 0.2]), (8, 1)),
}
result = Trainer(model, supervised_mse(), optimizer).fit(
    state, data, epochs=2, batch_size=4
)
outputs, _ = model.apply(
    result.state.parameters, result.state.model_state, jnp.ones((4, 1))
)
print(outputs["projected"])
```

## Architecture

OHAGI provides the shared learning workflow for the HardNet stack. The libraries
separate framework-specific problem formulations from model training and
numerical projection:

- **KKTHN, DAEHN and NLPOPTNet** supply their problem formulations and
  framework-specific processing to the shared engine.
- **OHAGI** composes backbones, projections and losses, runs training, and
  manages checkpoints and JAX model export for `metadata.json` inference.
- **PURIN** compiles projection problems and provides the solvers, sparse
  kernels and differentiation used by the projection layers.

## Inference

Export a trained frontend model with `model.export_inference("deployment/metadata.json")`.
Then load once and reuse the predictor:

```python
from ohagi.inference import load

predictor = load("deployment/metadata.json")
prediction = predictor.predict(single_input)
```

Keep the referenced `.jax_export` file alongside `metadata.json`.
`predict` returns NumPy values; `predict_jax` keeps results on the JAX device.
Deployment uses JAX and does not require a frontend, Flax or Orbax.
See [model artifacts and checkpoints](docs/artifacts.md) for details.

## Project information

[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [Citation](CITATION.bib)

## Cross-platform compatibility

CPU compatibility is verified in GitHub Actions:

| Platform | Python versions |
| --- | --- |
| Linux x86-64 | 3.10–3.14 |
| Windows x86-64 | 3.10–3.14 |
| macOS Apple Silicon | 3.10–3.14 |
| Linux ARM64 | 3.13 |

CI checks current compatible JAX releases and JAX 0.6.2 on Python 3.12
across all three OSes. See the [CI matrix](CONTRIBUTING.md#automated-checks).
Flax, Optax, Orbax and Linux-exported inference bundles are also tested
on Linux, Windows and macOS.


## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
