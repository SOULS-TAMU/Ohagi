<p align="center">
  <img src="https://raw.githubusercontent.com/SOULS-TAMU/Ohagi/main/assets/ohagi-logo.png" alt="OHAGI logo" width="200">
</p>

# OHAGI

**One JAX engine for building, training and deploying hard-constrained neural models.**

[![PyPI](https://img.shields.io/pypi/v/ohagi)](https://pypi.org/project/ohagi/)
[![Documentation](https://img.shields.io/readthedocs/ohagi/latest?logo=readthedocs&label=docs)](https://ohagi.readthedocs.io/en/latest/)

[![CI](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/ci.yml?query=branch%3Amain)
[![Validation](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/validation.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/validation.yml?query=branch%3Amain)
[![CodeQL](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/codeql.yml?query=branch%3Amain)
[![Dependency review](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/dependency-review.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Ohagi/actions/workflows/dependency-review.yml?query=branch%3Amain)
[![Coverage](https://raw.githubusercontent.com/SOULS-TAMU/Ohagi/codex/coverage/coverage.svg)](https://github.com/SOULS-TAMU/Ohagi/tree/codex/coverage)

[![Python 3.10â€“3.15](https://img.shields.io/badge/Python-3.10%E2%80%933.15-3776AB?logo=python&logoColor=white)](.github/workflows/release.yml)
[![CPU platforms](https://img.shields.io/badge/CPU-Linux%20%7C%20Windows%20%7C%20macOS-555555)](.github/workflows/ci.yml)
[![JAX](https://img.shields.io/badge/Powered_by-JAX-8A2BE2)](https://docs.jax.dev/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
[![Citation: BibTeX](https://img.shields.io/badge/Cite-BibTeX-6F4E37)](CITATION.bib)

Combine a neural backbone, a differentiable projection and a loss in one
learning workflow. OHAGI powers KKTHN, DAEHN and NLPOPTNet while leaving
framework-specific problem processing in the frontends and numerical projection
in PURIN.

- **Composable models:** shared backbone, projection and loss interfaces.
- **Flexible training:** functional JAX networks, optional Flax/Optax adapters,
  and optional Orbax checkpoints for training continuation.
- **One deployment pathway:** JAX export and `metadata.json` loading across frontends.
- **Repeated inference:** load and warm the exported signature once, then reuse it.

[Installation](#installation) Â· [Basic API](#basic-api) Â· [Load and predict](#load-and-predict) Â· [Documentation](#documentation) Â· [Contributing](#contributing) Â· [Citation](#citation)

## Installation

Requires **Python 3.10 or newer**. Install the latest compatible release from PyPI:

```shell
python -m pip install ohagi
```

To update an existing installation:

```shell
python -m pip install --upgrade ohagi
```

PURIN is installed automatically. Optional integrations are available as
`ohagi[flax,orbax,optax,plotting]`. For source/editable installs, see
[installation options](docs/INSTALL.md).

## Basic API

Train a small projected network whose outputs satisfy **u + v = b**:

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

The backbone predicts a candidate; PURIN projects it onto the constraint set;
OHAGI evaluates the loss and updates the model through the shared training engine.
Choose solver tolerances and numerical precision for the problem.

## Load and predict

A trained frontend can export its graph using
`model.export_inference("deployment/metadata.json")`. Load that bundle once:

```python
from ohagi.inference import load

predictor = load("deployment/metadata.json")
for value in [0.4, 0.7, 1.0]:
    prediction = predictor.predict([value])
```

Use inputs matching the exported signature. Keep the referenced `.jax_export`
file with `metadata.json`; metadata alone is not the complete model.
`predict` returns synchronized NumPy values, while `predict_jax` returns a JAX
array. The deployment graph does not require the originating frontend, Flax
or Orbax. Orbax training checkpoints are separate from inference bundles.

## The HardNet stack

- **[KKTHN](https://github.com/SOULS-TAMU/kkt-hardnet),
  [DAEHN](https://github.com/SOULS-TAMU/dae-hardnet) and
  [NLPOPTNet](https://github.com/SOULS-TAMU/nlpoptnet)** supply problem formulations
  and framework-specific processing.
- **[OHAGI](https://github.com/SOULS-TAMU/Ohagi)** composes models,
  runs the shared learning engine, and handles checkpoints and JAX deployment.
- **[PURIN](https://github.com/SOULS-TAMU/Purin)** owns projection
  problems, numerical solvers, sparse kernels and projection differentiation.

## Compatibility and validation

The PyPI CPU installation matrix covers **Python 3.10â€“3.15** on Linux, Windows and macOS,
plus Linux ARM64 on Python 3.13. It uses current compatible JAX releases;
regular CI additionally tests JAX 0.6.2 on Python 3.12. See the [release workflow](.github/workflows/release.yml)
for exact environments; accelerator installation follows the
[JAX guide](https://docs.jax.dev/en/latest/installation.html).

- **CI:** tests, distribution builds and installation checks.
- **Validation:** public APIs, numerical references, documentation and code quality.
- **Coverage:** at least **90% combined statement and branch coverage**; the badge
  links to the measured report for this branch.
- **Security:** CodeQL analysis and dependency review.

Optional Flax, Optax and Orbax integrations and exported inference bundles are
also exercised across Linux, Windows and macOS.

## Documentation

**[Read the full documentation](https://ohagi.readthedocs.io/en/latest/)** â€” installation, tutorials, user guide and API reference.

The full Sphinx guide follows the Read the Docs style: [contents](docs/index.rst),
[runnable examples](docs/examples/index.rst), [API reference](docs/api/index.rst)
and [build instructions](docs/BUILD.md).

| Resource | Contents |
| --- | --- |
| [Artifacts and checkpoints](docs/artifacts.md) | Model bundles, JAX deployment and training continuation |
| [Contribution guide](CONTRIBUTING.md) | Engine boundaries, development setup and validation |
| [Validation policy](.github/validation/README.md) | API, numerical and quality checks |
| [Changelog](CHANGELOG.md) | Initial release scope and changes |

## Contributing

Bug reports, examples and focused pull requests are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, numerical checks and compatibility
requirements. Keep generated results, environments and checkpoints out of Git.

## Citation

Use [CITATION.bib](CITATION.bib) for a copyable BibTeX entry or GitHub's
**Cite this repository** menu for [citation metadata](CITATION.cff).

## License

[Apache License 2.0](LICENSE). Authorship and upstream attribution are recorded
in [AUTHORS](AUTHORS) and [NOTICE](NOTICE).

[Changelog](CHANGELOG.md) Â· [Report an issue](https://github.com/SOULS-TAMU/Ohagi/issues)

Measured source structure and existing quality limits are documented in [Quality](docs/QUALITY.md).
