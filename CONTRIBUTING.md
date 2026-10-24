# Contributing to OHAGI

Thank you for helping improve OHAGI. Bug reports, documentation fixes,
examples and numerical improvements are welcome. This guide accompanies
the first public version of the project.

## Before you start

For a small fix, open a focused pull request. For a new public API, dependency
or change to numerical behavior, open an issue first so the design can be
discussed before implementation.

Bug reports should include a minimal reproducer, the expected and observed
behavior, package versions, device, precision settings and random seed.
Include the traceback when an exception occurs. Remove private data and
credentials from examples and logs.

## Development setup

Use Python 3.10 or newer in an isolated environment. With sibling OHAGI and
PURIN source checkouts, run the following from their parent directory:

```sh
python -m pip install -e ./purin -e "./ohagi[dev]"
```

Install optional features only when needed:

```sh
python -m pip install -e "./ohagi[flax,orbax,optax,plotting]"
```

For frontend integration work, also install the relevant KKTHN, DAEHN or
NLPOPTNet checkout in editable mode. Follow the
[JAX installation instructions](https://docs.jax.dev/en/latest/installation.html)
for your device. See the [README](README.md) for a runnable quick start.

## Keep responsibilities clear

OHAGI owns the shared learning engine: backbone adapters, training, model
export and metadata-driven inference. PURIN owns projection algorithms and
numerical kernels. KKTHN, DAEHN and NLPOPTNet own their problem definitions
and framework-specific policies.

Keep these dependencies directed toward the shared libraries. Do not add
frontend imports to OHAGI or copy projection solvers into it. Reuse the
shared engine instead of introducing a separate training or inference path.

## Code and documentation

Use descriptive names, small functions and Google-style docstrings for
public APIs. Document array shapes, coordinate conventions, residual signs
and gradient behavior where they affect callers. Explain mathematical
choices and invariants in comments; avoid restating the code.

Run these checks from the OHAGI directory:

```sh
python -m ruff format --check src
python -m ruff check --select F,E9 src
```

Use `python -m ruff format src` to apply formatting. Keep unrelated formatting
changes out of a pull request, and leave benchmark scripts unchanged unless
the proposed change specifically concerns them.

## Validate your change

Include a focused regression test or a reproducible validation script with
the pull request. Record the commands, environment and results so another
contributor can repeat the checks. Run the README quick start for changes
that affect installation or the public workflow.

Match validation to the affected behavior:

| Change | Required comparison |
| --- | --- |
| Numerical or projection interfaces | Outputs, parameter gradients and constraint residuals against the previous implementation. |
| Training or checkpoint handling | Training histories and parameters, including resumed versus uninterrupted training. |
| Export or inference | Export, reload through `metadata.json` in a fresh process, and repeated single-instance predictions. |
| Public interfaces | Signatures, aliases and the relevant unchanged frontend benchmark scripts. |
| Restructuring or documentation | Confirm executable behavior is unchanged; check examples and links. |

Use identical inputs, seeds, JAX versions, devices and precision settings for
numerical comparisons. Report tolerances explicitly and investigate any
difference. Separate intentional numerical changes from restructuring.
For inference timing, warm up compilation and synchronize device execution.

The [model artifact guide](docs/artifacts.md) describes inference bundles
and training checkpoints. Preserve their documented contracts.

## Packaging checks

From the OHAGI directory, build and validate distributions:

```sh
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Install the built wheel in a separate environment with PURIN available and
exercise the affected public workflow. Do not commit environments, caches,
build output, generated results or checkpoints.

## Pull request checklist

- Explain the problem and the resulting behavior.
- Keep the change focused and identify any compatibility implications.
- Include reproducible validation results and numerical tolerances.
- Update documentation and the appropriate unreleased changelog entry.
- Preserve benchmark APIs and metadata-driven inference unless an API change
  has been explicitly agreed.

Contributions are provided under the project's [Apache-2.0 license](LICENSE).
Preserve relevant attribution in [AUTHORS](AUTHORS) and [NOTICE](NOTICE).
Keep [CITATION.bib](CITATION.bib) and [CITATION.cff](CITATION.cff) consistent.
The CFF metadata supports GitHub's citation panel; BibTeX remains available directly.

## Automated checks

CI runs the full numerical suite on CPU across these configurations:

| Platform | Python | JAX configuration |
| --- | --- | --- |
| Linux x86-64, Windows x86-64, macOS Apple Silicon | 3.10–3.14 | Newest compatible dependencies for each interpreter. |
| Linux x86-64, Windows x86-64, macOS Apple Silicon | 3.12 | JAX/JAXlib 0.6.2 baseline. |
| Linux ARM64 | 3.13 | Newest compatible dependencies. |

The configuration tests additionally exercise float32/float64, eager/JIT
execution and batch sizes one and four for the equality projection and
functional backbone interfaces. Full solver regression tests use float64.
Ruff runs on Linux; wheel and source-distribution builds and installation
checks run on Linux, Windows and macOS. Each core test job uploads its OS,
architecture, Python, package versions and JAX devices as a diagnostic artifact.
The test suite is in `tests/`. CPU results do not establish GPU/TPU support.

Flax, Optax and Orbax have separate integration jobs on all three OSes.
Another job exports float32/float64 inference bundles on Linux and reloads
the same artifacts on Linux, Windows and macOS using the exporter's JAX
version. It exercises repeated single-instance inference from `metadata.json`.

Install the test dependencies with `python -m pip install ".[test]"`, then run:

```sh
python -m pytest -q --ignore=tests/test_optional.py --cov=ohagi --cov-branch --cov-report=term-missing --cov-report=xml --cov-report=json
```

Coverage measures the entire package, including unexecuted modules. The badge
combines statement and branch coverage from the Linux x86-64 / Python 3.12 / current-JAX core test run.
Reports are retained as workflow artifacts; the badge and measured totals are
published to `codex/coverage` after successful trusted pushes to `main`.
Pull requests run checks with read-only repository access and do not publish
coverage. No third-party coverage account or additional secret is required.

CodeQL runs extended Python security analysis on pushes, pull requests and a
weekly schedule. Its badge reports workflow execution status; inspect code
scanning alerts separately. GitHub Actions dependencies are pinned to commits
and monitored by Dependabot.

The initial combined coverage floor is 24%, based on the measured baseline.
Raise this floor as coverage improves; do not omit modules to improve the score.
