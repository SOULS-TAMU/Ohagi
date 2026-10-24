# Changelog

Notable changes to OHAGI are recorded here. This history starts with the
initial public version; internal development steps are not separate releases.
The initial release is still in preparation, so no release date is assigned.

## 0.1.0 — Unreleased

### Added

- API, numerical, frontend integration, dependency review, documentation,
  quality regression and informational performance workflows.

- Expanded numerical, training, artifact and optional-adapter regression tests,
  with an inclusive 85% combined statement-and-branch coverage gate.

- Cross-platform CPU CI for Linux, Windows and Apple Silicon macOS, with
  baseline/current JAX, runtime configuration artifacts and ARM64 coverage.

- GitHub Actions CI for numerical tests, package checks and Ruff, with measured
  branch coverage reporting and scheduled Python CodeQL analysis.

- A shared JAX learning engine for the KKTHN, DAEHN and NLPOPTNet frontends,
  with framework-specific problem definitions supplied by those packages.
- Projected neural models using PURIN's projection algorithms and kernels.
- Functional JAX backbones and optional Flax backbone adapters.
- Shared training, optimizer and batching utilities, with optional Optax
  integration and Orbax training checkpoints.
- A shared JAX model bundle for export and loading through `metadata.json`,
  supporting repeated single-instance inference.
- Python package metadata, an executable quick start, model artifact
  documentation and contribution guidance.
- Apache-2.0 licensing and machine-readable software citation metadata.

### Fixed

- Preserve training-state buffers when discarding a donated Flax warmup update.
