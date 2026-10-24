"""Explicit model artifacts, portable exports, and predictor loading."""

from ohagi.inference import Predictor
from ohagi.export.artifacts import ArtifactPaths
from ohagi.export.artifacts import artifact_paths
from ohagi.export.manifest import SUPPORTED_PREDICTOR_FORMATS
from ohagi.export.manifest import validate_format
from ohagi.inference import load as load_portable_predictor
from ohagi.export.weights import load_pytree
from ohagi.export.weights import save_pytree


# Older benchmark scripts import these names even without selecting native
# execution. Keep imports working, but expose only the JAX API in __all__.
def load_native_predictor(*args, **kwargs):
    """Reject the retired native predictor API and direct callers to JAX loading."""
    from ohagi.execution.inference import require_jax_backend

    require_jax_backend("native")


__all__ = [
    "ArtifactPaths",
    "artifact_paths",
    "Predictor",
    "save_predictor",
    "load_predictor",
    "save_problem",
    "load_problem",
    "save_sparse_structure",
    "load_sparse_structure",
    "symbolic_problem_spec",
    "load_symbolic_problem",
    "save_pytree",
    "load_pytree",
    "load_portable_predictor",
    "register",
    "resolve",
    "projected_mlp_factory",
    "SUPPORTED_PREDICTOR_FORMATS",
    "validate_format",
]


def __getattr__(name):
    from ohagi.compat.plugins import public_symbol

    return public_symbol(__name__, name)
