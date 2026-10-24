"""OHAGI: composable constrained learning in JAX.

Precision is controlled by the application. Legacy frontends retain their
historical precision settings; importing the common package is lightweight.
"""

from importlib import import_module
from ohagi.compat import install_aliases
from ohagi.compat import plugins as _plugins
from ohagi._projection_aliases import ALIASES as _PROJECTION_ALIASES

install_aliases(_PROJECTION_ALIASES)
__version__ = "0.1.0"
_EXPORTS = {
    "Problem": ("problems", "Problem"),
    "Projection": ("problems", "Projection"),
    "FunctionalProblem": ("problems", "FunctionalProblem"),
    "SymbolicProblem": ("problems", "SymbolicProblem"),
    "MatrixProblem": ("problems", "MatrixProblem"),
    "ParametricQP": ("problems", "ParametricQP"),
    "ParametricLP": ("problems", "ParametricLP"),
    "MLP": ("nn", "MLP"),
    "MLPBackbone": ("nn", "MLPBackbone"),
    "Loss": ("losses", "Loss"),
    "prepare_projection": ("solvers", "prepare_projection"),
    "ChambollePockConfig": ("solvers", "ChambollePockConfig"),
    "ADMMConfig": ("solvers", "ADMMConfig"),
    "FischerBurmeisterConfig": ("solvers", "FischerBurmeisterConfig"),
    "ProjectionConfig": ("solvers", "ProjectionConfig"),
    "LinearEqualityConfig": ("solvers", "LinearEqualityConfig"),
    "LinearSolverConfig": ("solvers", "LinearSolverConfig"),
    "ImplicitGradientConfig": ("solvers", "ImplicitGradientConfig"),
    "SolveStatus": ("solvers", "SolveStatus"),
    "Distance": ("projections", "Distance"),
    "WeightedDistance": ("projections", "WeightedDistance"),
    "LocalQuadratic": ("projections", "LocalQuadratic"),
    "Sequential": ("projections", "Sequential"),
    "ProjectedModel": ("models", "ProjectedModel"),
    "Trainer": ("training.engine", "Trainer"),
    "Adam": ("optim", "Adam"),
    "save_predictor": ("export", "save_predictor"),
    "load_predictor": ("export", "load_predictor"),
    "load_inference": ("inference", "load"),
    "export_inference": ("inference", "export"),
}
__all__ = list(dict.fromkeys([*_EXPORTS, *_plugins.public_symbols(__name__)]))


def __getattr__(name):
    if name in _EXPORTS:
        module, symbol = _EXPORTS[name]
        value = getattr(import_module("ohagi." + module), symbol)
    else:
        try:
            return _plugins.public_symbol(__name__, name)
        except AttributeError:
            pass
        try:
            value = import_module("ohagi." + name)
        except ModuleNotFoundError as error:
            if error.name != "ohagi." + name:
                raise
            raise AttributeError(name) from error
    globals()[name] = value
    return value
