"""Training configuration, pure steps, state containers, and user-facing trainer."""

from ohagi.nn.mlp import MLPBackbone
from purin.solvers.implicit import ImplicitProjectionLayer
from purin.solvers.implicit import make_implicit_projection_layer
from purin.solvers.types import ProjectionLayerOutput
from purin.execution.runtime import configure_cache as configure_compilation_cache
from ohagi.training.config import DualPretrainConfig
from ohagi.training.config import GradientPolicy
from ohagi.training.config import InvalidBatchPolicy
from ohagi.training.config import TrainingConfig
from ohagi.training.logging import KKTTableLogger
from ohagi.training.logging import print_postprocessing_finish
from ohagi.training.logging import print_postprocessing_start
from ohagi.training.logging import print_research_citation
from ohagi.training.schedules import ProjectionSchedule
from ohagi.training.state import AdamState
from ohagi.training.state import BuildReport
from ohagi.training.state import CompiledProjectedGraphs
from ohagi.training.state import PredictionOutput
from ohagi.training.state import ProjectedMetrics
from ohagi.training.steps import Adam
from ohagi.training.steps import apply_updates
from ohagi.training.steps import make_eval_step
from ohagi.training.steps import make_train_step

__all__ = [
    "DualPretrainConfig",
    "GradientPolicy",
    "InvalidBatchPolicy",
    "TrainingConfig",
    "make_train_step",
    "make_eval_step",
    "ProjectionSchedule",
    "KKTTableLogger",
    "print_postprocessing_start",
    "print_postprocessing_finish",
    "print_research_citation",
    "Adam",
    "AdamState",
    "BuildReport",
    "CompiledProjectedGraphs",
    "ImplicitProjectionLayer",
    "MLPBackbone",
    "PredictionOutput",
    "ProjectedMetrics",
    "ProjectedNeuralModel",
    "ProjectionLayerOutput",
    "apply_updates",
    "configure_compilation_cache",
    "make_implicit_projection_layer",
]


def __getattr__(name):
    if name == "ProjectedNeuralModel":
        from .projected import ReportedProjectedModel

        return ReportedProjectedModel
    from ohagi.compat.plugins import public_symbol

    return public_symbol(__name__, name)


from ohagi.training.engine import Engine as Engine, EpochPlan as EpochPlan
from ohagi.training.steps import (
    StepPolicy as StepPolicy,
    parameter_train_step as parameter_train_step,
)
