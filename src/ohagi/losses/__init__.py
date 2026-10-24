"""Public pure-JAX loss functions and user-facing loss specifications."""

from ohagi.losses.config import Loss
from ohagi.losses.core import configured_training_loss
from ohagi.losses.core import consistency
from ohagi.losses.core import inverse_observation
from ohagi.losses.core import latent_dual_consistency
from ohagi.losses.core import masked_mean
from ohagi.losses.core import mse
from ohagi.losses.core import problem_objective
from ohagi.losses.core import projected_output_mse
from ohagi.losses.core import projection_consistency
from ohagi.losses.core import raw_output_mse
from ohagi.losses.core import supervised_mse
from ohagi.losses.core import weighted_sum

__all__ = [
    "Loss",
    "masked_mean",
    "mse",
    "raw_output_mse",
    "projected_output_mse",
    "projection_consistency",
    "consistency",
    "latent_dual_consistency",
    "configured_training_loss",
    "supervised_mse",
    "problem_objective",
    "inverse_observation",
    "weighted_sum",
]


from .residual import residual_mse as residual_mse


def __getattr__(name):
    from ohagi.compat.plugins import public_symbol

    return public_symbol(__name__, name)
