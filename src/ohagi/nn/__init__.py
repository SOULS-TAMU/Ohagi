"""Neural-network functions and architecture specifications."""

from jax.nn import (
    celu,
    elu,
    gelu,
    glu,
    hard_sigmoid,
    hard_silu,
    hard_swish,
    initializers,
    leaky_relu,
    log_sigmoid,
    log_softmax,
    one_hot,
    relu,
    relu6,
    selu,
    sigmoid,
    silu,
    soft_sign,
    softmax,
    softplus,
    standardize,
    swish,
)
from ohagi.nn import layers
from ohagi.nn.config import MLP
from ohagi.nn.config import resolve_activation
from ohagi.nn.mlp import MLPBackbone

__all__ = [
    "celu",
    "elu",
    "gelu",
    "glu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "one_hot",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "soft_sign",
    "softmax",
    "softplus",
    "standardize",
    "swish",
    "initializers",
    "layers",
    "MLPBackbone",
    "MLP",
    "resolve_activation",
]

from .backbone import (
    Backbone as Backbone,
    FunctionalBackbone as FunctionalBackbone,
    MLPAdapter as MLPAdapter,
)
from .adapters import FlaxLinenBackbone, FlaxFunctionalBackbone, FlaxMLPBackbone

__all__ += ["FlaxLinenBackbone", "FlaxFunctionalBackbone", "FlaxMLPBackbone"]
