"""Functional neural-network layers."""

from ohagi.nn.layers.activations import GELU
from ohagi.nn.layers.activations import Gelu
from ohagi.nn.layers.activations import ReLU
from ohagi.nn.layers.activations import Relu
from ohagi.nn.layers.activations import Sigmoid
from ohagi.nn.layers.activations import SiLU
from ohagi.nn.layers.activations import Silu
from ohagi.nn.layers.activations import Softplus
from ohagi.nn.layers.activations import Tanh
from ohagi.nn.layers.combinators import FanInSum
from ohagi.nn.layers.combinators import FanOut
from ohagi.nn.layers.combinators import Parallel
from ohagi.nn.layers.combinators import Residual
from ohagi.nn.layers.combinators import Serial
from ohagi.nn.layers.core import Dropout
from ohagi.nn.layers.core import Elementwise
from ohagi.nn.layers.core import Identity
from ohagi.nn.layers.core import Lambda
from ohagi.nn.layers.core import Layer
from ohagi.nn.layers.core import LayerNorm
from ohagi.nn.layers.linear import Dense


def MLP(output_size, hidden_sizes=(64, 64), *, activation=Relu, **dense_kwargs):
    """Compose dense layers and activations into a functional multilayer perceptron."""
    layers = []
    for width in hidden_sizes:
        layers.extend((Dense(width, **dense_kwargs), activation()))
    layers.append(Dense(output_size, **dense_kwargs))
    return Serial(*layers)


__all__ = [
    "Layer",
    "Dense",
    "Identity",
    "Lambda",
    "Elementwise",
    "Dropout",
    "LayerNorm",
    "Relu",
    "ReLU",
    "Gelu",
    "GELU",
    "Silu",
    "SiLU",
    "Sigmoid",
    "Tanh",
    "Softplus",
    "Serial",
    "Residual",
    "Parallel",
    "FanOut",
    "FanInSum",
    "MLP",
]
