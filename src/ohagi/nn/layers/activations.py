"""Activation layers wrapping :mod:`jax.nn` functions."""

import jax
import jax.numpy as jnp
from ohagi.nn.layers.core import Elementwise


def Relu():
    """Create a parameter-free rectified linear activation layer."""
    return Elementwise(jax.nn.relu)


def ReLU():
    """Create a parameter-free rectified linear activation layer."""
    return Relu()


def Gelu():
    """Create a parameter-free Gaussian error linear activation layer."""
    return Elementwise(jax.nn.gelu)


def GELU():
    """Create a parameter-free Gaussian error linear activation layer."""
    return Gelu()


def Silu():
    """Create a parameter-free sigmoid-weighted linear activation layer."""
    return Elementwise(jax.nn.silu)


def SiLU():
    """Create a parameter-free sigmoid-weighted linear activation layer."""
    return Silu()


def Sigmoid():
    """Create a parameter-free logistic sigmoid activation layer."""
    return Elementwise(jax.nn.sigmoid)


def Tanh():
    """Create a parameter-free hyperbolic tangent activation layer."""
    return Elementwise(jnp.tanh)


def Softplus():
    """Create a parameter-free softplus activation layer."""
    return Elementwise(jax.nn.softplus)
