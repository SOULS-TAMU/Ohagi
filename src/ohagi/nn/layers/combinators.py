"""Functional layer combinators."""

from __future__ import annotations
import jax
from purin._utils.validation import strict_zip
from ohagi.nn.layers.core import Layer


def Serial(*layers: Layer) -> Layer:
    """Compose functional layers in order, passing each output to the next layer."""
    layers = tuple(layers)

    def init(key, input_shape):
        if not layers:
            return tuple(input_shape), ()
        keys = jax.random.split(key, len(layers))
        shape = tuple(input_shape)
        params = []
        for layer, subkey in strict_zip(layers, keys):
            shape, layer_params = layer.init(subkey, shape)
            params.append(layer_params)
        return shape, tuple(params)

    def apply(params, inputs, **kwargs):
        values = inputs
        for layer, layer_params in strict_zip(layers, params):
            values = layer.apply(layer_params, values, **kwargs)
        return values

    return Layer(init, apply)


def Residual(layer: Layer) -> Layer:
    """Wrap a functional layer with an additive input-to-output skip connection."""

    def init(key, input_shape):
        output_shape, params = layer.init(key, input_shape)
        if tuple(output_shape) != tuple(input_shape):
            raise ValueError("Residual layer must preserve shape")
        return tuple(output_shape), params

    def apply(params, inputs, **kwargs):
        return inputs + layer.apply(params, inputs, **kwargs)

    return Layer(init, apply)


def Parallel(*layers: Layer) -> Layer:
    """Apply each functional branch to the same input and return all outputs."""
    layers = tuple(layers)

    def init(key, input_shape):
        keys = jax.random.split(key, len(layers))
        shapes, params = [], []
        for layer, subkey in strict_zip(layers, keys):
            shape, p = layer.init(subkey, input_shape)
            shapes.append(shape)
            params.append(p)
        return tuple(shapes), tuple(params)

    def apply(params, inputs, **kwargs):
        return tuple(
            layer.apply(p, inputs, **kwargs) for layer, p in strict_zip(layers, params)
        )

    return Layer(init, apply)


def FanOut(count: int) -> Layer:
    """Replicate one input into the requested number of branches."""
    count = int(count)
    if count < 1:
        raise ValueError("count must be positive")

    def init(_key, input_shape):
        return tuple(tuple(input_shape) for _ in range(count)), ()

    def apply(_params, inputs, **_kwargs):
        return tuple(inputs for _ in range(count))

    return Layer(init, apply)


def FanInSum() -> Layer:
    """Combine input branches by elementwise summation."""

    def init(_key, input_shapes):
        shapes = tuple(tuple(s) for s in input_shapes)
        if not shapes or any(s != shapes[0] for s in shapes):
            raise ValueError("FanInSum input shapes must match")
        return shapes[0], ()

    def apply(_params, inputs, **_kwargs):
        first, *rest = inputs
        out = first
        for item in rest:
            out = out + item
        return out

    return Layer(init, apply)
