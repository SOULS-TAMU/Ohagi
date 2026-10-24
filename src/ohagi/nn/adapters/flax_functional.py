"""Optional Linen networks implementing the existing two-function interface.

These networks are deterministic and params-only. Stateful Linen modules belong
in the explicit-state ``FlaxLinenBackbone``/OHAGI Trainer interface instead.
"""

from dataclasses import dataclass
import jax.numpy as jnp


def _linen():
    try:
        from flax import linen
    except ImportError as error:
        raise ImportError("Install ohagi[flax] to use Flax backbones.") from error
    return linen


@dataclass(frozen=True)
class FlaxFunctionalBackbone:
    """Bridge a custom params-only Linen module to KKTHN/DAEHN callbacks.

    The module must accept inputs without training flags or RNGs. Dropout and
    mutable collections are deliberately unsupported by this legacy contract.
    Use ``init`` and ``apply`` as the existing model's backbone callbacks.
    """

    module: object

    def init(self, key, input_dim, output_dim, dtype=jnp.float64, **kwargs):
        """Initialize this backbone for the requested input/output dimensions and dtype."""
        _linen()
        if kwargs.get("n_equality_duals", 0) or kwargs.get("n_inequality_duals", 0):
            raise ValueError(
                "Custom Flax callbacks require explicit dual-head initialization; use MLP(backend='flax') for automatic dual heads."
            )
        variables = self.module.init(key, jnp.zeros((1, input_dim), dtype))
        if set(variables) - {"params"}:
            raise ValueError(
                "Mutable Flax collections require the explicit-state OHAGI Trainer interface."
            )
        params = variables.get("params", {})
        result = self.apply(params, jnp.zeros((1, input_dim), dtype))
        if result.shape != (1, output_dim):
            raise ValueError(
                f"Flax output must have shape (1, {output_dim}); got {result.shape}."
            )
        return params

    def apply(self, parameters, inputs):
        """Evaluate this backbone with the supplied parameter tree and input arrays."""
        return self.module.apply({"params": parameters}, inputs)


@dataclass(frozen=True)
class FlaxMLPBackbone:
    """Linen MLP with identical OHAGI initialization, including dual heads."""

    hidden_sizes: tuple = (64, 64)
    activation: object = jnp.tanh

    def _module(self, output_dim):
        nn = _linen()
        activation = self.activation

        class Network(nn.Module):
            @nn.compact
            def __call__(self, inputs):
                from .linen import mlp_forward

                return mlp_forward(
                    inputs,
                    hidden_sizes=self_hidden,
                    primal_size=output_dim,
                    activation=activation,
                    name_prefix="layer_",
                    dtype=inputs.dtype,
                    param_dtype=inputs.dtype,
                )

        self_hidden = self.hidden_sizes
        return Network()

    def init(self, key, input_dim, output_dim, dtype=jnp.float64, **kwargs):
        """Initialize this backbone for the requested input/output dimensions and dtype."""
        _linen()
        from ohagi.nn.mlp import MLPBackbone

        original = MLPBackbone(self.hidden_sizes, self.activation).init(
            key, input_dim, output_dim, dtype, **kwargs
        )
        return {
            f"layer_{i}": {"kernel": layer["W"], "bias": layer["b"]}
            for i, layer in enumerate(original)
        }

    def apply(self, parameters, inputs):
        """Evaluate this backbone with the supplied parameter tree and input arrays."""
        size = parameters[f"layer_{len(self.hidden_sizes)}"]["bias"].shape[0]
        return self._module(size).apply({"params": parameters}, inputs)
