"""Pure-JAX multilayer perceptron initialization and application."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import jax
import jax.numpy as jnp
import numpy as np
from purin._utils.validation import strict_zip


def _inverse_softplus(value: float) -> float:
    """Stable scalar inverse of softplus for a strictly positive value."""

    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inverse-softplus input must be finite and positive.")
    # log(expm1(value)) is accurate for the small gamma values used here.
    return float(np.log(np.expm1(value)))


@dataclass(frozen=True)
class MLPBackbone:
    """Functional MLP backbone with Glorot-uniform initialization.

    ``init`` accepts optional output-partition metadata. When latent dual heads
    are present, their final-layer weights are initialized to zero. Equality
    dual biases start at zero. Inequality raw biases are initialized so that
    the projection-phase ``softplus(raw) + floor`` value equals
    ``dual_initial_gamma``. This prevents a zero raw bias from becoming
    ``softplus(0) ~= 0.693`` when the pretraining box is removed.
    """

    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: Callable = jax.nn.relu

    def init(
        self,
        key,
        input_dim: int,
        output_dim: int,
        dtype: Any = jnp.float64,
        *,
        n_primal_outputs: int | None = None,
        n_equality_duals: int = 0,
        n_inequality_duals: int = 0,
        dual_initial_gamma: float = 1e-3,
        inequality_dual_floor: float = 1e-8,
    ):
        """Initialize this backbone for the requested input/output dimensions and dtype."""
        sizes = (int(input_dim), *map(int, self.hidden_sizes), int(output_dim))
        keys = jax.random.split(key, len(sizes) - 1)
        layers = []
        for layer_index, (layer_key, fan_in, fan_out) in enumerate(
            strict_zip(keys, sizes[:-1], sizes[1:])
        ):
            limit = np.sqrt(6.0 / float(fan_in + fan_out))
            weights = jax.random.uniform(
                layer_key, (fan_in, fan_out), minval=-limit, maxval=limit, dtype=dtype
            )
            bias = jnp.zeros((fan_out,), dtype=dtype)

            is_final = layer_index + 1 == len(sizes) - 1
            if is_final and n_primal_outputs is not None:
                n_primal = int(n_primal_outputs)
                n_eq = int(n_equality_duals)
                n_ineq = int(n_inequality_duals)
                if min(n_primal, n_eq, n_ineq) < 0:
                    raise ValueError("Output partition sizes must be nonnegative.")
                if n_primal + n_eq + n_ineq != int(output_dim):
                    raise ValueError(
                        "Output partition does not match output_dim: "
                        f"{n_primal}+{n_eq}+{n_ineq}!={int(output_dim)}."
                    )
                gamma = float(dual_initial_gamma)
                floor = float(inequality_dual_floor)
                if not math.isfinite(gamma) or gamma <= 0.0:
                    raise ValueError("dual_initial_gamma must be finite and positive.")
                if not math.isfinite(floor) or floor < 0.0:
                    raise ValueError(
                        "inequality_dual_floor must be finite and nonnegative."
                    )
                if n_ineq and gamma <= floor:
                    raise ValueError(
                        "dual_initial_gamma must exceed inequality_dual_floor "
                        "when inequality duals are predicted."
                    )

                dual_start = n_primal
                if dual_start < int(output_dim):
                    weights = weights.at[:, dual_start:].set(
                        jnp.asarray(0.0, dtype=dtype)
                    )

                inequality_start = n_primal + n_eq
                if n_ineq:
                    raw_inequality_bias = _inverse_softplus(gamma - floor)
                    bias = bias.at[inequality_start : inequality_start + n_ineq].set(
                        jnp.asarray(raw_inequality_bias, dtype=dtype)
                    )

            layers.append({"W": weights, "b": bias})
        return tuple(layers)

    def apply(self, parameters: Sequence[dict], inputs):
        """Evaluate this backbone with the supplied parameter tree and input arrays."""
        values = inputs
        for index, layer in enumerate(parameters):
            values = values @ layer["W"] + layer["b"]
            if index + 1 < len(parameters):
                values = self.activation(values)
        return values
