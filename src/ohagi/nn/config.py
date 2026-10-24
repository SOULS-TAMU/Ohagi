"""User-facing neural-network architecture specifications."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import jax
import jax.numpy as jnp
from ohagi.nn.mlp import MLPBackbone

_ACTIVATIONS = {
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "softplus": jax.nn.softplus,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
    "swish": jax.nn.swish,
}


def resolve_activation(value: Any):
    """Resolve a supported activation name or preserve a supplied callable."""
    if callable(value):
        return value
    name = str(value).strip().lower()
    try:
        return _ACTIVATIONS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown activation {value!r}; choose one of {sorted(_ACTIVATIONS)}."
        ) from error


def boxed_relu(values, lower=0.0, upper=None):
    """Map latent values through a ReLU box.

    The three-argument form ``boxed_relu(z, lower, upper)`` supports scalar,
    vector, and batched bounds. Infinite bounds mean that side is unavailable.

    For a two-sided finite box ``[lower, upper]`` the transform is

    ``lower + min(relu(z), upper - lower)``.

    The one-sided cases are the natural extensions:

    * lower only: ``lower + relu(z)``
    * upper only: ``upper - relu(-z)``
    * no bounds: ``z``

    For backward compatibility, ``boxed_relu(z, gamma)`` is interpreted as
    ``boxed_relu(z, 0, gamma)``.
    """

    if upper is None:
        upper = lower
        lower = 0.0

    z = jnp.asarray(values)
    lower_array = jnp.asarray(lower, dtype=z.dtype)
    upper_array = jnp.asarray(upper, dtype=z.dtype)

    has_lower = jnp.isfinite(lower_array)
    has_upper = jnp.isfinite(upper_array)

    # Avoid carrying +/-inf through arithmetic in branches that will not be
    # selected by jnp.where. This is important for stable JAX gradients.
    lower_safe = jnp.where(has_lower, lower_array, jnp.asarray(0.0, dtype=z.dtype))
    upper_safe = jnp.where(has_upper, upper_array, jnp.asarray(0.0, dtype=z.dtype))

    both = lower_safe + jnp.minimum(jax.nn.relu(z), upper_safe - lower_safe)
    lower_only = lower_safe + jax.nn.relu(z)
    upper_only = upper_safe - jax.nn.relu(-z)

    result = z
    result = jnp.where(has_lower & ~has_upper, lower_only, result)
    result = jnp.where(~has_lower & has_upper, upper_only, result)
    result = jnp.where(has_lower & has_upper, both, result)
    return result


@dataclass(frozen=True)
class MLP:
    """Shared-trunk MLP architecture specification.

    The MLP owns only architecture construction and output sizing. Its final
    linear layer produces the raw concatenated vector

    ``(y_raw, lambda_eq_raw, mu_ineq_raw)``.

    Optimization-specific output splitting and transformations are handled by
    the training/model layers rather than by this architecture object.
    """

    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    predict_duals: bool = True
    inequality_dual_floor: float = 1e-8
    backend: str = "jax"

    def __post_init__(self):
        if self.backend not in {"jax", "flax"}:
            raise ValueError("MLP backend must be 'jax' or 'flax'.")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer size.")
        if any(int(size) < 1 for size in self.hidden_sizes):
            raise ValueError("All hidden layer sizes must be positive.")
        if float(self.inequality_dual_floor) < 0.0:
            raise ValueError("inequality_dual_floor must be nonnegative.")
        resolve_activation(self.activation)

    def build(self) -> MLPBackbone:
        """Construct the configured functional JAX or optional Flax MLP backbone."""
        if self.backend == "flax":
            from ohagi.nn.adapters.flax_functional import FlaxMLPBackbone

            return FlaxMLPBackbone(
                tuple(map(int, self.hidden_sizes)), resolve_activation(self.activation)
            )
        return MLPBackbone(
            hidden_sizes=tuple(int(value) for value in self.hidden_sizes),
            activation=resolve_activation(self.activation),
        )

    def output_size(self, *, n_y: int, n_eq: int, n_ineq: int) -> int:
        """Compute primal-plus-multiplier output width from the problem dimensions."""
        if not self.predict_duals:
            return int(n_y)
        return int(n_y + n_eq + n_ineq)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable network architecture configuration."""
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        if self.backend == "jax":
            payload.pop("backend")  # Preserve historical benchmark manifests.
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MLP":
        """Reconstruct an MLP specification from its configuration mapping."""
        values = dict(payload)
        values["hidden_sizes"] = tuple(values.get("hidden_sizes", (64, 64)))
        return cls(**values)
