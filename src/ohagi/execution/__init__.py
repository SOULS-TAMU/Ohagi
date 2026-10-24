"""Device placement, dtype resolution and prepared JAX execution services."""

from .runtime import (
    resolve_device as resolve_device,
    resolve_jax_dtype as resolve_jax_dtype,
    configure_cache as configure_cache,
)
