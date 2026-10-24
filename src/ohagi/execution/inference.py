"""Shared policy for the JAX-only inference and export interface."""


def require_jax_backend(value=None):
    """Validate that the requested inference backend is JAX."""
    name = "jax" if value is None else str(value).strip().lower()
    if name not in {"jax", "auto"}:
        raise ValueError("Only JAX inference is supported; use backend='jax'.")
    return "jax"


def require_jax_preference(value):
    """Validate an inference preference against the supported JAX pathway."""
    name = str(value).strip().lower()
    if name not in {"jax", "auto", "json", "portable", "fastest"}:
        raise ValueError("Only JAX loading is supported; use preference='jax'.")
    return name
