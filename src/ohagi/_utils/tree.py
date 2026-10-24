"""Small PyTree predicates and selections used by training steps."""

from __future__ import annotations
import jax
import jax.numpy as jnp


def tree_all_finite(tree):
    """Return a scalar JAX predicate indicating whether every pytree leaf is finite."""
    leaves = jax.tree_util.tree_leaves(tree)
    result = jnp.asarray(True)
    for leaf in leaves:
        result = result & jnp.all(jnp.isfinite(leaf))
    return result


def tree_select(condition, true_tree, false_tree):
    """Select corresponding pytree leaves with a scalar or broadcastable predicate."""
    return jax.tree_util.tree_map(
        lambda true_value, false_value: jnp.where(condition, true_value, false_value),
        true_tree,
        false_tree,
    )
