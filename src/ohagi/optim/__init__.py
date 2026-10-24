"""Optax-compatible optimizer contract with a dependency-free Adam default."""

from ohagi.training.steps import Adam as Adam, apply_updates as apply_updates
from ohagi.training.state import AdamState as AdamState
