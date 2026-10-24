"""Functional backbone contract with explicit nontrainable state and RNG."""

from dataclasses import dataclass
from typing import Protocol


class Backbone(Protocol):
    """Initialization and application protocol for a stateful JAX backbone."""

    def init(self, key, input_dim, output_dim, dtype):
        """Initialize this backbone for the requested input/output dimensions and dtype."""
        ...

    def apply(self, parameters, state, inputs, *, key=None, training=False):
        """Evaluate this backbone with the supplied parameter tree and input arrays."""
        ...


@dataclass(frozen=True)
class FunctionalBackbone:
    """Adapt custom JAX functions to the shared backbone protocol.

    Attributes:
        init_fn: Callable accepting ``(key, input_dim, output_dim, dtype)``.
            Returns params when stateless, or ``(params, state)`` when stateful.
        apply_fn: Stateless callable ``(params, inputs) -> outputs``, or a
            stateful callable ``(params, state, inputs, key=..., training=...)``
            returning ``(outputs, next_state)``.
        stateful: Select the stateful signatures when True. Otherwise the
            adapter creates and carries an empty nontrainable state mapping.

    Notes:
        Callables must remain compatible with the JAX transforms used by the
        trainer. This adapter supports custom architectures without changing
        frontend problem definitions or the projection engine.
    """

    init_fn: object
    apply_fn: object
    stateful: bool = False

    def init(self, key, input_dim, output_dim, dtype):
        """Initialize this backbone for the requested input/output dimensions and dtype."""
        result = self.init_fn(key, input_dim, output_dim, dtype)
        return result if self.stateful else (result, {})

    def apply(self, parameters, state, inputs, *, key=None, training=False):
        """Evaluate this backbone with the supplied parameter tree and input arrays."""
        if self.stateful:
            return self.apply_fn(parameters, state, inputs, key=key, training=training)
        return self.apply_fn(parameters, inputs), state


@dataclass(frozen=True)
class MLPAdapter:
    """Adapt a functional layer pair to the shared backbone protocol."""

    backbone: object

    def init(self, key, input_dim, output_dim, dtype):
        """Initialize this backbone for the requested input/output dimensions and dtype."""
        return self.backbone.init(key, input_dim, output_dim, dtype), {}

    def apply(self, parameters, state, inputs, *, key=None, training=False):
        """Evaluate this backbone with the supplied parameter tree and input arrays."""
        return self.backbone.apply(parameters, inputs), state
