"""Optional Flax Linen adapter; importing OHAGI does not import Flax."""

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class FlaxLinenBackbone:
    """Adapt a Flax Linen module to OHAGI's stateful backbone protocol."""

    module: object
    mode_keyword: str | None = "training"

    def init(self, key, input_dim, output_dim, dtype):
        """Initialize this backbone for the requested input/output dimensions and dtype."""
        kwargs = {} if self.mode_keyword is None else {self.mode_keyword: False}
        variables = self.module.init(
            {"params": key, "dropout": key}, jnp.zeros((1, input_dim), dtype), **kwargs
        )
        return variables["params"], {
            k: v for k, v in variables.items() if k != "params"
        }

    def apply(self, parameters, state, inputs, *, key=None, training=False):
        """Evaluate this backbone with the supplied parameter tree and input arrays."""
        kwargs = {} if self.mode_keyword is None else {self.mode_keyword: training}
        variables = {"params": parameters, **state}
        rngs = None if key is None else {"dropout": key}
        if training and state:
            output, new_state = self.module.apply(
                variables, inputs, rngs=rngs, mutable=list(state), **kwargs
            )
            return output, new_state
        return self.module.apply(variables, inputs, rngs=rngs, **kwargs), state
