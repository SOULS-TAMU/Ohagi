"""Optional Linen MLP with shared primal and dual head construction."""

from flax import linen as nn
from .linen import mlp_forward


class ProjectedMLP(nn.Module):
    """Flax MLP with separate primal, equality and inequality output heads."""

    hidden_sizes: tuple
    primal_size: int
    equality_size: int = 0
    inequality_size: int = 0

    @nn.compact
    def __call__(self, inputs):
        return mlp_forward(
            inputs,
            hidden_sizes=self.hidden_sizes,
            primal_size=self.primal_size,
            equality_size=self.equality_size,
            inequality_size=self.inequality_size,
            separate_heads=True,
        )
