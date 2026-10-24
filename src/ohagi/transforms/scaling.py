"""Explicit coordinate transforms; multipliers have separate row units."""

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class DiagonalCoordinates:
    """Diagonal coordinate transform between normalized and physical variables."""

    scale: object
    offset: object = 0.0

    def __post_init__(self):
        scale = np.asarray(self.scale)
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("coordinate scales must be positive and finite")

    def normalize(self, physical):
        """Map physical values into normalized coordinates."""
        return (physical - jnp.asarray(self.offset)) / jnp.asarray(self.scale)

    def physical(self, normalized):
        """Map normalized values back to physical coordinates."""
        return normalized * jnp.asarray(self.scale) + jnp.asarray(self.offset)

    def physical_gradient(self, normalized_gradient):
        """Transform a normalized-coordinate gradient into physical coordinates."""
        return normalized_gradient / jnp.asarray(self.scale)

    @staticmethod
    def physical_multipliers(scaled_multipliers, row_scale):
        """Recover physical constraint multipliers from scaled multipliers."""
        return scaled_multipliers * row_scale


@dataclass(frozen=True)
class Standardizer:
    """Feature-wise mean and scale used for standardization and its inverse."""

    mean: object
    scale: object

    @classmethod
    def fit(cls, training_data):
        """Estimate feature statistics from the supplied observations."""
        data = np.asarray(training_data)
        if data.ndim != 2 or not len(data) or not np.all(np.isfinite(data)):
            raise ValueError("training data must be a finite nonempty matrix")
        scale = data.std(axis=0)
        return cls(data.mean(axis=0), np.where(scale > 1e-12, scale, 1.0))

    def transform(self, values):
        """Standardize observations using the stored mean and scale."""
        return (values - jnp.asarray(self.mean)) / jnp.asarray(self.scale)

    def inverse(self, values):
        """Restore physical observations from standardized values."""
        return values * jnp.asarray(self.scale) + jnp.asarray(self.mean)
