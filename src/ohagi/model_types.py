"""Task-type declarations shared by the high-level frontend."""

from __future__ import annotations
from enum import Enum


class ModelType(str, Enum):
    """Supported model families used when normalizing frontend configuration."""

    SURROGATE = "surrogate"
    PARAMETER_ESTIMATION = "parameter_estimation"
    OPTIMIZATION = "optimization"

    @classmethod
    def normalize(cls, value):
        """Resolve a model-family name or enum value to its canonical member."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "simulation": cls.SURROGATE,
            "data_fitting": cls.SURROGATE,
            "estimate": cls.PARAMETER_ESTIMATION,
            "parameter_estimate": cls.PARAMETER_ESTIMATION,
            "optimize": cls.OPTIMIZATION,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as error:
            raise ValueError(
                "type must be surrogate, parameter_estimation, or optimization."
            ) from error
