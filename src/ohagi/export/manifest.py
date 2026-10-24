"""Manifest validation for explicit predictor artifact formats."""

from __future__ import annotations

SUPPORTED_PREDICTOR_FORMATS = {"ohagi_model_v1"}


def validate_format(manifest: dict, supported=SUPPORTED_PREDICTOR_FORMATS):
    """Reject metadata whose format differs from the expected schema identifier."""
    value = manifest.get("format")
    if value not in supported:
        raise ValueError(
            f"unsupported artifact format {value!r}; expected one of {sorted(supported)}"
        )
    return value
