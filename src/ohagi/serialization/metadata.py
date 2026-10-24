"""Shared model metadata validation independent of export and frontend code."""

from pathlib import Path
import json

FORMAT = "ohagi_model_v1"


def read_model(metadata_path, *, framework=None):
    """Read and validate metadata for a shared OHAGI model bundle."""
    path = Path(metadata_path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError("Expected an OHAGI JAX model bundle.")
    if framework is not None and data.get("framework") != framework:
        raise ValueError("The model bundle belongs to a different frontend.")
    return data


read_model.__module__ = "ohagi.serialization.bundle"
