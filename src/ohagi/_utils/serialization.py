"""Shared JSON conversion, checksums, and atomic artifact writes."""

from __future__ import annotations
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
import jax
import numpy as np


def json_safe(value):
    """Convert nested numerical values into JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(jax.device_get(value)).tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256(path):
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, data):
    """Replace a file atomically using a temporary file in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    """Atomically write a deterministic, human-readable JSON document."""
    data = json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n"
    atomic_write(path, data.encode("utf-8"))


def timestamp():
    """Compact filesystem-friendly run timestamp."""
    import time

    return time.strftime("%Y%m%d_%H%M%S")


def resolve_path(path):
    """Expand and resolve a filesystem path for artifact access."""
    return Path(path).expanduser().resolve()
