"""Private JAX export encoding; JAX uses StableHLO internally."""

from typing import Callable as Callable, Any as Any, Sequence as Sequence
from types import SimpleNamespace
import jax


def resolve_export_api():
    """Locate the export interface supported by the installed JAX version."""
    from jax import export

    return SimpleNamespace(module=export, name="jax.export", modern=True)


def export_and_serialize(function, *argument_specs, platforms=None):
    """Export a JAX callable and serialize its portable program."""
    api = resolve_export_api()
    exported = api.module.export(
        jax.jit(function), platforms=None if platforms is None else tuple(platforms)
    )(*argument_specs)
    return exported, bytes(exported.serialize()), api


def deserialize_exported(serialized):
    """Restore a portable JAX program from its serialized bytes."""
    api = resolve_export_api()
    exported = api.module.deserialize(bytearray(serialized))
    return exported, exported.call, api


def serialize_predictor(
    function, *, batch_size, input_size, output_size, dtype, platforms=None
):
    """Serialize a fixed-shape inference graph without native compilation."""
    import hashlib
    import numpy as np
    from purin.execution.runtime import resolve_jax_dtype

    dtype = resolve_jax_dtype(dtype)
    example = jax.ShapeDtypeStruct((int(batch_size), int(input_size)), dtype)
    exported, serialized, api = export_and_serialize(
        function, example, platforms=platforms
    )
    manifest = {
        "format": "ohagi_jax_program_v1",
        "batch_size": int(batch_size),
        "input_size": int(input_size),
        "output_size": int(output_size),
        "dtype": np.dtype(dtype).name,
        "platforms": list(exported.platforms),
        "jax_version": jax.__version__,
        "export_api": api.name,
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }
    return serialized, manifest
