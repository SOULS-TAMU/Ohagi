"""Metadata-driven export and prepared single-instance JAX inference.

Only the metadata path is supplied by the caller; its referenced portable
program stays alongside it. Loading prepares one fixed vector signature.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from purin.execution.runtime import resolve_device, resolve_jax_dtype
from ohagi._utils.serialization import atomic_write, write_json

FORMAT = "ohagi_jax_inference_v1"
__all__ = ["Predictor", "export", "load"]


def _dtype(value):
    # Loading is an explicit precision-setting operation. Importing OHAGI alone
    # still leaves the application's JAX configuration unchanged.
    name = {"fp64": "float64", "64": "float64", "fp32": "float32", "32": "float32"}.get(
        str(value), value
    )
    if np.dtype(name) == np.dtype("float64") and not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
    return resolve_jax_dtype(value)


def specification(
    portable, *, framework, input_names=None, output_names=None, example_input=None
):
    """Describe a portable program using the common deployment schema."""
    if int(portable["batch_size"]) != 1:
        raise ValueError("Shared inference requires an exported batch size of one.")
    p, n = int(portable["input_size"]), int(portable["output_size"])
    names = {}
    for key, value, size in [
        ("input_names", input_names, p),
        ("output_names", output_names, n),
    ]:
        if value is not None:
            if len(value) != size:
                raise ValueError(f"{key} must contain {size} names.")
            names[key] = list(value)
    warmup = np.zeros(p) if example_input is None else np.asarray(example_input)
    if warmup.shape != (p,) or not np.isfinite(warmup).all():
        raise ValueError(f"example_input must be a finite vector of shape ({p},).")
    return {
        "format": FORMAT,
        "backend": "jax",
        "framework": str(framework),
        "portable": dict(portable),
        "warmup_input": warmup.tolist(),
        **names,
    }


def write_program(serialized, manifest, directory):
    """Atomically write one content-addressed portable JAX program."""
    manifest = dict(manifest)
    digest = hashlib.sha256(serialized).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("Portable program checksum does not match its manifest.")
    manifest["format"] = "ohagi_jax_program_v1"
    path = Path(directory) / f"model_{digest}.jax_export"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["file"] = path.name
    atomic_write(path, serialized)
    return manifest


def export(
    metadata_path,
    predict_fn,
    *,
    input_size,
    output_size,
    dtype="float64",
    framework="custom",
    input_names=None,
    output_names=None,
    example_input=None,
    platforms=("cpu",),
):
    """Export a pure single-instance JAX predictor and update its metadata.

    Args:
        metadata_path: Destination metadata path. Existing fields are retained.
        predict_fn: Pure callable accepting ``(1, p)`` and returning ``(1, n)``.
            Capture frozen weights and problem resources in this callable.
        input_size: Input feature count ``p``.
        output_size: Output feature count ``n``.
        dtype: Declared input and output floating-point dtype.
        framework: Frontend identifier recorded in metadata.
        input_names: Optional ordered names for the input features.
        output_names: Optional ordered names for the output features.
        example_input: Optional finite ``(p,)`` warm-up vector; defaults to zeros.
        platforms: Target platforms accepted by JAX export; defaults to CPU.

    Returns:
        Mapping containing the metadata path and shared inference specification.

    Raises:
        ValueError: If the callable's output shape or dtype violates the
            declared signature, or the warm-up example is invalid.

    Notes:
        The content-addressed portable program is written beside the metadata.
        Loading this bundle uses :func:`load`; native backends are unsupported.
    """
    from ohagi.serialization._jax import serialize_predictor

    path = Path(metadata_path).expanduser().resolve()
    data = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"format": "ohagi_model_v1", "framework": framework}
    )
    dtype = _dtype(dtype)
    with jax.default_device(resolve_device("cpu")):
        signature = jax.ShapeDtypeStruct((1, int(input_size)), dtype)
        result = jax.eval_shape(predict_fn, signature)
        if not hasattr(result, "shape") or result.shape != (1, int(output_size)):
            raise ValueError(
                "Exported function must return one array with shape (1, output_size)."
            )
        if np.dtype(result.dtype) != np.dtype(dtype):
            raise ValueError(
                "Exported output dtype differs from the declared inference dtype."
            )
        serialized, manifest = serialize_predictor(
            predict_fn,
            batch_size=1,
            input_size=input_size,
            output_size=output_size,
            dtype=dtype,
            platforms=platforms,
        )
    # Validate metadata before changing any files.
    info = specification(
        manifest,
        framework=framework,
        input_names=input_names,
        output_names=output_names,
        example_input=example_input,
    )
    info["portable"] = write_program(serialized, manifest, path.parent)
    data["inference"] = info
    write_json(path, data)
    return {"metadata_path": str(path), **info}


class Predictor:
    """Forward predictor with frozen model parameters and one vector signature.

    Calls accept (p,) or (1,p); batches larger than one are rejected rather
    than compiling another signature. No files or frontends are consulted
    after load. predict_jax returns a JAX array; predict
    returns host NumPy output and therefore synchronizes each request.
    """

    batch_size = 1

    def __init__(
        self,
        graph,
        manifest,
        *,
        metadata_path,
        framework,
        device,
        example_input=None,
        input_names=None,
        output_names=None,
        source="portable",
    ):
        self.input_size, self.output_size = (
            int(manifest["input_size"]),
            int(manifest["output_size"]),
        )
        self.dtype = _dtype(manifest["dtype"])
        self.device = resolve_device(device)
        self.metadata_path, self.framework = str(metadata_path), framework
        self.backend, self.source = "jax", source
        self.input_names = None if input_names is None else tuple(input_names)
        self.output_names = None if output_names is None else tuple(output_names)
        for value, size in [
            (self.input_names, self.input_size),
            (self.output_names, self.output_size),
        ]:
            if value is not None and len(value) != size:
                raise ValueError(
                    "Inference variable names do not match the declared dimensions."
                )

        def single(values):
            return graph(values[None, :])[0]

        with jax.default_device(self.device):
            arg = jax.ShapeDtypeStruct((self.input_size,), self.dtype)
            out = jax.eval_shape(single, arg)
            if out.shape != (self.output_size,) or np.dtype(out.dtype) != np.dtype(
                self.dtype
            ):
                raise ValueError(
                    "Prediction graph does not match metadata output shape/dtype."
                )
            self._compiled = jax.jit(single)
            warmup = (
                np.zeros(self.input_size)
                if example_input is None
                else np.asarray(example_input)
            )
            if warmup.shape != (self.input_size,) or not np.isfinite(warmup).all():
                raise ValueError("Invalid warmup_input in inference metadata.")
        # Warm under the same JAX context used by ordinary calls. Keeping an
        # explicit default_device context only for warm-up causes a second
        # cache entry on the first real request in some JAX versions.
        jax.block_until_ready(
            self._compiled(jax.device_put(warmup.astype(self.dtype), self.device))
        )

    def _shape(self, shape):
        if shape == (self.input_size,):
            return False
        if shape == (1, self.input_size):
            return True
        raise ValueError(
            f"Expected one instance of shape ({self.input_size},) or (1, {self.input_size}); got {shape}."
        )

    def predict(self, values):
        """Predict one instance and return synchronized host NumPy values.

        Args:
            values: Input array of shape ``(p,)`` or ``(1, p)``, where ``p`` is
                ``input_size``. Values are converted to the exported dtype.

        Returns:
            NumPy output of shape ``(n,)`` or ``(1, n)``, preserving whether the
            input has a batch axis. Here ``n`` is ``output_size``.

        Raises:
            ValueError: If the input does not represent exactly one instance.

        Notes:
            Conversion of the result to NumPy synchronizes device execution.
            Use :meth:`predict_jax` to keep the result on the JAX device.
        """
        host = np.asarray(values, dtype=np.dtype(self.dtype))
        batched = self._shape(host.shape)
        value = host.reshape(self.input_size)
        result = np.asarray(self._compiled(jax.device_put(value, self.device)))
        return result[None, :] if batched else result

    __call__ = predict

    def predict_jax(self, values):
        """Predict one instance and keep the result as a device-resident JAX array.

        Args:
            values: Input array of shape ``(p,)`` or ``(1, p)``. Inputs are cast
                to the exported dtype and placed on this predictor's device.

        Returns:
            JAX output with the same batch-axis convention as the input and
            final dimension ``output_size``.

        Raises:
            ValueError: If the input shape differs from the fixed single-instance
                signature. Larger batches are not compiled implicitly.

        Notes:
            Device execution may be asynchronous. Call ``jax.block_until_ready``
            when timing completed predictions.
        """
        values = jnp.asarray(values, dtype=self.dtype)
        batched = self._shape(values.shape)
        result = self._compiled(
            jax.device_put(values.reshape(self.input_size), self.device)
        )
        return result[None, :] if batched else result


def load(metadata_path, *, device="cpu"):
    """Prepare a single-instance predictor from shared OHAGI JAX metadata.

    Args:
        metadata_path: Path to ``metadata.json``. Its referenced portable
            program must remain in the model bundle.
        device: JAX device or supported device selector; defaults to CPU.

    Returns:
        A warmed :class:`Predictor` with frozen parameters and one fixed
        input signature, ready for repeated calls.

    Notes:
        Loading validates the format, program checksum, dimensions and dtype.
        It enables JAX float64 when required by the exported program. No
        frontend or training checkpoint is needed for subsequent predictions.
    """
    from ohagi.serialization.metadata import read_model
    from ohagi.serialization._jax import deserialize_exported

    path = Path(metadata_path).expanduser().resolve()
    data = read_model(path)
    info = data.get("inference", {})
    if info.get("format") != FORMAT:
        raise ValueError("Expected shared OHAGI JAX inference metadata.")
    if info.get("backend") != "jax":
        raise ValueError("Only JAX inference metadata is supported.")
    if info.get("framework") != data.get("framework"):
        raise ValueError("Inference graph and bundle framework disagree.")
    portable = info.get("portable")
    if (
        not isinstance(portable, dict)
        or portable.get("format") != "ohagi_jax_program_v1"
    ):
        raise ValueError("Expected an OHAGI JAX program in the model bundle.")
    if int(portable.get("batch_size", 0)) != 1:
        raise ValueError(
            "Portable metadata must describe a batch-one prediction graph."
        )
    target = (path.parent / portable["file"]).resolve()
    if not target.is_relative_to(path.parent):
        raise ValueError("Program file must remain inside its model bundle.")
    serialized = target.read_bytes()
    if hashlib.sha256(serialized).hexdigest() != portable["sha256"]:
        raise ValueError("Portable program checksum mismatch.")
    _dtype(portable["dtype"])
    exported, graph, _ = deserialize_exported(serialized)
    dtype = np.dtype(portable["dtype"])
    for avals, size in [
        (exported.in_avals, portable["input_size"]),
        (exported.out_avals, portable["output_size"]),
    ]:
        if (
            len(avals) != 1
            or avals[0].shape != (1, int(size))
            or np.dtype(avals[0].dtype) != dtype
        ):
            raise ValueError("Portable graph signature does not match metadata.")
    return Predictor(
        graph,
        portable,
        metadata_path=path,
        framework=data["framework"],
        device=device,
        example_input=info.get("warmup_input"),
        input_names=info.get("input_names"),
        output_names=info.get("output_names"),
    )
