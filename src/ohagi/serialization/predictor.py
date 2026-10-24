"""Save, load, compile, and run fixed-shape KKTHN predictors."""

from __future__ import annotations
import importlib
from pathlib import Path
from typing import Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from ohagi.execution.inference import require_jax_preference
from purin.solvers import ImplicitGradientConfig
from purin.solvers import LinearSolverConfig
from purin.solvers import ProjectionConfig
from purin.solvers import prepare_projection
from purin.solvers.linear_equality import LinearEqualityProjection
from purin.execution.runtime import resolve_jax_dtype as _dtype
from ohagi._utils.serialization import json_safe as _json_safe
from ohagi.serialization.problem import save_problem


def _activation(name: str):
    normalized = str(name).lower()
    choices = {
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "silu": jax.nn.silu,
        "swish": jax.nn.silu,
        "sigmoid": jax.nn.sigmoid,
        "tanh": jnp.tanh,
        "softplus": jax.nn.softplus,
    }
    try:
        return choices[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported serialized activation {name!r}") from exc


def make_network_apply(spec: dict[str, Any]) -> Callable:
    """Reconstruct a pure network application function from its saved architecture."""
    network_type = spec.get("type", "mlp")
    if network_type != "mlp":
        raise ValueError(f"unsupported serialized network type {network_type!r}")
    activation = _activation(spec.get("activation", "relu"))
    parameter_format = spec.get("parameter_format", "auto")

    parameter_path = tuple(spec.get("parameter_path", ()))

    def apply(params, inputs):
        values = inputs
        selected = params
        if parameter_path:
            for key in parameter_path:
                selected = selected[key]
        elif isinstance(selected, dict) and "backbone" in selected:
            # ProjectedNeuralModel stores trainable parameters under a
            # ``backbone`` key. Keep direct layer tuples supported for the
            # low-level serialization API while transparently accepting the
            # model-owned PyTree.
            selected = selected["backbone"]
        layers = tuple(selected)
        for index, layer in enumerate(layers):
            if parameter_format == "legacy" or "W" in layer:
                values = values @ layer["W"] + layer["b"]
            else:
                values = values @ layer["kernel"]
                if "bias" in layer:
                    values = values + layer["bias"]
            if index + 1 < len(layers):
                values = activation(values)
        return values

    return apply


def _projection_config_from_dict(value: dict[str, Any]) -> ProjectionConfig:
    linear = LinearSolverConfig(**value.get("linear_solver", {}))
    implicit = ImplicitGradientConfig(**value.get("implicit_gradient", {}))
    fields = {
        key: item
        for key, item in value.items()
        if key not in {"linear_solver", "implicit_gradient"}
    }
    return ProjectionConfig(**fields, linear_solver=linear, implicit_gradient=implicit)


def _save_projection_arrays(path: Path, projection_spec: dict[str, Any]):
    arrays = {}
    serializable = dict(projection_spec)
    for name in ("A", "b", "b0", "b_parameter_map", "lower", "upper"):
        if name in serializable and serializable[name] is not None:
            key = f"projection_{name}"
            arrays[key] = np.asarray(serializable.pop(name))
            serializable[f"{name}_key"] = key
    if arrays:
        np.savez(path, **arrays)
        serializable["arrays_file"] = path.name
    return serializable


def _load_projection_arrays(root: Path, spec: dict[str, Any]):
    restored = dict(spec)
    file = restored.pop("arrays_file", None)
    if file is None:
        return restored
    with np.load(root / file, allow_pickle=False) as arrays:
        for name in ("A", "b", "b0", "b_parameter_map", "lower", "upper"):
            key = restored.pop(f"{name}_key", None)
            if key is not None:
                restored[name] = np.asarray(arrays[key])
    return restored


def _prepare_projection(
    spec: dict[str, Any], *, problem, root: Path, batch_size: int, dtype, backend: str
):
    """Reconstruct the saved projection definition with fixed inference dimensions."""
    spec = _load_projection_arrays(root, spec)
    kind = spec.get("type", "none")
    if kind == "none":
        return None
    if kind == "box":
        lower = jnp.asarray(spec["lower"], dtype=dtype)
        upper = jnp.asarray(spec["upper"], dtype=dtype)
        return lambda values, parameters=None: jnp.clip(values, lower, upper)
    if kind == "linear_equality":
        return LinearEqualityProjection(
            A=spec["A"],
            b=spec.get("b"),
            b0=spec.get("b0"),
            b_parameter_map=spec.get("b_parameter_map"),
            regularization=float(spec.get("regularization", 0.0)),
        ).prepare(batch_size=batch_size, dtype=dtype, backend=backend)
    if problem is None:
        raise ValueError(f"projection {kind!r} requires a serialized problem")
    if kind == "implicit-distance":
        from purin.projections.implicit_distance import SparseProjection

        return SparseProjection(
            problem, _projection_config_from_dict(spec.get("config", {}))
        ).prepare(batch_size=batch_size, dtype=dtype, backend=backend)
    if kind == "dense-semismooth":
        from purin.solvers.fischer_burmeister.dense_semismooth import (
            FischerBurmeisterProjection,
        )

        options = {key: value for key, value in spec.items() if key != "type"}
        return FischerBurmeisterProjection(problem, **options).prepare(
            batch_size=batch_size, dtype=dtype, backend=backend
        )
    if kind == "sparse":
        config = _projection_config_from_dict(spec.get("config", {}))
        return prepare_projection(
            problem,
            method="primal-dual-interior-point",
            config=config,
            batch_size=batch_size,
            dtype=dtype,
            backend=backend,
        )
    if kind == "fischer_burmeister":
        kwargs = {key: value for key, value in spec.items() if key != "type"}
        method = kwargs.pop("method", "fischer-burmeister-newton")
        return prepare_projection(
            problem,
            method=method,
            config=kwargs.pop("config", None),
            batch_size=batch_size,
            dtype=dtype,
            backend=backend,
        )
    raise ValueError(f"unsupported serialized projection type {kind!r}")


from ohagi.inference import Predictor


def save_predictor(
    directory,
    *,
    network_spec,
    params,
    input_size,
    output_size,
    batch_size,
    projection_spec=None,
    problem=None,
    input_transform=None,
    dtype="float64",
    backend="cpu",
    metadata=None,
    predictor_factory=None,
    factory_config=None,
    _inference_spec=None,
    framework="custom",
):
    """Save the complete predictor in the shared OHAGI JAX bundle format."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    problem_manifest = (
        None if problem is None else save_problem(root / "problem_arrays.npz", problem)
    )
    projection_spec = _save_projection_arrays(
        root / "projection_arrays.npz", projection_spec or {"type": "none"}
    )
    transform = dict(input_transform or {"type": "identity"})
    transform_arrays = {}
    for name in ("mean", "scale"):
        if name in transform:
            key = f"transform_{name}"
            transform_arrays[key] = np.asarray(transform.pop(name))
            transform[f"{name}_key"] = key
    if transform_arrays:
        np.savez(root / "normalization.npz", **transform_arrays)
        transform["arrays_file"] = "normalization.npz"
    manifest = dict(
        network=_json_safe(network_spec),
        problem=problem_manifest,
        projection=_json_safe(projection_spec),
        input_transform=_json_safe(transform),
        input_size=int(input_size),
        output_size=int(output_size),
        batch_size=int(batch_size),
        dtype=str(dtype),
        metadata=_json_safe(metadata or {}),
        predictor_factory=(
            f"{predictor_factory.__module__}:{predictor_factory.__qualname__}"
            if callable(predictor_factory)
            else predictor_factory
        ),
        factory_config=_json_safe(factory_config or {}),
    )
    if _inference_spec is not None:
        spec = _inference_spec
    else:
        if predictor_factory:
            graph = _load_factory(predictor_factory)(
                params=params,
                manifest=manifest,
                root=root,
                backend=backend,
                dtype=_dtype(dtype),
            )
        else:
            projection = _prepare_projection(
                projection_spec,
                problem=problem,
                root=root,
                batch_size=batch_size,
                dtype=_dtype(dtype),
                backend=backend,
            )
            from ohagi.models.composition import inference_graph

            graph = inference_graph(
                make_network_apply(network_spec),
                params,
                _load_transform(root, manifest["input_transform"]),
                projection,
            )

        def single(inputs):
            return graph(jnp.pad(inputs, ((0, batch_size - 1), (0, 0))))[:1]

        spec = dict(
            predict_fn=single,
            framework=framework,
            input_size=input_size,
            output_size=output_size,
            dtype=dtype,
        )
    from ohagi.serialization import save_model
    from ohagi.inference import load

    save_model(
        root / "metadata.json",
        framework=framework,
        parameters=params,
        definition=manifest,
        inference_spec=spec,
    )
    return load(root / "metadata.json")


def _load_transform(root: Path, spec: dict[str, Any]):
    restored = dict(spec)
    file = restored.pop("arrays_file", None)
    if file is not None:
        with np.load(root / file, allow_pickle=False) as arrays:
            for name in ("mean", "scale"):
                key = restored.pop(f"{name}_key", None)
                if key is not None:
                    restored[name] = jnp.asarray(arrays[key])
    kind = restored.get("type", "identity")
    if kind == "identity":
        return lambda x: x
    if kind == "standard":
        mean = jnp.asarray(restored["mean"])
        scale = jnp.asarray(restored["scale"])
        return lambda x: (x - mean) / scale
    raise ValueError(f"unsupported input transform {kind!r}")


def _load_factory(path: str):
    if callable(path):
        return path
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError(
            "Use an importable module:factory path for custom graph creation."
        )
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("Predictor factory must be callable.")
    return factory


def load_predictor(
    model_json: str | Path, *, backend: str | None = None, preference: str = "json"
) -> Predictor:
    """Load shared OHAGI metadata; preference is a compatibility argument."""
    from ohagi.inference import load

    require_jax_preference(preference)
    return load(model_json, device=backend or "cpu")


def model_graph(model, *, framework="custom"):
    """Capture a built projected trainer as a pure batch-one JAX function."""
    model._require_built()
    parameters = jax.device_put(model.parameters, jax.devices("cpu")[0])
    forward = model._inference_forward

    def graph(inputs):
        batch = jnp.repeat(inputs, int(model.batch_size), axis=0)
        return forward(parameters, batch).y_tilde[:1]

    names = list(model.prepared_projection.compiled.variable_names)
    return dict(
        predict_fn=graph,
        framework=framework,
        input_size=model.input_dim,
        output_size=len(names),
        output_names=names,
        dtype="float64",
    )
