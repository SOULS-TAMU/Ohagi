"""Framework-independent model metadata and numeric parameters.

Inference graphs are serialized by JAX. Frontends supply definitions and pure
graphs; this module owns the disk format and the only new-model writer.
"""

from pathlib import Path
from ohagi._utils.serialization import json_safe, write_json
from ohagi.export.weights import save_pytree, load_pytree

from .metadata import FORMAT, read_model


def load_parameters(metadata_path, metadata=None):
    """Load the parameter artifact referenced by shared model metadata."""
    path = Path(metadata_path).expanduser().resolve()
    data = read_model(path) if metadata is None else metadata
    if data.get("format") != FORMAT:
        raise ValueError("Expected an OHAGI JAX model bundle.")
    if data.get("format") != FORMAT:
        raise ValueError("Expected an OHAGI JAX model bundle.")
    manifest = data["parameters"]
    candidate = (path.parent / manifest["file"]).resolve()
    if not candidate.is_relative_to(path.parent):
        raise ValueError("Parameter file must remain inside its model bundle.")
    return load_pytree(candidate, manifest)


def save_model(
    metadata_path, *, framework, parameters, definition=None, inference_spec
):
    """Write shared model metadata, parameters and optional inference resources."""
    from ohagi.inference import export

    path = Path(metadata_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(definition or {})
    data.pop("format", None)
    data.pop("inference", None)
    data.update(format=FORMAT, framework=str(framework))
    data["parameters"] = save_pytree(path.parent / "weights.npz", parameters)
    write_json(path, json_safe(data))
    spec = dict(inference_spec)
    if spec.get("framework", framework) != framework:
        raise ValueError("Inference graph and bundle framework disagree.")
    spec["framework"] = framework
    return export(path, **spec)
