"""Versioned, pickle-free checkpoints of complete training state."""

import hashlib
import json
from pathlib import Path
import numpy as np
import jax
from ohagi.training.engine import TrainState
from ohagi.training.state import AdamState


def save_checkpoint(path, state, *, metadata=None):
    """Write model training state using the selected supported checkpoint format."""
    from ohagi.export.weights import save_pytree

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    # NamedTuple types must be reconstructed explicitly on load.
    builtin = isinstance(state.optimizer_state, AdamState)
    key = np.asarray(jax.random.key_data(state.key))
    payload = {
        "parameters": state.parameters,
        "model_state": state.model_state,
        "optimizer_state": tuple(state.optimizer_state)
        if builtin
        else jax.tree_util.tree_leaves(state.optimizer_state),
        "key": key,
        "step": state.step,
    }
    manifest = save_pytree(root / "state.npz", payload)
    record = {
        "format": "hax_training_v1",
        "tree": manifest,
        "optimizer": "hax.Adam" if builtin else "template",
        "typed_key": jax.dtypes.issubdtype(state.key.dtype, jax.dtypes.prng_key),
        "key_impl": str(jax.random.key_impl(state.key)),
        "sha256": hashlib.sha256((root / "state.npz").read_bytes()).hexdigest(),
        "metadata": metadata or {},
    }
    (root / "manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return root / "manifest.json"


def load_checkpoint(path, *, optimizer_template=None):
    """Restore training state from a saved checkpoint and matching template."""
    from ohagi.export.weights import load_pytree

    root = Path(path)
    record = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if record.get("format") != "hax_training_v1" or record.get("optimizer") not in {
        "hax.Adam",
        "template",
    }:
        raise ValueError("Unsupported checkpoint schema")
    if (
        hashlib.sha256((root / "state.npz").read_bytes()).hexdigest()
        != record["sha256"]
    ):
        raise ValueError("Checkpoint checksum mismatch")
    data = load_pytree(root / "state.npz", record["tree"])
    key = (
        jax.random.wrap_key_data(data["key"], impl=record["key_impl"])
        if record["typed_key"]
        else data["key"]
    )
    if record["optimizer"] == "hax.Adam":
        optimizer_state = AdamState(*data["optimizer_state"])
    else:
        if optimizer_template is None:
            raise ValueError(
                "An optimizer state template is required for external optimizers"
            )
        leaves = jax.tree_util.tree_leaves(data["optimizer_state"])
        structure = jax.tree_util.tree_structure(optimizer_template)
        if len(leaves) != structure.num_leaves:
            raise ValueError("Optimizer template does not match checkpoint")
        optimizer_state = jax.tree_util.tree_unflatten(structure, leaves)
    state = TrainState(
        data["parameters"], data["model_state"], optimizer_state, key, data["step"]
    )
    return state, record["metadata"]
