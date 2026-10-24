"""Lazy, synchronous Orbax checkpoints; deployment exports are independent.

Restore against a live template to preserve optimizer namedtuples, FrozenDicts,
array dtypes and device placement. Never reconstruct Python objects from files.
"""

from pathlib import Path
import json
import jax
import numpy as np


def _orbax():
    try:
        import orbax.checkpoint as ocp
    except ImportError as error:
        raise ImportError("Install ohagi[orbax] to use Orbax checkpointing.") from error
    from ._orbax_compat import prepare_orbax

    prepare_orbax(ocp)
    return ocp


def save_orbax(path, tree, *, metadata=None):
    """Save a pytree atomically to a NEW directory; never overwrite a checkpoint."""
    ocp = _orbax()
    root = Path(path).expanduser().resolve()
    # Serialize user metadata before starting an expensive checkpoint operation.
    leaves = jax.tree_util.tree_leaves(tree)
    # Save flat leaves rather than custom classes; restore their types only from
    # the trusted live template, independent of Orbax custom-pytree support.
    keys = {
        str(i): str(jax.random.key_impl(value))
        for i, value in enumerate(leaves)
        if hasattr(value, "dtype")
        and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
    }
    arrays = {
        str(i): np.asarray(
            jax.device_get(jax.random.key_data(value) if str(i) in keys else value)
        )
        for i, value in enumerate(leaves)
    }
    record = json.dumps(
        {
            "format": "hax_orbax_v1",
            "metadata": metadata or {},
            "structure": str(jax.tree_util.tree_structure(tree)),
            "keys": keys,
        },
        indent=2,
    )
    if any(value.dtype.hasobject for value in arrays.values()):
        raise TypeError("Orbax checkpoint leaves must be numeric arrays or scalars.")
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(str(root), args=ocp.args.PyTreeSave(arrays))
    (root / "hax.json").write_text(record + "\n", encoding="utf-8")
    return root


def load_orbax(path, template):
    """Return (restored_tree, metadata), validating leaf shapes and dtypes."""
    ocp = _orbax()
    root = Path(path).expanduser().resolve()
    record = json.loads((root / "hax.json").read_text(encoding="utf-8"))
    if record.get("format") != "hax_orbax_v1":
        raise ValueError("Unsupported OHAGI Orbax checkpoint format.")
    leaves, structure = jax.tree_util.tree_flatten(template)
    if record.get("structure") != str(structure):
        raise ValueError("Checkpoint template structure mismatch.")
    with ocp.PyTreeCheckpointer() as checkpointer:
        values = checkpointer.restore(str(root))
    if set(values) != {str(i) for i in range(len(leaves))}:
        raise ValueError("Checkpoint template leaf count mismatch.")
    restored = []
    for i, target in enumerate(leaves):
        value = np.asarray(values[str(i)])
        if str(i) in record.get("keys", {}):
            if not (
                hasattr(target, "dtype")
                and jax.dtypes.issubdtype(target.dtype, jax.dtypes.prng_key)
            ):
                raise ValueError(f"Checkpoint template PRNG key mismatch at leaf {i}.")
            if str(jax.random.key_impl(target)) != record["keys"][str(i)]:
                raise ValueError(
                    f"Checkpoint template PRNG implementation mismatch at leaf {i}."
                )
            expected = np.asarray(jax.random.key_data(target))
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(
                    f"Checkpoint template shape/dtype mismatch at leaf {i}."
                )
            restored.append(
                jax.device_put(
                    jax.random.wrap_key_data(value, impl=record["keys"][str(i)]),
                    target.sharding,
                )
            )
            continue
        expected = np.asarray(jax.device_get(target))
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"Checkpoint template shape/dtype mismatch at leaf {i}.")
        restored.append(
            jax.device_put(value, target.sharding)
            if isinstance(target, jax.Array)
            else value
        )
    return jax.tree_util.tree_unflatten(structure, restored), record["metadata"]


def save_model_checkpoint(model, path):
    """Save a task-supplied numeric state using the shared Orbax transport."""
    tree, metadata = model._checkpoint_payload()
    return save_orbax(path, tree, metadata=metadata)


def restore_model_checkpoint(model, path):
    """Restore model and optimizer resources from an Orbax training checkpoint."""
    template, expected = model._checkpoint_payload()
    tree, metadata = load_orbax(path, template)
    if metadata.get("framework") != expected.get("framework"):
        raise ValueError("Checkpoint belongs to a different framework.")
    model._restore_checkpoint_payload(tree, metadata)
    return model
