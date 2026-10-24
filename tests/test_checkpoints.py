"""Versioned pickle-free checkpoints preserve trees, optimizer states and keys."""

import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ohagi import Adam
from ohagi.training.engine import TrainState
from ohagi.io.checkpoints import save_checkpoint, load_checkpoint
from ohagi.export.weights import save_pytree, load_pytree


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_checkpoint_roundtrip(tmp_path, typed, external):
    params = {"weights": jnp.array([[1.0, 2.0], [3.0, 4.0]])}
    opt = (
        {"count": jnp.array(4), "trace": jnp.array([2.0])}
        if external
        else Adam().init(params)
    )
    key = jax.random.key(3) if typed else jax.random.PRNGKey(3)
    state = TrainState(params, {"a": [None, "text", True, 1, 2.0]}, opt, key, 7)
    save_checkpoint(tmp_path, state, metadata={"epoch": 2})
    loaded, metadata = load_checkpoint(tmp_path, optimizer_template=opt)
    assert metadata == {"epoch": 2}
    assert loaded.model_state == state.model_state
    assert loaded.step == 7
    np.testing.assert_array_equal(
        jax.random.key_data(loaded.key), jax.random.key_data(key)
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(loaded.optimizer_state),
        jax.tree_util.tree_leaves(opt),
    ):
        np.testing.assert_array_equal(actual, expected)
    if external:
        with pytest.raises(ValueError, match="template"):
            load_checkpoint(tmp_path)
        with pytest.raises(ValueError, match="match"):
            load_checkpoint(tmp_path, optimizer_template=())
    manifest = tmp_path / "manifest.json"
    record = json.loads(manifest.read_text())
    record["format"] = "bad"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(tmp_path)
    record["format"] = "hax_training_v1"
    record["sha256"] = "bad"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(tmp_path)


def test_pytree_nested_roundtrip_and_corruption(tmp_path):
    tree = {"a": (jnp.arange(3), [None, False, "hi", 3.2]), "b": jnp.ones((2, 2))}
    file = tmp_path / "tree.npz"
    manifest = save_pytree(file, tree)
    restored = load_pytree(file, manifest)
    assert restored["a"][1] == tree["a"][1]
    np.testing.assert_array_equal(restored["a"][0], tree["a"][0])
    with pytest.raises(ValueError):
        load_pytree(file, {"format": "bad"})
    manifest["structure"] = {"kind": "bad"}
    with pytest.raises(ValueError):
        load_pytree(file, manifest)
    manifest["structure"] = {"kind": "array", "key": "array_00000", "shape": [99]}
    with pytest.raises(ValueError, match="shape"):
        load_pytree(file, manifest)
