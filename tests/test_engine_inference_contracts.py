"""Shared model-state updates, early stopping and inference contract failures."""

from types import SimpleNamespace
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ohagi import Adam
from ohagi.models.projected import OutputLayout, ProjectedModel
from ohagi.training.engine import (
    Trainer,
    create_train_state,
    make_model_train_step,
    build_compiled_epoch_fns,
)


class StatefulScale:
    def init(self, key, input_dim, output_dim, dtype):
        return jnp.array(2.0, dtype), jnp.array(0)

    def apply(self, params, state, inputs, *, key=None, training=False):
        return params * inputs, state + int(training)


def squared_loss(outputs, batch):
    error = jnp.mean((outputs["projected"] - batch["targets"]) ** 2, axis=-1)
    mask = batch.get("mask", jnp.ones_like(error))
    loss = jnp.sum(error * mask) / jnp.sum(mask)
    return loss, {"mse": loss, "gradient_valid": jnp.array(True)}


@pytest.mark.parametrize("jit", [False, True])
def test_trainer_validation_and_early_stop(jit):
    backbone = StatefulScale()
    optimizer = Adam(learning_rate=0.05)
    state = create_train_state(
        backbone, optimizer, jax.random.PRNGKey(0), input_dim=1, output_dim=1
    )
    data = {"inputs": np.ones((3, 1)), "targets": np.zeros((3, 1))}
    seen = []

    def stop(epoch, current, record):
        seen.append((epoch, int(current.step)))
        return epoch == 2

    trainer = Trainer(ProjectedModel(backbone), squared_loss, optimizer, jit=jit)
    result = trainer.fit(
        state, data, epochs=5, batch_size=2, validation=data, callbacks=(stop,)
    )
    assert seen == [(1, 2), (2, 4)]
    assert int(result.state.model_state) == 4
    assert result.history[-1]["validation_loss"] < result.history[0]["validation_loss"]
    assert result.history[-1]["epoch"] == 2
    for epochs in [0, -1, 1.5]:
        with pytest.raises(ValueError, match="epochs"):
            trainer.fit(state, data, epochs=epochs, batch_size=2)


@pytest.mark.parametrize("reason", ["invalid_gradient", "nonfinite", "ineligible"])
def test_model_step_rejects_invalid_updates_without_committing_state(reason):
    backbone = StatefulScale()
    optimizer = Adam()
    state = create_train_state(
        backbone, optimizer, jax.random.PRNGKey(1), input_dim=1, output_dim=1
    )
    base = ProjectedModel(backbone)

    class Model:
        def apply(self, *args, **kwargs):
            outputs, new_state = base.apply(*args, **kwargs)
            if reason == "ineligible":
                outputs["dual_valid"] = jnp.zeros(2, bool)
            return outputs, new_state

    def loss(outputs, batch):
        value, metrics = squared_loss(outputs, batch)
        if reason == "nonfinite":
            value = value * jnp.nan
        metrics["gradient_valid"] = jnp.array(reason != "invalid_gradient")
        return value, metrics

    step = jax.jit(make_model_train_step(Model(), loss, optimizer))
    new, metrics = step(
        state, {"inputs": jnp.ones((2, 1)), "targets": jnp.zeros((2, 1))}
    )
    assert not bool(metrics["update_accepted"])
    for previous, current in zip(
        jax.tree_util.tree_leaves(state[:3]), jax.tree_util.tree_leaves(new[:3])
    ):
        np.testing.assert_array_equal(current, previous)
    assert int(new.step) == 0
    assert not np.array_equal(new.key, state.key)


def test_compiled_epoch_reducers_match_manual_arithmetic():
    def train(state, batch):
        return state + batch, {"value": state + batch}

    def evaluate(state, batch):
        return {"value": state * batch}

    train_epoch, evaluate_epoch = build_compiled_epoch_fns(train, evaluate, ("value",))
    state, metrics = train_epoch(jnp.array(0.0), jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(state, 6)
    np.testing.assert_allclose(metrics, [10 / 3])
    np.testing.assert_allclose(evaluate_epoch(state, jnp.array([1.0, 2.0, 3.0])), [12])


def test_output_layout_and_model_preparation_hooks():
    for args in [(0,), (1, -1), (1, 0, -1)]:
        with pytest.raises(ValueError):
            OutputLayout(*args)
    layout = OutputLayout(1, 1, 1)
    with pytest.raises(ValueError, match="tuple"):
        layout.split((jnp.ones((2, 1)),))
    with pytest.raises(ValueError, match="width"):
        layout.split(jnp.ones((2, 2)))
    parts = tuple(jnp.full((2, 1), i) for i in (1, 2, 3))
    assert layout.split(parts) is parts
    np.testing.assert_array_equal(
        jnp.concatenate(layout.split(jnp.array([[1, 2, 3]])), axis=-1), [[1, 2, 3]]
    )
    model = ProjectedModel(
        StatefulScale(),
        prepare_inputs=lambda p, x, c: (p, x + 1, c + 2),
        output_transform=lambda output, context: (output, output[:, :0], output[:, :0]),
        prepare_projection=lambda raw, eq, iq, context: (
            raw + context,
            eq,
            iq,
            context,
        ),
        return_projection=True,
        return_context=True,
    )
    output, state = model.apply(jnp.array(2.0), jnp.array(0), jnp.ones((2, 1)))
    np.testing.assert_array_equal(output["projected"], 7 * np.ones((2, 1)))
    np.testing.assert_array_equal(output["context"], 3 * np.ones((2, 1)))
    assert output["projection_result"] is None
    assert int(state) == 0


def test_inference_metadata_validation_and_graph_contracts(tmp_path):
    from ohagi.inference import Predictor, export, load, specification, write_program

    manifest = {"batch_size": 1, "input_size": 1, "output_size": 1, "dtype": "float64"}
    for changes, options in [
        ({"batch_size": 2}, {}),
        ({}, {"input_names": []}),
        ({}, {"output_names": []}),
        ({}, {"example_input": [np.nan]}),
        ({}, {"example_input": [1, 2]}),
    ]:
        with pytest.raises(ValueError):
            specification({**manifest, **changes}, framework="test", **options)
    with pytest.raises(ValueError, match="checksum"):
        write_program(b"bad", {"sha256": "wrong"}, tmp_path)
    for options in [
        {"input_names": []},
        {"output_names": []},
        {"example_input": [np.nan]},
    ]:
        with pytest.raises(ValueError):
            Predictor(
                lambda x: x,
                manifest,
                metadata_path="metadata.json",
                framework="test",
                device="cpu",
                **options,
            )
    with pytest.raises(ValueError, match="shape/dtype"):
        Predictor(
            lambda x: jnp.concatenate((x, x), axis=-1),
            manifest,
            metadata_path="metadata.json",
            framework="test",
            device="cpu",
        )
    for fn in [lambda x: (x, x), lambda x: x.astype(jnp.float32)]:
        with pytest.raises(ValueError):
            export(tmp_path / "invalid.json", fn, input_size=1, output_size=1)
    path = tmp_path / "metadata.json"
    export(
        path,
        lambda x: 2 * x,
        input_size=1,
        output_size=1,
        input_names=["x"],
        output_names=["y"],
    )
    valid = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        ("format", "old"),
        ("backend", "native"),
        ("framework", "other"),
        ("portable", None),
        ("portable.batch_size", 2),
        ("portable.file", "../outside.jax_export"),
        ("portable.sha256", "wrong"),
        ("portable.input_size", 2),
        ("warmup_input", [np.nan]),
        ("input_names", []),
    ]
    for key, value in cases:
        data = json.loads(json.dumps(valid))
        target = data["inference"]
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError):
            load(path)
    path.write_text(json.dumps(valid), encoding="utf-8")
    predictor = load(path)
    np.testing.assert_allclose(predictor.predict_jax([[3.0]]), [[6.0]])
    assert predictor.input_names == ("x",)


def test_orbax_coarse_clock_patch_is_idempotent(monkeypatch):
    import ohagi.io._orbax_compat as compat

    calls = []
    handler = SimpleNamespace(
        time=SimpleNamespace(time=lambda: 10.0),
        _log_io_metrics=lambda size, start: calls.append((size, start)),
    )
    compat._patch_handler(handler)
    patched = handler._log_io_metrics
    compat._patch_handler(handler)
    assert handler._log_io_metrics is patched
    patched(5, 10.0)
    patched(7, 9.0)
    assert calls[0][1] < 10 and calls[1] == (7, 9.0)
    monkeypatch.setattr(compat, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(compat, "sys", SimpleNamespace(version_info=(3, 12)))
    monkeypatch.setattr(compat.importlib, "import_module", lambda _: handler)
    compat.prepare_orbax(SimpleNamespace(__version__="0.11.5"))
    compat.prepare_orbax(SimpleNamespace(__version__="0.12.0"))
