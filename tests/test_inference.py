"""Metadata-only JAX export, repeated inference and corrupt-bundle rejection."""

import json
import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from ohagi.inference import export, load


def make_bundle(path):
    weights = jnp.array([[2.0, -1.0], [0.5, 3.0]])
    export(
        path,
        lambda x: x @ weights,
        input_size=2,
        output_size=2,
        framework="custom",
        example_input=np.ones(2),
    )


def test_metadata_only_repeated_inference_in_fresh_process(tmp_path):
    path = tmp_path / "metadata.json"
    make_bundle(path)
    code = """
import sys
import numpy as np
from ohagi.inference import load
predictor = load(sys.argv[1])
for _ in range(20):
    np.testing.assert_allclose(predictor.predict([1., 2.]), [3., 5.], atol=1e-12)
    np.testing.assert_allclose(predictor.predict_jax(np.array([1., 2.])), [3., 5.], atol=1e-12)
assert not any(name in sys.modules for name in ('kkthn', 'daehn', 'nlpoptnet', 'flax', 'orbax'))
"""
    subprocess.run(
        [sys.executable, "-c", code, str(path)],
        check=True,
        timeout=90,
        env=os.environ.copy(),
    )


def test_inference_shape_and_checksum_validation(tmp_path):
    path = tmp_path / "metadata.json"
    make_bundle(path)
    predictor = load(path)
    with pytest.raises(ValueError):
        predictor.predict(np.ones((2, 2)))
    record = json.loads(path.read_text())
    program = path.parent / record["inference"]["portable"]["file"]
    program.write_bytes(program.read_bytes() + b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        load(path)


def test_inference_rejects_escaping_program_path(tmp_path):
    path = tmp_path / "metadata.json"
    make_bundle(path)
    record = json.loads(path.read_text())
    record["inference"]["portable"]["file"] = "../outside.jax_export"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="inside"):
        load(path)


def test_export_rejects_inconsistent_output_shape(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        export(tmp_path / "metadata.json", lambda x: x, input_size=2, output_size=3)
