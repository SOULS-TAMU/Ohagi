"""Export on Linux and verify the same numeric JAX bundles on every OS."""

from pathlib import Path
import sys

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np

from ohagi.inference import export, load


def main(mode, directory):
    jax.config.update("jax_enable_x64", True)
    root = Path(directory)
    if mode == "export":
        root.mkdir(parents=True, exist_ok=True)
        (root / "requirements.txt").write_text(
            f"jax=={jax.__version__}\njaxlib=={jaxlib.__version__}\n", encoding="utf-8"
        )
    elif mode != "check":
        raise ValueError("mode must be export or check")
    for dtype in ("float32", "float64"):
        metadata_path = root / dtype / "metadata.json"
        if mode == "export":
            weights = jnp.array([[2.0, -1.0], [0.5, 3.0]], dtype=dtype)
            export(
                metadata_path,
                lambda x: x @ weights,
                input_size=2,
                output_size=2,
                dtype=dtype,
                framework="custom",
                example_input=np.ones(2),
            )
        predictor = load(metadata_path)
        for _ in range(20):
            for values in ([1.0, 2.0], [0.1, -0.3]):
                a, b = values
                expected = np.array([2 * a + 0.5 * b, -a + 3 * b])
                tolerance = 1e-6 if dtype == "float32" else 1e-12
                np.testing.assert_allclose(
                    predictor.predict(values), expected, atol=tolerance, rtol=0
                )
                np.testing.assert_allclose(
                    predictor.predict_jax(jnp.array(values, dtype=dtype)),
                    expected,
                    atol=tolerance,
                    rtol=0,
                )
        print(f"{mode}: {dtype} portable metadata inference passed")
    assert not any(
        name in sys.modules for name in ("kkthn", "daehn", "nlpoptnet", "flax", "orbax")
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
