"""Export and reload a single-instance JAX graph."""

import jax
import jax.numpy as jnp
import numpy as np
from ohagi.inference import export, load

jax.config.update("jax_enable_x64", True)


def predict_function(inputs):
    return jnp.concatenate((0.6 * inputs, 0.4 * inputs), axis=1)


export(
    "deployment/metadata.json",
    predict_function,
    input_size=1,
    output_size=2,
    framework="ohagi",
    dtype="float64",
)
predictor = load("deployment/metadata.json")
for value in [0.4, 0.7, 1.0]:
    prediction = predictor.predict([value])
    np.testing.assert_allclose(prediction, [0.6 * value, 0.4 * value], atol=1e-12)
    print(prediction)
