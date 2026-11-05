import jax
import jax.numpy as jnp
from purin import Distance, Projection
from ohagi import Adam, ProjectedModel, Trainer
from ohagi.nn import MLPAdapter, MLPBackbone
from ohagi.losses import supervised_mse
from ohagi.training.engine import create_train_state

jax.config.update("jax_enable_x64", True)

problem = Projection()
p = problem.add_parameter(["b"])
y = problem.add_variable(["u", "v"])
problem.objective = 0.0
problem.constraints.add(y.u + y.v == p.b)

projection = Distance(problem, "linear-equality").prepare(batch_size=4)
backbone = MLPAdapter(MLPBackbone(hidden_sizes=(16, 16)))
model = ProjectedModel(backbone, projection)
optimizer = Adam(learning_rate=1e-3)
state = create_train_state(
    backbone, optimizer, jax.random.PRNGKey(0), input_dim=1, output_dim=2
)
data = {
    "inputs": jnp.ones((8, 1)),
    "theta": jnp.ones((8, 1)),
    "solution": jnp.tile(jnp.array([0.8, 0.2]), (8, 1)),
}
result = Trainer(model, supervised_mse(), optimizer).fit(
    state, data, epochs=2, batch_size=4
)
outputs, _ = model.apply(
    result.state.parameters, result.state.model_state, jnp.ones((4, 1))
)
print(outputs["projected"])

# Check the declared physical/output contract and model transport.

import numpy as np

assert outputs["projected"].shape == (4, 2)
np.testing.assert_allclose(np.sum(outputs["projected"], axis=1), 1.0, atol=1e-9)
