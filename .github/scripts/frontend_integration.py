"""Real frozen frontend consumers train briefly and reload metadata in a fresh process."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import jax
import jax.numpy as jnp
import numpy as np


def reload_check(path, inputs, expected, framework, full_metadata=None):
    code = """
import importlib.abc, json, sys
import numpy as np
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'kkthn','daehn','nlpoptnet','flax','orbax'}:
   raise AssertionError('Deployment imported '+fullname)
sys.meta_path.insert(0,Block())
from ohagi.inference import load
predictor=load(sys.argv[1])
x=json.loads(sys.argv[2]); expected=json.loads(sys.argv[3])
for _ in range(25):
 np.testing.assert_allclose(predictor.predict(x),expected,atol=1e-8,rtol=1e-7)
 np.testing.assert_allclose(predictor.predict_jax(x),expected,atol=1e-8,rtol=1e-7)
"""
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            str(path),
            json.dumps(np.asarray(inputs).tolist()),
            json.dumps(np.asarray(expected).tolist()),
        ],
        check=True,
        timeout=180,
    )
    if framework == "kkthn":
        from kkthn import KKTHardNet

        facade = KKTHardNet().load(full_metadata or path)
    elif framework == "daehn":
        from daehn import DAEHardNet

        facade = DAEHardNet().load(path)
    else:
        from nlpoptnet import NLPOptNet

        facade = NLPOptNet().load(full_metadata or path)
    np.testing.assert_allclose(
        np.asarray(facade.predict(np.asarray(inputs)[None])).reshape(-1),
        np.asarray(expected).reshape(-1),
        atol=1e-8,
        rtol=1e-7,
    )


def kkthn(root):
    from kkthn import KKTHardNet, Projection, MLP, Loss

    x = np.linspace(0.4, 1.2, 8)
    data = root / "data.csv"
    np.savetxt(
        data,
        np.column_stack((x, 0.6 * x, 0.4 * x)),
        delimiter=",",
        header="p,u,v",
        comments="",
    )
    problem = Projection(method="chambolle-pock")
    p = problem.add_parameter(["p"])
    y = problem.add_variable(["u", "v"])
    problem.constraints.add(y.u + y.v == p.p)
    model = KKTHardNet(
        "surrogate",
        MLP(hidden_sizes=(4,)),
        Loss.mse(),
        problem,
        name="ci",
        train={
            "epochs": 2,
            "batch_size": 2,
            "seed": 7,
            "pretrain": False,
            "print_every": 0,
            "output_dir": str(root / "models"),
        },
    )
    model.dataset(parameters=data, variables=data)
    result = model.model(projection={"max_iterations": 1000, "tolerance": 1e-9})
    assert len(result["history"]) == 2
    path = root / "metadata.json"
    model.export_inference(path)
    from ohagi.inference import load

    expected = load(path).predict([0.7])
    np.testing.assert_allclose(expected.sum(), 0.7, atol=1e-7)
    reload_check(path, [0.7], expected, "kkthn", result["metadata_path"])
    return {"epochs": 2, "prediction": expected.tolist()}


def daehn(root):
    from daehn.dae import (
        Coordinate,
        DifferentialState,
        DerivativeSpec,
        EquationBlock,
        DAESystem,
        TaylorNeighborhood,
        ForwardDifference,
        CentralDifference,
        compile_dae,
        DAETrainer,
        DAETrainingConfig,
    )
    from daehn import MLPBackbone
    from ohagi.inference import export, load

    records = {}
    for name, strategy in [
        ("taylor", TaylorNeighborhood(deltas=(0.01,), order=1)),
        ("forward", ForwardDifference(deltas=(0.01,))),
        ("central", CentralDifference(deltas=(0.01,))),
    ]:
        state = DifferentialState("y")
        system = DAESystem(
            coordinates=(Coordinate("t"),),
            differential_states=(state,),
            derivatives=(DerivativeSpec(state, (1,)),),
            differential_equations=EquationBlock(
                lambda c, s, d, p: jnp.array([d[0] + s[0]]),
                size=1,
                affine_in_projection_variables=True,
            ),
            algebraic_equalities=EquationBlock(affine_in_projection_variables=True),
            algebraic_inequalities=EquationBlock(affine_in_projection_variables=True),
            name="decay",
        )
        compiled = compile_dae(system, discretization=strategy)
        trainer = DAETrainer(
            compiled=compiled,
            prepared_projection=compiled.prepare_projection(
                batch_size=2, method="linear"
            ),
            backbone=MLPBackbone(hidden_sizes=(4,)),
            config=DAETrainingConfig(
                epochs=2,
                pretrain_epochs=1,
                batch_size=2,
                hidden_sizes=(4,),
                seed=7,
                print_every=0,
            ),
        )
        x = np.linspace(0, 1, 8)[:, None]
        p = np.empty((8, 0))
        y = np.exp(-x)
        result = trainer.fit(x[:4], p[:4], y[:4], -y[:4], x[4:], p[4:], y[4:], -y[4:])
        decoded, *_ = trainer.model.forward(
            result.parameters, jnp.asarray(x[:2]), jnp.asarray(p[:2])
        )
        np.testing.assert_allclose(
            decoded["states"] + decoded["derivatives"], 0, atol=1e-7
        )

        def graph(inputs):
            decoded, *_ = trainer.model.forward(
                result.parameters, jnp.repeat(inputs, 2, axis=0), jnp.empty((2, 0))
            )
            return decoded["states"][:1]

        path = root / name / "metadata.json"
        export(path, graph, input_size=1, output_size=1, framework="daehn")
        expected = load(path).predict([0.7])
        reload_check(path, [0.7], expected, "daehn")
        records[name] = {"epochs": len(result.history), "prediction": expected.tolist()}
    return records


def nlpoptnet(root):
    from nlpoptnet import NLPOptNet
    from ohagi.inference import load

    records = {}
    for mode in ("fixed", "accelerated"):
        model = NLPOptNet(
            name="ci_" + mode,
            config={
                "epochs": 2,
                "batch_size": 2,
                "hidden_size": 4,
                "hidden_layers": 1,
                "learning_rate": 1e-3,
                "cp_mode": mode,
                "cp_iters": 200,
                "knorm_iters": 5,
                "adjoint_iters": 15,
                "verbose": False,
                "seed": 7,
                "warmup": False,
            },
        )
        p = model.add_parameter(["p"])
        y = model.add_variable(["u", "v"])
        model.objective((y.u - p.p) ** 2 + 0.5 * y.v**2 + 0.01 * y.v**4)
        model.constraints.equality.add(y.u + y.v == p.p)
        model.constraints.inequality.add(y.u <= 1.5)
        model.constraints.box.add(y.u >= 0, y.v >= 0, y.u <= 2, y.v <= 2)
        model.box(lower=0.4, upper=1.4, num_samples=8)
        model.build()
        result = model.optimize()
        assert Path(result["metadata_path"]).is_file()
        path = root / mode / "metadata.json"
        model.export_inference(path)
        expected = load(path).predict([0.7])
        assert np.all(np.isfinite(expected))
        np.testing.assert_allclose(expected.sum(), 0.7, atol=2e-4)
        reload_check(path, [0.7], expected, "nlpoptnet", result["metadata_path"])
        records[mode] = {"epochs": 2, "prediction": expected.tolist()}
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=["kkthn", "daehn", "nlpoptnet"])
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    root = Path(__file__).resolve().parents[2]
    fixtures = root / ".github/fixtures/frontends"
    for name, digest in json.loads((fixtures / "manifest.json").read_text()).items():
        assert (
            hashlib.sha256(
                (fixtures / name).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            == digest
        ), name
    with tempfile.TemporaryDirectory() as directory:
        original = Path.cwd()
        os.chdir(directory)
        try:
            result = globals()[args.framework](Path(directory))
        finally:
            os.chdir(original)
    (root / f"integration-{args.framework}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
