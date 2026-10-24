"""Historical core projection imports point to PURIN."""

ALIASES = {
    "ohagi._utils.validation": ("purin._utils.validation", False),
    "ohagi.autodiff": ("purin.autodiff", True),
    "ohagi.autodiff.complementarity": ("purin.autodiff.complementarity", False),
    "ohagi.autodiff.fixed_point": ("purin.autodiff.fixed_point", False),
    "ohagi.compilation": ("purin.compilation", True),
    "ohagi.differentiation": ("purin.differentiation", True),
    "ohagi.differentiation.policies": ("purin.differentiation.policies", False),
    "ohagi.execution.runtime": ("purin.execution.runtime", False),
    "ohagi.linalg": ("purin.linalg", True),
    "ohagi.linalg.equilibration": ("purin.linalg.equilibration", False),
    "ohagi.linalg.operators": ("purin.linalg.operators", False),
    "ohagi.linalg.sparse": ("purin.linalg.sparse", True),
    "ohagi.linalg.sparse.assembly": ("purin.linalg.sparse.assembly", False),
    "ohagi.linalg.sparse.conversion": ("purin.linalg.sparse.conversion", False),
    "ohagi.linalg.sparse.minres": ("purin.linalg.sparse.minres", False),
    "ohagi.linalg.sparse.operators": ("purin.linalg.sparse.operators", False),
    "ohagi.linalg.sparse.types": ("purin.linalg.sparse.types", False),
    "ohagi.problems": ("purin.problems", True),
    "ohagi.problems.canonical": ("purin.problems.canonical", False),
    "ohagi.problems.compile": ("purin.problems.compile", False),
    "ohagi.problems.expressions": ("purin.problems.expressions", False),
    "ohagi.problems.functional": ("purin.problems.functional", False),
    "ohagi.problems.problem": ("purin.problems.problem", False),
    "ohagi.problems.projection": ("purin.problems.projection", False),
    "ohagi.problems.qp": ("purin.problems.qp", False),
    "ohagi.problems.sparsity": ("purin.problems.sparsity", False),
    "ohagi.problems.symbolic": ("purin.problems.symbolic", False),
    "ohagi.problems.types": ("purin.problems.types", False),
    "ohagi.projections": ("purin.projections", True),
    "ohagi.projections.adapters": ("purin.projections.adapters", False),
    "ohagi.projections.diagnostics": ("purin.projections.diagnostics", False),
    "ohagi.projections.distance": ("purin.projections.distance", False),
    "ohagi.projections.local_quadratic": ("purin.projections.local_quadratic", False),
    "ohagi.projections.sequential": ("purin.projections.sequential", False),
    "ohagi.solvers": ("purin.solvers", True),
    "ohagi.solvers.admm": ("purin.solvers.admm", True),
    "ohagi.solvers.admm.config": ("purin.solvers.admm.config", False),
    "ohagi.solvers.admm.factor": ("purin.solvers.admm.factor", False),
    "ohagi.solvers.admm.kernels": ("purin.solvers.admm.kernels", False),
    "ohagi.solvers.admm.rows": ("purin.solvers.admm.rows", False),
    "ohagi.solvers.admm.solver": ("purin.solvers.admm.solver", False),
    "ohagi.solvers.api": ("purin.solvers.api", False),
    "ohagi.solvers.capabilities": ("purin.solvers.capabilities", False),
    "ohagi.solvers.chambolle_pock": ("purin.solvers.chambolle_pock", True),
    "ohagi.solvers.chambolle_pock.bounds": (
        "purin.solvers.chambolle_pock.bounds",
        False,
    ),
    "ohagi.solvers.chambolle_pock.config": (
        "purin.solvers.chambolle_pock.config",
        False,
    ),
    "ohagi.solvers.chambolle_pock.dense": ("purin.solvers.chambolle_pock.dense", False),
    "ohagi.solvers.chambolle_pock.dense_implicit": (
        "purin.solvers.chambolle_pock.dense_implicit",
        False,
    ),
    "ohagi.solvers.chambolle_pock.dense_layer": (
        "purin.solvers.chambolle_pock.dense_layer",
        False,
    ),
    "ohagi.solvers.chambolle_pock.kernels": (
        "purin.solvers.chambolle_pock.kernels",
        False,
    ),
    "ohagi.solvers.chambolle_pock.solver": (
        "purin.solvers.chambolle_pock.solver",
        False,
    ),
    "ohagi.solvers.config": ("purin.solvers.config", False),
    "ohagi.solvers.diagonal_qp": ("purin.solvers.diagonal_qp", False),
    "ohagi.solvers.fischer_burmeister": ("purin.solvers.fischer_burmeister", True),
    "ohagi.solvers.fischer_burmeister.common": (
        "purin.solvers.fischer_burmeister.common",
        False,
    ),
    "ohagi.solvers.fischer_burmeister.complementarity": (
        "purin.solvers.fischer_burmeister.complementarity",
        False,
    ),
    "ohagi.solvers.fischer_burmeister.config": (
        "purin.solvers.fischer_burmeister.config",
        False,
    ),
    "ohagi.solvers.fischer_burmeister.dense_semismooth": (
        "purin.solvers.fischer_burmeister.dense_semismooth",
        False,
    ),
    "ohagi.solvers.fischer_burmeister.newton": (
        "purin.solvers.fischer_burmeister.newton",
        False,
    ),
    "ohagi.solvers.fischer_burmeister.operators": (
        "purin.solvers.fischer_burmeister.operators",
        False,
    ),
    "ohagi.solvers.fischer_burmeister.reduced": (
        "purin.solvers.fischer_burmeister.reduced",
        False,
    ),
    "ohagi.solvers.fischer_burmeister.solver": (
        "purin.solvers.fischer_burmeister.solver",
        False,
    ),
    "ohagi.solvers.implicit": ("purin.solvers.implicit", False),
    "ohagi.solvers.linear_equality": ("purin.solvers.linear_equality", True),
    "ohagi.solvers.linear_equality.config": (
        "purin.solvers.linear_equality.config",
        False,
    ),
    "ohagi.solvers.linear_equality.solver": (
        "purin.solvers.linear_equality.solver",
        False,
    ),
    "ohagi.solvers.metric": ("purin.solvers.metric", False),
    "ohagi.solvers.primal_dual_interior_point": (
        "purin.solvers.primal_dual_interior_point",
        True,
    ),
    "ohagi.solvers.primal_dual_interior_point.assembly": (
        "purin.solvers.primal_dual_interior_point.assembly",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.config": (
        "purin.solvers.primal_dual_interior_point.config",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.derivatives": (
        "purin.solvers.primal_dual_interior_point.derivatives",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.newton": (
        "purin.solvers.primal_dual_interior_point.newton",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.operators": (
        "purin.solvers.primal_dual_interior_point.operators",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.reference": (
        "purin.solvers.primal_dual_interior_point.reference",
        True,
    ),
    "ohagi.solvers.primal_dual_interior_point.reference.bounds": (
        "purin.solvers.primal_dual_interior_point.reference.bounds",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.reference.filter": (
        "purin.solvers.primal_dual_interior_point.reference.filter",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.reference.iteration": (
        "purin.solvers.primal_dual_interior_point.reference.iteration",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.reference.line_search": (
        "purin.solvers.primal_dual_interior_point.reference.line_search",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.reference.newton": (
        "purin.solvers.primal_dual_interior_point.reference.newton",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.reference.residuals": (
        "purin.solvers.primal_dual_interior_point.reference.residuals",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.reference.state": (
        "purin.solvers.primal_dual_interior_point.reference.state",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.reference.step": (
        "purin.solvers.primal_dual_interior_point.reference.step",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.residuals": (
        "purin.solvers.primal_dual_interior_point.residuals",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.scalar": (
        "purin.solvers.primal_dual_interior_point.scalar",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.solver": (
        "purin.solvers.primal_dual_interior_point.solver",
        False,
    ),
    "ohagi.solvers.primal_dual_interior_point.types": (
        "purin.solvers.primal_dual_interior_point.types",
        False,
    ),
    "ohagi.solvers.registry": ("purin.solvers.registry", False),
    "ohagi.solvers.semismooth": ("purin.solvers.semismooth", False),
    "ohagi.solvers.types": ("purin.solvers.types", False),
}
