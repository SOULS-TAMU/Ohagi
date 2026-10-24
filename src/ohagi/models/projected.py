"""Prediction composition independent of datasets, training loops, and I/O."""

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class OutputLayout:
    """Primal and equality/inequality head dimensions for network output splitting."""

    primal: int
    equalities: int = 0
    inequalities: int = 0

    def __post_init__(self):
        if self.primal < 1 or min(self.equalities, self.inequalities) < 0:
            raise ValueError("invalid primal/dual output layout")

    def split(self, output):
        """Partition the final network axis into primal and multiplier arrays."""
        if isinstance(output, tuple):
            if len(output) != 3:
                raise ValueError(
                    "tuple backbone output must be (primal, equality_duals, inequality_duals)"
                )
            return output
        expected = self.primal + self.equalities + self.inequalities
        if output.shape[-1] != expected:
            raise ValueError("backbone output width does not match layout")
        return (
            output[..., : self.primal],
            output[..., self.primal : self.primal + self.equalities],
            output[..., self.primal + self.equalities :],
        )


@dataclass(frozen=True)
class ProjectedModel:
    """Compose a backbone with a projection resource using the shared model contract."""

    backbone: object
    projection: object = None
    layout: OutputLayout | None = None
    prepare_inputs: object = None
    output_transform: object = None
    return_projection: bool = False
    prepare_projection: object = None
    return_context: bool = False

    def apply(self, params, state, inputs, *, context=None, key=None, training=False):
        """Evaluate the backbone, project its outputs and return outputs plus model state."""
        context = inputs if context is None else context
        if self.prepare_inputs is not None:
            params, inputs, context = self.prepare_inputs(params, inputs, context)
        output, new_state = self.backbone.apply(
            params, state, inputs, key=key, training=training
        )
        if self.output_transform is not None:
            raw, eq, iq = self.output_transform(output, context)
        elif self.layout is None:
            if isinstance(output, tuple):
                raw, eq, iq = output
            else:
                raw = output
                eq = jnp.empty(raw.shape[:-1] + (0,), raw.dtype)
                iq = eq
        else:
            raw, eq, iq = self.layout.split(output)
        if self.prepare_projection is not None:
            raw, eq, iq, context = self.prepare_projection(raw, eq, iq, context)
        r = None
        if self.projection is None:
            projected = raw
            pe = eq
            pi = iq
            valid = jnp.ones(raw.shape[0], bool)
        elif (
            eq.shape[-1]
            or iq.shape[-1]
            or getattr(self.projection, "accept_empty_duals", False)
        ) and hasattr(self.projection, "project_training"):
            r = self.projection.project_training(
                raw, context, multipliers_eq=eq, multipliers_ineq=iq
            )
            projected, pe, pi, valid = (
                r.y_tilde,
                getattr(r, "multipliers_eq", eq),
                getattr(r, "multipliers_ineq", iq),
                r.forward_gradient_eligible,
            )
        else:
            r = self.projection.solve(raw, context)
            projected, pe, pi, valid = (
                r.y,
                r.multipliers_eq,
                r.multipliers_ineq,
                r.gradient_eligible,
            )
            if eq.shape[-1] == 0:
                eq = jnp.zeros_like(pe)
            if iq.shape[-1] == 0:
                iq = jnp.zeros_like(pi)
        outputs = {
            "raw": raw,
            "projected": projected,
            "lambda_raw": eq,
            "mu_raw": iq,
            "lambda_projected": pe,
            "mu_projected": pi,
            "dual_valid": valid,
        }
        if self.return_projection:
            outputs["projection_result"] = r
        if self.return_context:
            outputs["context"] = context
        return outputs, new_state
