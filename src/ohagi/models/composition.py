"""Common inference composition for stored network and callable projection specs."""

from .projected import ProjectedModel
from ohagi.nn.backbone import FunctionalBackbone
from purin.projections.adapters import CallableCorrection


def inference_graph(network_apply, params, transform, projection=None):
    """Capture parameters and model state in a pure JAX prediction callable."""

    def prepare(parameters, inputs, context):
        return parameters, transform(inputs), context

    def correct(context, target, equality, inequality):
        return projection(target, context), equality, inequality

    model = ProjectedModel(
        FunctionalBackbone(None, network_apply),
        None if projection is None else CallableCorrection(correct),
        prepare_inputs=prepare,
    )

    def graph(inputs):
        outputs, _ = model.apply(params, {}, inputs)
        return outputs["projected"]

    return graph
