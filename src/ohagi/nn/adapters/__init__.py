"""Adapters connecting functional and Flax backbones to shared model contracts."""

from ohagi.nn.backbone import (
    FunctionalBackbone as FunctionalBackbone,
    MLPAdapter as MLPAdapter,
)
from .flax import FlaxLinenBackbone as FlaxLinenBackbone
from .flax_functional import (
    FlaxFunctionalBackbone as FlaxFunctionalBackbone,
    FlaxMLPBackbone as FlaxMLPBackbone,
)
