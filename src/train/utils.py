"""Minimal model utility retained by the IDR-SSCL implementations."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def wrap_pretrained_model(
    c_extractor_arch: Callable,
    pretrain_model: bool = False,
) -> Callable[[int | None], nn.Module]:
    """Create a torchvision backbone and optionally replace its output head.

    The original project used this helper for the CEM and CBM ResNet50
    integration. Weights are supplied by torchvision at user runtime and are
    not distributed in this repository.
    """

    def build(output_dim: int | None = None) -> nn.Module:
        if c_extractor_arch is not resnet50:
            raise TypeError("The public IDR-SSCL implementation supports ResNet50")
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrain_model else None
        model = c_extractor_arch(weights=weights)

        if output_dim is None:
            return model
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model

    return build
