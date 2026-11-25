"""Model builders for the EuroSAT benchmark.

Weight regimes:
- ImageNet RGB: generic vision baseline
- Sentinel-2 MoCo (TorchGeo): in-domain EO pretraining on MSI
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights as TVResNet18Weights
from torchvision.models import resnet18


def build_model(
    name: str,
    in_channels: int,
    num_classes: int,
    weights: str = "none",
) -> nn.Module:
    if name != "resnet18":
        raise ValueError(f"Unsupported model: {name!r}")

    if weights == "imagenet":
        if in_channels != 3:
            raise ValueError("ImageNet weights require in_channels=3 (RGB).")
        model = resnet18(weights=TVResNet18Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if weights == "sentinel2_moco":
        # Lazy import: RGB-only workflows do not need torchgeo.models.
        from torchgeo.models import ResNet18_Weights as GeoWeights
        from torchgeo.models import resnet18 as geo_resnet18

        if in_channels != 13:
            raise ValueError("Sentinel-2 MoCo weights require in_channels=13.")
        # timm creates a fresh fc for num_classes; backbone weights load with strict=False.
        return geo_resnet18(
            weights=GeoWeights.SENTINEL2_ALL_MOCO,
            num_classes=num_classes,
        )

    if weights == "none":
        model = resnet18(weights=None)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(
        f"Unknown weights={weights!r}. Use 'imagenet', 'sentinel2_moco', or 'none'."
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
