"""EuroSAT loaders with fixed splits.

Splits follow Neumann et al. / TorchGeo defaults (not random file splits),
so nearby patches are less likely to leak across train/val/test.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT, EuroSAT100


DATASET_REGISTRY = {
    "eurosat": EuroSAT,
    "eurosat100": EuroSAT100,
}

# SSL4EO / TorchGeo MoCo expects B8A after B08 (not at the end).
# EuroSAT.BAND_SETS['all'] ends with B8A — wrong order for MoCo transfer.
S2_MOCO_BANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)


def _bands_for_mode(bands: str) -> tuple[str, ...]:
    if bands == "rgb":
        return EuroSAT.BAND_SETS["rgb"]
    if bands == "all":
        # Default MSI order for MoCo-compatible training in this repo.
        return S2_MOCO_BANDS
    if bands == "eurosat_native":
        return EuroSAT.BAND_SETS["all"]
    raise ValueError(
        f"Unknown bands mode: {bands!r}. Use 'rgb', 'all', or 'eurosat_native'."
    )


def build_datasets(
    root: str,
    name: str = "eurosat100",
    bands: str = "rgb",
    download: bool = True,
) -> dict[str, Any]:
    """Return train/val/test datasets sharing the same band selection."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(DATASET_REGISTRY)}")

    cls = DATASET_REGISTRY[name]
    band_tuple = _bands_for_mode(bands)

    common = dict(root=root, bands=band_tuple, download=download)
    train_ds = cls(split="train", **common)
    class_names = list(getattr(train_ds, "classes", [])) or None
    return {
        "train": train_ds,
        "val": cls(split="val", **common),
        "test": cls(split="test", **common),
        "num_classes": 10,
        "in_channels": len(band_tuple),
        "class_names": class_names,
    }


def _collate_torchgeo(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack TorchGeo sample dicts into a mini-batch."""
    images = torch.stack([sample["image"].float() for sample in batch])
    labels = torch.tensor([int(sample["label"]) for sample in batch], dtype=torch.long)
    return {"image": images, "label": labels}


def build_dataloaders(
    datasets: dict[str, Any],
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test"):
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=_collate_torchgeo,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
