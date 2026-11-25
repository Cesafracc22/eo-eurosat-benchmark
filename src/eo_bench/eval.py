from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from eo_bench.metrics import compute_classification_metrics
from eo_bench.models import build_model


@torch.no_grad()
def evaluate_model(
    checkpoint_path: str | Path,
    loader: torch.utils.data.DataLoader,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    model = build_model(
        name=cfg.get("model", "resnet18"),
        in_channels=int(ckpt["in_channels"]),
        num_classes=int(ckpt["num_classes"]),
        weights="none",  # load trained weights from checkpoint only
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        preds = model(images).argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return compute_classification_metrics(
        y_true, y_pred, class_names=ckpt.get("class_names")
    )
