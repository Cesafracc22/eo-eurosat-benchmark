from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from eo_bench.config import save_config
from eo_bench.data import build_dataloaders, build_datasets
from eo_bench.eval import evaluate_model
from eo_bench.metrics import compute_classification_metrics
from eo_bench.models import build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _run_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        if criterion is not None:
            losses.append(float(criterion(logits, labels).item()))
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return mean_loss, y_true, y_pred


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    return total / max(n, 1)


def run_training(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    datasets = build_datasets(
        root=cfg.get("data_root", "data"),
        name=cfg["dataset"],
        bands=cfg["bands"],
        download=bool(cfg.get("download", True)),
    )
    loaders = build_dataloaders(
        datasets,
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 0)),
    )

    model = build_model(
        name=cfg.get("model", "resnet18"),
        in_channels=datasets["in_channels"],
        num_classes=datasets["num_classes"],
        weights=cfg.get("weights", "none"),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    best_val_f1 = -1.0
    best_path = out_dir / "best.pt"
    epochs = int(cfg.get("epochs", 10))
    patience = int(cfg.get("early_stopping_patience", 5))
    stale = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        val_loss, y_true, y_pred = _run_split(
            model, loaders["val"], device, criterion=criterion
        )
        val_metrics = compute_classification_metrics(
            y_true, y_pred, class_names=datasets.get("class_names")
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        print(
            f"epoch {epoch:03d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
                    "in_channels": datasets["in_channels"],
                    "num_classes": datasets["num_classes"],
                    "class_names": datasets.get("class_names"),
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch} (patience={patience})")
                break

    # Final test evaluation with best checkpoint
    test_metrics = evaluate_model(best_path, loaders["test"], device)
    summary = {
        "dataset": cfg["dataset"],
        "bands": cfg["bands"],
        "weights": cfg.get("weights"),
        "seed": cfg.get("seed", 42),
        "device": str(device),
        "num_parameters": count_parameters(model),
        "best_val_macro_f1": best_val_f1,
        "test": test_metrics,
    }

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("wrote", out_dir / "metrics.json")
    return summary
