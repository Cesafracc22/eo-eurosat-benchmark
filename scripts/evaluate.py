#!/usr/bin/env python3
"""Evaluate a saved checkpoint on the test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eo_bench.config import load_config
from eo_bench.data import build_dataloaders, build_datasets
from eo_bench.eval import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EuroSAT checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        help="Optional config for data settings; defaults to checkpoint folder config.yaml",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    cfg_path = Path(args.config) if args.config else ckpt_path.parent / "config.yaml"
    cfg = load_config(cfg_path)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(ckpt_path, loaders["test"], device)

    out = ckpt_path.parent / "test_metrics.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1")}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
