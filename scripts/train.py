#!/usr/bin/env python3
"""Train one experiment from a YAML config.

Examples:
  python scripts/train.py --config configs/smoke_eurosat100.yaml
  python scripts/train.py --config configs/imagenet_rgb.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eo_bench.config import load_config
from eo_bench.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EuroSAT EO benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(cfg)


if __name__ == "__main__":
    main()
