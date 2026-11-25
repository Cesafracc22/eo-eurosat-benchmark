# eo-eurosat-benchmark

Reproducible land-cover classification on **Sentinel-2 / EuroSAT**.

Compares a generic ImageNet ResNet-18 (RGB) with an in-domain Sentinel-2 MoCo ResNet-18 (13 MSI bands) under the same training and evaluation protocol.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
# tiny subset — check the pipeline first
python scripts/train.py --config configs/smoke_eurosat100.yaml

# full EuroSAT baselines
python scripts/train.py --config configs/imagenet_rgb.yaml
python scripts/train.py --config configs/sentinel2_moco.yaml
```

Re-evaluate a checkpoint:

```bash
python scripts/evaluate.py --checkpoint results/imagenet_rgb/best.pt
```

## Layout

```
configs/          experiment YAML
src/eo_bench/     data, models, train, eval, metrics
scripts/          CLI entrypoints
data/             downloaded datasets (gitignored)
results/          metrics + checkpoints (checkpoints gitignored)
NOTES.md          design notes and references
```

Each run writes `results/<name>/config.yaml`, `metrics.json`, `history.json`, `best.pt`.

## Metrics

Primary: **accuracy** and **macro-F1** on the official test split. Confusion matrix is stored in `metrics.json`.

## Design choices

- Plain PyTorch training loop
- TorchGeo for EuroSAT and Sentinel-2 pretrained weights
- Fixed TorchGeo / Neumann splits (not random file splits)
- One seed in configs; multi-seed is optional follow-up
- MSI band order aligned with SSL4EO / MoCo (see `NOTES.md`)

## Optional next steps

- 3 seeds → mean ± std
- few-shot (5% labels)
- geospatial foundation model linear probe (e.g. Prithvi)

## License

Code: MIT. EuroSAT data follows its own license (see TorchGeo / EuroSAT authors).
