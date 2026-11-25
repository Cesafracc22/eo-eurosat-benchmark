# eo-eurosat-benchmark

Small, reproducible land-cover benchmark on **Sentinel-2 / EuroSAT**.

Built as a one-week prep project for ESA Φ-lab **AI for Earth Observation**: dataset prep, model training, and a fair comparison between generic and EO-pretrained weights.

## Why this setup

- **EuroSAT** — 27k patches, 64×64, 10 classes, downloadable, standard in EO ML.
- **EuroSAT100** — tiny subset for smoke checks before a full run.
- **Two weight regimes**
  1. ImageNet ResNet-18 on RGB
  2. Sentinel-2 MoCo ResNet-18 on 13 MSI bands (TorchGeo)

That comparison matches Topic 1 language: prepare EO data, train/benchmark EO models, and touch in-domain EO pretraining (stepping stone toward geospatial FMs).

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run (do smoke first)

```bash
# 1) pipeline check (~minutes, tiny data)
python scripts/train.py --config configs/smoke_eurosat100.yaml

# 2) full baselines (download EuroSAT once into data/)
python scripts/train.py --config configs/imagenet_rgb.yaml
python scripts/train.py --config configs/sentinel2_moco.yaml
```

Re-evaluate a checkpoint:

```bash
python scripts/evaluate.py --checkpoint results/imagenet_rgb/best.pt
```

## Layout

```
configs/          experiment YAML (source of truth)
src/eo_bench/     data, models, train, eval, metrics
scripts/          CLI entrypoints
data/             downloaded datasets (gitignored)
results/          metrics + checkpoints (checkpoints gitignored)
NOTES.md          decisions from similar projects
```

Each run writes `results/<name>/config.yaml`, `metrics.json`, `history.json`, `best.pt`.

## Metrics

Primary: **accuracy** and **macro-F1** on the official test split. Confusion matrix is stored in `metrics.json`.

## Design choices

- Plain PyTorch (no Lightning) so the training loop is readable in an interview.
- TorchGeo only for datasets + Sentinel-2 weights.
- Fixed TorchGeo/Neumann splits (not random file splits).
- One seed in configs; multi-seed is a follow-up.

## Next (optional)

- 3 seeds → mean ± std
- few-shot (5% labels)
- Prithvi / TerraTorch linear probe on a 6-band subset

## License

Code: MIT. EuroSAT data follows its own license (see TorchGeo / EuroSAT authors).
