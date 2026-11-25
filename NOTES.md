# Notes from similar projects

Short takeaways that shaped this repo. Sources: TorchGeo docs/tutorials, TorchGeo paper benchmarks, Corley et al. (CVPRW 2024) on RS pretrained-model evaluation, TerraTorch/Prithvi EuroSAT demos.

## What good AI4EO benchmarks do

1. **Fixed geographic-aware splits** — TorchGeo EuroSAT uses Neumann et al. splits, not random shuffles of patches. Random splits leak nearby tiles into val/test.
2. **Report uncertainty** — TorchGeo paper reports mean ± std over seeds. We start with one seed; multi-seed is a follow-up.
3. **Compare weight regimes** — ImageNet RGB vs in-domain Sentinel-2 pretraining (MoCo/SeCo) is a standard, readable story for Φ-lab.
4. **Preprocessing matters** — resize/normalization change transfer numbers a lot (Corley et al.). Keep transforms explicit and identical across runs when comparing.
5. **GFMs are heavier** — Prithvi/TerraTorch need band subsetting (6 HLS-like bands), resize 64→224, more deps. Valuable next step; not required for a clean first week.

## Dataset choice

| Dataset | Size | Why |
|---|---|---|
| **EuroSAT100** | ~100 patches | Smoke / pipeline check |
| **EuroSAT** | 27k × 64×64 Sentinel-2 | Light, standard land-cover benchmark |
| BigEarthNet / SSL4EO | much larger | Too heavy for a 1-week prep sprint |

## Internship Topic 1 mapping

| Φ-lab task | Where it shows up here |
|---|---|
| Preparing EO datasets for AI pipelines | `data.py`, band selection RGB vs 13-band MSI, fixed splits |
| Training & benchmarking EO models | two configs + `metrics.json` |
| AI design / optimisation | early stopping, LR schedule, AdamW; keep ablations small |
| Geospatial foundation models (asset) | Sentinel-2 MoCo as EO pretraining; Prithvi listed as next step |

## Band order caveat (important)

TorchGeo `EuroSAT.BAND_SETS['all']` ends with **B8A**.
SSL4EO / `ResNet18_Weights.SENTINEL2_ALL_MOCO` expects **B8A after B08**.

This repo’s `bands: all` uses the MoCo order (`S2_MOCO_BANDS` in `data.py`).
Use `bands: eurosat_native` only if you intentionally want the dataset’s native order.

## Deliberate non-goals (week 1)

- Agentic EO workflows
- Full Prithvi/Clay fine-tune
- SAR fusion
- Heavy Lightning/Hydra stacks
