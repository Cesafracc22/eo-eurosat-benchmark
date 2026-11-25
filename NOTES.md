# Notes

Takeaways that shaped this repo. Sources: TorchGeo docs/tutorials, TorchGeo paper benchmarks, Corley et al. (CVPRW 2024) on remote-sensing pretrained-model evaluation, TerraTorch / Prithvi EuroSAT demos.

## What good EO benchmarks do

1. **Fixed splits** — TorchGeo EuroSAT uses Neumann et al. splits, not random shuffles of patches. Random splits can leak nearby tiles into val/test.
2. **Report uncertainty** — TorchGeo paper reports mean ± std over seeds. This repo starts with one seed; multi-seed is a follow-up.
3. **Compare weight regimes** — ImageNet RGB vs in-domain Sentinel-2 pretraining (MoCo/SeCo) is a standard comparison in EO transfer learning.
4. **Preprocessing matters** — resize/normalization change transfer numbers a lot (Corley et al.). Keep transforms explicit and identical across runs when comparing.
5. **Larger FMs are heavier** — Prithvi / TerraTorch need band subsetting, resize 64→224, and more dependencies. Useful extension; not required for a first working benchmark.

## Dataset choice

| Dataset | Size | Why |
|---|---|---|
| **EuroSAT100** | ~100 patches | Smoke / pipeline check |
| **EuroSAT** | 27k × 64×64 Sentinel-2 | Light, standard land-cover benchmark |
| BigEarthNet / SSL4EO | much larger | Heavier; better as a later scale-up |

## Band order caveat

TorchGeo `EuroSAT.BAND_SETS['all']` ends with **B8A**.
SSL4EO / `ResNet18_Weights.SENTINEL2_ALL_MOCO` expects **B8A after B08**.

This repo’s `bands: all` uses the MoCo order (`S2_MOCO_BANDS` in `data.py`).
Use `bands: eurosat_native` only if you intentionally want the dataset’s native order.

## Out of scope for now

- Agentic EO workflows
- Full Prithvi / Clay fine-tune
- SAR fusion
- Heavy Lightning / Hydra stacks
