# WoundVision AI

Experimental wound tissue segmentation REST API powered by **V6 Two-Stage U-Net (`ResNet18` Binary Wound Localizer + `ResNet34` ROI Tissue Segmenter)** running on **ONNX Runtime** (CPU, zero PyTorch in production).

## Model Comparison (LUTSeg Validation — 30 images, patient-level split)
| Metric | `V5-Augmented` (Single-Stage) | **`V6-TwoStage`** *(Current)* | Delta |
| :--- | :---: | :---: | :---: |
| **Pixel Accuracy** | `90.54%` | **`92.19%`** | `+1.65%` |
| **Wound Dice (Binary)** | `81.18%` | **`85.94%`** | `+4.76%` |
| **Clinical-3 Macro Dice** *(Slough / Granulation / Necrotic)* | `59.45%` | **`64.04%`** | `+4.59%` |
| **Macro Tissue Dice (5 classes)** | `43.16%` | **`47.04%`** | `+3.88%` |
| Granulation Dice | `74.89%` | **`78.02%`** | `+3.13%` |
| Necrotic Dice | `60.91%` | **`65.91%`** | `+5.00%` |
| Slough Dice | `42.54%` | **`48.18%`** | `+5.64%` |
| Epithelial Dice | `37.43%` | **`43.09%`** | `+5.66%` |
| Normal Skin False Positive Rate | — | **`0.0%`** | — |

## Architecture
1. **Stage 1 (`woundvision_v6_stage1_wound.onnx`, `54.66 MB`)**: `U-Net ResNet18` (`320×320`, H-Flip TTA) localizes the wound boundary and rejects healthy skin (`0.0%` FPR).
2. **ROI Auto-Crop**: Expands the Stage 1 bounding box into a square ROI (`+18%` margin) to boost effective tissue resolution by 3–5×.
3. **Stage 2 (`woundvision_v6_stage2_tissue.onnx`, `93.22 MB`)**: `U-Net ResNet34` (`320×320`) classifies tissue classes (`epithelial`, `slough`, `granulation`, `necrotic`, `other`) inside the wound bed.

## API Endpoints
- `GET /` — Service metadata
- `GET /health` — Health, model readiness, uptime, and RSS memory check (`200` when ready, `503` otherwise)
- `POST /predict` — Upload wound image (`multipart/form-data`, field `file`)
  - Optional query params: `opacity` (`0.0`–`1.0`, default `0.5`), `include_overlay` (`true`/`false`, default `true`)
  - Optional header: `X-API-Key` (enforced when `API_KEY` env var is set)
  - Returns `request_id`, `timings_ms` (`decode`, `stage1`, `stage2`, `inference`, `overlay`, `total`), `confidence` (`stage1_max_prob`, `stage1_mean_prob`, `stage2_confidence`, `stage2_entropy`), `analysis`, and optional `overlay`
- `GET /metrics/summary` — Rolling performance & drift summary (`hours` query param, default `24`, `0` = all-time; protected by `METRICS_API_KEY` or `API_KEY`)
- `GET /metrics/recent` — Recent prediction metadata events (`limit`, `only_errors`, `low_confidence_only`; protected by `METRICS_API_KEY` or `API_KEY`)

## Quick Deploy (Docker / Easypanel)
1. Mount a persistent **Volume** to `/app/models` (ONNX weights auto-download once from `salmanid/woundvision-ai` on Hugging Face).
2. *(Optional for persistent telemetry)* Mount a second volume to `/app/telemetry` **or** set `METRICS_DB_PATH=/app/models/metrics.db` to reuse the existing volume.
3. Set `PORT=7860`, `CORS_ORIGINS`, `API_KEY`, `METRICS_API_KEY`, and `ORT_INTRA_THREADS=2` (recommended container limit: `1024 MB` RAM, `2` vCPU).
4. Build with `Dockerfile`.

## Dataset, Attribution & Citation
- **Primary Multi-Class Tissue Dataset**: **[LUTSeg](https://huggingface.co/datasets/ksanchez84/LUTSeg)** (*LUTSeg: A Longitudinal Multi-Expert Dataset for Ulcer Tissue Segmentation*, Eleventh ISIC Skin Image Analysis Workshop @ MICCAI 2026) — licensed under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** ([Paper](https://huggingface.co/papers/2608.25866)). Source images originate from **[CO2Wounds](https://data.mendeley.com/datasets/nkw5gx57hw/1)** ([DOI: 10.1016/j.compbiomed.2023.107753](https://doi.org/10.1016/j.compbiomed.2023.107753)).
- **Stage 1 Research Track Auxiliary Datasets**:
  - [praveen202/DFU_Datasets](https://huggingface.co/datasets/praveen202/DFU_Datasets) (FUSeg 2021 & DFUC2022 research reference cohorts — research/non-commercial demo use)
  - [rupeshsah/wound](https://huggingface.co/datasets/rupeshsah/wound) (`normal_*.png` healthy skin hard negatives, Apache 2.0)

## Disclaimer
Experimental research prototype. Not clinically validated, not a medical device, and not intended for diagnosis or clinical treatment decisions.
