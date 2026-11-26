# WoundVision AI

Experimental wound tissue segmentation REST API powered by **U-Net ResNet34 (`V5-Augmented`)** running on **ONNX Runtime** (CPU, ~272 MB peak RAM).

## Model Summary
- **Version**: `V5-Augmented`
- **Architecture**: U-Net (ResNet34 encoder)
- **Runtime**: ONNX Runtime (CPU, FP32, `93.22 MB`, 100% pixel parity vs PyTorch)
- **Classes**: `background`, `epithelial`, `slough`, `granulation`, `necrotic`, `other`

### Validation Metrics (LUTSeg, 30 images, patient-level split)
| Metric | Score |
| :--- | :---: |
| Pixel Accuracy | `90.54%` |
| Wound Dice (Binary) | `81.18%` |
| Clinical-3 Macro Dice (Slough / Granulation / Necrotic) | `59.45%` |
| Macro Tissue Dice (5 tissue classes) | `43.16%` |
| Granulation Dice | `74.89%` |
| Necrotic Dice | `60.91%` |
| Slough Dice | `42.54%` |
| Epithelial Dice | `37.43%` |

## API Endpoints
- `GET /` — Service metadata
- `GET /health` — Health & model readiness check (`200` when ready, `503` otherwise)
- `POST /predict` — Upload wound image (`multipart/form-data`, field `file`)
  - Optional query params: `opacity` (`0.0`–`1.0`, default `0.5`), `include_overlay` (`true`/`false`, default `true`)
  - Optional header: `X-API-Key` (enforced when `API_KEY` env var is set)

## Quick Deploy (Docker / Easypanel)
1. Set environment variables (see `.env.example`):
   - `PORT=7860`
   - `CORS_ORIGINS=https://your-web-domain.com`
   - `API_KEY=your-secret-key`
   - `MODEL_URL=https://huggingface.co/salmanid/woundvision-ai/resolve/main/woundvision_v5_augmented.onnx`
   - `ORT_INTRA_THREADS=2`
2. Mount a persistent volume to `/app/models` so the ONNX file is cached across restarts.
3. Build with `Dockerfile` (recommended container limits: `768 MB` RAM, `2` vCPU).

## Dataset, Attribution & Citation
This model was trained and evaluated using the **[LUTSeg](https://huggingface.co/datasets/ksanchez84/LUTSeg)** dataset, released under the **[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**.

- **Dataset**: [ksanchez84/LUTSeg](https://huggingface.co/datasets/ksanchez84/LUTSeg) (*LUTSeg: A Longitudinal Multi-Expert Dataset for Ulcer Tissue Segmentation*, Eleventh ISIC Skin Image Analysis Workshop @ MICCAI 2026)
- **Paper**: [LUTSeg Paper (Hugging Face Papers)](https://huggingface.co/papers/2608.25866)
- **Source Images**: [CO2Wounds Dataset](https://data.mendeley.com/datasets/nkw5gx57hw/1) ([DOI: 10.1016/j.compbiomed.2023.107753](https://doi.org/10.1016/j.compbiomed.2023.107753))
- **Dataset License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Disclaimer
Experimental research prototype. Not clinically validated, not a medical device, and not intended for diagnosis or clinical treatment decisions.
