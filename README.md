# WoundVision AI

Experimental wound tissue segmentation REST API powered by **U-Net ResNet34 (`V5-Augmented`)** running on **ONNX Runtime** (CPU, ~272 MB peak RAM).

## Model Summary
- **Version**: `V5-Augmented`
- **Architecture**: U-Net (ResNet34 encoder)
- **Runtime**: ONNX Runtime (CPU, FP32, 93.22 MB)
- **Validation (`clinical_3_dice`)**: `59.45%` (Granulation `74.89%`, Necrotic `60.91%`, Slough `42.54%`)
- **Classes**: `background`, `epithelial`, `slough`, `granulation`, `necrotic`, `other`

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

## Disclaimer
Experimental research prototype. Not clinically validated and not intended for diagnosis or clinical decision-making.
