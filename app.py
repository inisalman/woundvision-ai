"""Production FastAPI server for WoundVision AI V6-TwoStage (ONNX Runtime)."""

from contextlib import asynccontextmanager
from io import BytesIO
import base64
import hmac
import os
import time

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from inference import (
    load_model,
    predict,
    generate_overlay,
    calculate_tissue_percentage,
    DEVICE,
)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_PIXELS = 20_000_000           # 20 MP
API_KEY = os.getenv("API_KEY", "").strip()


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


CORS_ORIGINS = _parse_cors_origins()


# 1. Load Two-Stage ONNX models once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    print(f"WoundVision AI V6-TwoStage ONNX models ready on {DEVICE}!", flush=True)
    yield
    app.state.model = None


# 2. Initialize FastAPI
app = FastAPI(
    title="WoundVision AI",
    description="Wound Tissue Segmentation API (Two-Stage U-Net ResNet18 + ResNet34 / ONNX Runtime)",
    version="6.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=(CORS_ORIGINS != ["*"]),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _verify_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# 3. Root endpoint
@app.get("/")
def root():
    return {
        "name": "WoundVision AI",
        "version": "V6-TwoStage",
        "architecture": "Two-Stage U-Net (ResNet18 + ResNet34)",
        "runtime": "onnxruntime",
        "status": "running",
        "experimental": True,
    }


# 4. Health check endpoint
@app.get("/health")
def health():
    loaded = getattr(app.state, "model", None) is not None
    payload = {
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded,
        "model_version": "V6-TwoStage",
        "runtime": "onnxruntime",
        "device": str(DEVICE),
    }
    if not loaded:
        return JSONResponse(status_code=503, content=payload)
    return payload


# 5. Prediction endpoint
@app.post("/predict")
def predict_wound(
    file: UploadFile = File(...),
    opacity: float = Query(default=0.5, ge=0.0, le=1.0, description="Overlay opacity (0.0 - 1.0)"),
    include_overlay: bool = Query(default=True, description="Include base64 PNG overlay in response"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _verify_api_key(x_api_key)

    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Read uploaded file up to limit + 1 byte
    contents = file.file.read(MAX_FILE_SIZE_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Maximum file size is 10 MB")

    # Decode image and normalize EXIF orientation
    try:
        with Image.open(BytesIO(contents)) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image resolution is too large")
            image = ImageOps.exif_transpose(img).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid or unsupported image")

    # Run Two-Stage ONNX segmentation
    t0 = time.perf_counter()
    mask = predict(app.state.model, image)
    inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Calculate tissue percentages
    analysis = calculate_tissue_percentage(mask)

    # Optionally render PNG overlay
    overlay_data_url = None
    if include_overlay:
        overlay = generate_overlay(image, mask, alpha=opacity)
        buffer = BytesIO()
        overlay.save(buffer, format="PNG", optimize=True)
        overlay_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        overlay_data_url = f"data:image/png;base64,{overlay_base64}"

    return {
        "success": True,
        "model_version": "V6-TwoStage",
        "inference_ms": inference_ms,
        "analysis": analysis,
        "overlay": overlay_data_url,
        "disclaimer": (
            "Experimental AI prediction. "
            "Not validated for clinical decision-making."
        ),
    }
