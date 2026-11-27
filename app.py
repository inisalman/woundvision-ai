"""Production FastAPI server for WoundVision AI V6-TwoStage (ONNX Runtime)."""

from contextlib import asynccontextmanager
from io import BytesIO
import base64
import hmac
import os
import time
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from inference import (
    load_model,
    predict_with_telemetry,
    generate_overlay,
    calculate_tissue_percentage,
    DEVICE,
)
from monitoring import (
    MetricsStore,
    PredictionTelemetryEvent,
    emit_json_log,
    get_memory_mb,
)

MODEL_VERSION = "V6-TwoStage"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_PIXELS = 20_000_000           # 20 MP


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


CORS_ORIGINS = _parse_cors_origins()


def _get_api_key() -> str:
    return os.getenv("API_KEY", "").strip()


def _get_metrics_api_key() -> str:
    metrics_key = os.getenv("METRICS_API_KEY", "").strip()
    return metrics_key or _get_api_key()


def _get_metrics_store(fastapi_app: FastAPI) -> MetricsStore | None:
    return getattr(fastapi_app.state, "metrics", None)


# 1. Load Two-Stage ONNX models and initialize telemetry store once at startup
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    fastapi_app.state.started_at = time.time()
    fastapi_app.state.metrics = MetricsStore()
    fastapi_app.state.model = load_model()
    mem = get_memory_mb()
    emit_json_log(
        "service_started",
        {
            "model_version": MODEL_VERSION,
            "device": str(DEVICE),
            "metrics_db_ready": fastapi_app.state.metrics.db_ready,
            "metrics_db_path": str(fastapi_app.state.metrics.db_path),
            "current_rss_mb": mem["current_rss_mb"],
            "peak_rss_mb": mem["peak_rss_mb"],
        },
        enabled=fastapi_app.state.metrics.json_logs,
    )
    yield
    fastapi_app.state.model = None
    if getattr(fastapi_app.state, "metrics", None) is not None:
        fastapi_app.state.metrics.close()
        fastapi_app.state.metrics = None


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
    expected = _get_api_key()
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _verify_metrics_api_key(x_api_key: str | None) -> None:
    expected = _get_metrics_api_key()
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _record_prediction_event(event: PredictionTelemetryEvent) -> None:
    store = _get_metrics_store(app)
    if store is not None:
        store.record(event)


# 3. Root endpoint
@app.get("/")
def root():
    return {
        "name": "WoundVision AI",
        "version": MODEL_VERSION,
        "architecture": "Two-Stage U-Net (ResNet18 + ResNet34)",
        "runtime": "onnxruntime",
        "status": "running",
        "experimental": True,
    }


# 4. Health check endpoint
@app.get("/health")
def health():
    loaded = getattr(app.state, "model", None) is not None
    started_at = getattr(app.state, "started_at", None)
    uptime_seconds = round(time.time() - started_at, 1) if started_at else 0.0
    store = _get_metrics_store(app)
    mem = get_memory_mb()
    payload = {
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded,
        "model_version": MODEL_VERSION,
        "runtime": "onnxruntime",
        "device": str(DEVICE),
        "uptime_seconds": uptime_seconds,
        "memory": mem,
        "metrics_db_ready": store.db_ready if store is not None else False,
    }
    if not loaded:
        return JSONResponse(status_code=503, content=payload)
    return payload


# 5. Prediction endpoint
@app.post("/predict")
def predict_wound(
    response: Response,
    file: UploadFile = File(...),
    opacity: float = Query(default=0.5, ge=0.0, le=1.0, description="Overlay opacity (0.0 - 1.0)"),
    include_overlay: bool = Query(default=True, description="Include base64 PNG overlay in response"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    request_id = uuid.uuid4().hex[:12]
    response.headers["X-Request-ID"] = request_id
    t_req_start = time.perf_counter()

    file_size_bytes: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    decode_ms: float | None = None

    def _record_error(status_code: int, error_type: str) -> None:
        total_ms = round((time.perf_counter() - t_req_start) * 1000.0, 2)
        _record_prediction_event(
            PredictionTelemetryEvent(
                request_id=request_id,
                model_version=MODEL_VERSION,
                status_code=status_code,
                error_type=error_type,
                file_size_bytes=file_size_bytes,
                image_width=image_width,
                image_height=image_height,
                include_overlay=include_overlay,
                decode_ms=decode_ms,
                total_ms=total_ms,
            )
        )

    try:
        _verify_api_key(x_api_key)
    except HTTPException:
        _record_error(401, "invalid_api_key")
        raise

    if getattr(app.state, "model", None) is None:
        _record_error(503, "model_not_loaded")
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Read uploaded file up to limit + 1 byte
    t_decode_start = time.perf_counter()
    contents = file.file.read(MAX_FILE_SIZE_BYTES + 1)
    file_size_bytes = len(contents)
    if not contents:
        _record_error(400, "empty_file")
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        _record_error(413, "file_too_large")
        raise HTTPException(status_code=413, detail="Maximum file size is 10 MB")

    # Decode image and normalize EXIF orientation
    try:
        with Image.open(BytesIO(contents)) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS:
                _record_error(413, "resolution_too_large")
                raise HTTPException(status_code=413, detail="Image resolution is too large")
            image = ImageOps.exif_transpose(img).convert("RGB")
            image_width, image_height = image.size
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        _record_error(400, "invalid_image")
        raise HTTPException(status_code=400, detail="Invalid or unsupported image")

    decode_ms = round((time.perf_counter() - t_decode_start) * 1000.0, 2)

    try:
        # Run Two-Stage ONNX segmentation with stage-level timings & confidence
        pred = predict_with_telemetry(app.state.model, image)
        mask = pred.mask

        # Calculate tissue percentages
        analysis = calculate_tissue_percentage(mask)

        # Optionally render PNG overlay
        overlay_data_url = None
        overlay_ms = 0.0
        if include_overlay:
            t_overlay_start = time.perf_counter()
            overlay = generate_overlay(image, mask, alpha=opacity)
            buffer = BytesIO()
            overlay.save(buffer, format="PNG", optimize=True)
            overlay_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            overlay_data_url = f"data:image/png;base64,{overlay_base64}"
            overlay_ms = round((time.perf_counter() - t_overlay_start) * 1000.0, 2)

        total_ms = round((time.perf_counter() - t_req_start) * 1000.0, 2)
    except Exception:
        _record_error(500, "internal_error")
        raise

    tissues = analysis["tissues"]
    _record_prediction_event(
        PredictionTelemetryEvent(
            request_id=request_id,
            model_version=MODEL_VERSION,
            status_code=200,
            file_size_bytes=file_size_bytes,
            image_width=image_width,
            image_height=image_height,
            include_overlay=include_overlay,
            decode_ms=decode_ms,
            stage1_ms=pred.stage1_ms,
            stage2_ms=pred.stage2_ms,
            inference_ms=pred.inference_ms,
            overlay_ms=overlay_ms,
            total_ms=total_ms,
            wound_detected=analysis["wound_detected"],
            wound_coverage=analysis["wound_coverage"],
            stage1_max_prob=pred.stage1_max_prob,
            stage1_mean_prob=pred.stage1_mean_prob,
            stage2_confidence=pred.stage2_confidence,
            stage2_entropy=pred.stage2_entropy,
            epithelial_pct=tissues["epithelial"]["percentage"],
            slough_pct=tissues["slough"]["percentage"],
            granulation_pct=tissues["granulation"]["percentage"],
            necrotic_pct=tissues["necrotic"]["percentage"],
            other_pct=tissues["other"]["percentage"],
        )
    )

    return {
        "success": True,
        "request_id": request_id,
        "model_version": MODEL_VERSION,
        "inference_ms": pred.inference_ms,
        "timings_ms": {
            "decode": decode_ms,
            "stage1": pred.stage1_ms,
            "stage2": pred.stage2_ms,
            "inference": pred.inference_ms,
            "overlay": overlay_ms,
            "total": total_ms,
        },
        "confidence": {
            "stage1_max_prob": pred.stage1_max_prob,
            "stage1_mean_prob": pred.stage1_mean_prob,
            "stage2_confidence": pred.stage2_confidence,
            "stage2_entropy": pred.stage2_entropy,
        },
        "analysis": analysis,
        "overlay": overlay_data_url,
        "disclaimer": (
            "Experimental AI prediction. "
            "Not validated for clinical decision-making."
        ),
    }


# 6. Performance Monitoring & Telemetry endpoints
@app.get("/metrics/summary")
def metrics_summary(
    hours: float = Query(
        default=24.0,
        ge=0.0,
        le=8760.0,
        description="Rolling window in hours (0 = all recorded history)",
    ),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _verify_metrics_api_key(x_api_key)
    store = _get_metrics_store(app)
    if store is None:
        raise HTTPException(status_code=503, detail="Metrics store is not initialized")
    return store.get_summary(window_hours=hours)


@app.get("/metrics/recent")
def metrics_recent(
    limit: int = Query(default=20, ge=1, le=200, description="Max number of recent events"),
    only_errors: bool = Query(default=False, description="Return only error responses (HTTP >= 400)"),
    low_confidence_only: bool = Query(
        default=False,
        description="Return only wound detections flagged as low-confidence",
    ),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _verify_metrics_api_key(x_api_key)
    store = _get_metrics_store(app)
    if store is None:
        raise HTTPException(status_code=503, detail="Metrics store is not initialized")
    events = store.get_recent(
        limit=limit,
        only_errors=only_errors,
        low_confidence_only=low_confidence_only,
    )
    return {
        "count": len(events),
        "events": events,
    }
