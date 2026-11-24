from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import base64

from inference import (
    load_model,
    predict,
    generate_overlay,
    calculate_tissue_percentage,
    DEVICE
)


# 1. Load model saat aplikasi dimulai
@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.model = load_model()

    print("WoundVision AI model ready!")

    yield

    app.state.model = None


# 2. Inisialisasi FastAPI
app = FastAPI(
    title="WoundVision AI",
    description="Experimental Wound Tissue Segmentation API",
    version="5.0.0",
    lifespan=lifespan
)


# 3. Root endpoint
@app.get("/")
def root():

    return {
        "name": "WoundVision AI",
        "version": "V5-Augmented",
        "architecture": "U-Net ResNet34",
        "status": "running",
        "experimental": True
    }


# 4. Health check endpoint
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": app.state.model is not None,
        "device": str(DEVICE)
    }


# 5. Prediction endpoint
@app.post("/predict")
def predict_wound(file: UploadFile = File(...)):

    # Read uploaded file
    contents = file.file.read(10 * 1024 * 1024 + 1)

    # Maximum file size: 10 MB
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 10 MB"
        )

    # Decode image
    try:
        with Image.open(BytesIO(contents)) as img:

            if img.width * img.height > 20_000_000:
                raise HTTPException(
                    status_code=413,
                    detail="Image resolution is too large"
                )

            image = img.convert("RGB")

    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image"
        )

    # Run segmentation
    mask = predict(app.state.model, image)

    # Generate overlay
    overlay = generate_overlay(image, mask)

    # Calculate tissue percentages
    analysis = calculate_tissue_percentage(mask)

    # Convert overlay to Base64
    buffer = BytesIO()
    overlay.save(buffer, format="PNG")

    overlay_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    # Return JSON response
    return {
        "success": True,
        "model_version": "V5-Augmented",
        "analysis": analysis,
        "overlay": f"data:image/png;base64,{overlay_base64}",
        "disclaimer": (
            "Experimental AI prediction. "
            "Not validated for clinical decision-making."
        )
    }