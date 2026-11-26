"""Lightweight ONNX Runtime inference engine for WoundVision AI (V5-Augmented)."""

from pathlib import Path
import os
import urllib.request
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort


# 1. Project & runtime configuration
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(PROJECT_DIR / "models" / "woundvision_v5_augmented.onnx"),
    )
)
MODEL_URL = os.getenv("MODEL_URL", "").strip()
MODEL_AUTH_TOKEN = os.getenv("MODEL_AUTH_TOKEN", "").strip()

ORT_INTRA_THREADS = int(os.getenv("ORT_INTRA_THREADS", "2"))
ORT_INTER_THREADS = int(os.getenv("ORT_INTER_THREADS", "1"))

IMAGE_SIZE = 320
NUM_CLASSES = 6

# ImageNet normalization constants matching training/evaluation
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

DEVICE = "cpu (onnxruntime)"


def _ensure_model_file(model_path: Path) -> Path:
    """Ensure ONNX model exists locally, downloading from MODEL_URL if configured."""
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    if not MODEL_URL:
        raise FileNotFoundError(
            f"ONNX model not found at {model_path}. "
            "Either mount/copy models/woundvision_v5_augmented.onnx, run "
            "'python scripts/export_onnx.py', or set the MODEL_URL environment variable."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".onnx.tmp")

    headers = {"User-Agent": "WoundVision-AI/5.0"}
    if MODEL_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {MODEL_AUTH_TOKEN}"

    print(f"Downloading ONNX model from {MODEL_URL} ...")
    req = urllib.request.Request(MODEL_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp_path, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    tmp_path.replace(model_path)
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Model downloaded to {model_path} ({size_mb:.2f} MB)")
    return model_path


# 2. Load ONNX Runtime session
def load_model(model_path: Path | None = None) -> ort.InferenceSession:
    path = _ensure_model_file(model_path or MODEL_PATH)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = ORT_INTRA_THREADS
    sess_options.inter_op_num_threads = ORT_INTER_THREADS
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True

    session = ort.InferenceSession(
        str(path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    return session


# 3. Image preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # (1, 3, H, W)
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr, dtype=np.float32)


# 4. Segmentation prediction
def predict(model: ort.InferenceSession, image: Image.Image) -> np.ndarray:
    input_tensor = preprocess_image(image)
    input_name = model.get_inputs()[0].name
    logits = model.run(None, {input_name: input_tensor})[0]
    mask = logits.argmax(axis=1).squeeze(0).astype(np.uint8)
    return mask


# Tissue color mapping
TISSUE_COLORS = {
    1: (236, 72, 153),    # Epithelial
    2: (234, 179, 8),     # Slough
    3: (220, 38, 38),     # Granulation
    4: (38, 38, 38),      # Necrotic
    5: (168, 85, 247),    # Other
}


def generate_overlay(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")

    mask_image = Image.fromarray(mask).resize(
        image.size,
        Image.Resampling.NEAREST,
    )
    mask_np = np.asarray(mask_image)

    original = np.asarray(image, dtype=np.float32)
    overlay = original.copy()

    for class_id, color in TISSUE_COLORS.items():
        tissue_pixels = mask_np == class_id
        if np.any(tissue_pixels):
            overlay[tissue_pixels] = (
                (1.0 - alpha) * original[tissue_pixels]
                + alpha * np.array(color, dtype=np.float32)
            )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


# Tissue class mapping
CLASS_NAMES = {
    0: "background",
    1: "epithelial",
    2: "slough",
    3: "granulation",
    4: "necrotic",
    5: "other",
}


def calculate_tissue_percentage(mask: np.ndarray) -> dict:
    total_pixels = int(mask.size)
    wound_pixels = int(np.isin(mask, [1, 2, 3, 4, 5]).sum())

    results = {}
    for class_id, class_name in CLASS_NAMES.items():
        if class_id == 0:
            continue

        pixels = int((mask == class_id).sum())
        percentage = (
            pixels / wound_pixels * 100.0
            if wound_pixels > 0
            else None
        )

        results[class_name] = {
            "pixels": pixels,
            "percentage": round(percentage, 2) if percentage is not None else None,
        }

    return {
        "total_pixels": total_pixels,
        "wound_pixels": wound_pixels,
        "wound_detected": wound_pixels > 0,
        "wound_coverage": round(wound_pixels / total_pixels * 100.0, 2),
        "tissues": results,
    }


if __name__ == "__main__":
    session = load_model()
    print("=== WOUNDVISION AI (ONNX RUNTIME) ===")
    print("Model: U-Net ResNet34")
    print("Version: V5-Augmented")
    print("Path:", MODEL_PATH)
    print("Providers:", session.get_providers())
    print("Threads (intra/inter):", ORT_INTRA_THREADS, "/", ORT_INTER_THREADS)
    print("Input size:", IMAGE_SIZE)
    print("Number of classes:", NUM_CLASSES)
    print("Model loaded successfully!")
