"""Lightweight Two-Stage ONNX Runtime inference engine for WoundVision AI (V6-TwoStage)."""

from dataclasses import dataclass
from pathlib import Path
import os
import urllib.request
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort


# 1. Project & runtime configuration
PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_STAGE1_URL = (
    "https://huggingface.co/salmanid/woundvision-ai/resolve/main/"
    "woundvision_v6_stage1_wound.onnx"
)
DEFAULT_STAGE2_URL = (
    "https://huggingface.co/salmanid/woundvision-ai/resolve/main/"
    "woundvision_v6_stage2_tissue.onnx"
)

STAGE1_MODEL_PATH = Path(
    os.getenv(
        "STAGE1_MODEL_PATH",
        str(PROJECT_DIR / "models" / "woundvision_v6_stage1_wound.onnx"),
    )
)
STAGE2_MODEL_PATH = Path(
    os.getenv(
        "STAGE2_MODEL_PATH",
        str(PROJECT_DIR / "models" / "woundvision_v6_stage2_tissue.onnx"),
    )
)

STAGE1_MODEL_URL = os.getenv("STAGE1_MODEL_URL", DEFAULT_STAGE1_URL).strip()
STAGE2_MODEL_URL = os.getenv("STAGE2_MODEL_URL", DEFAULT_STAGE2_URL).strip()
MODEL_AUTH_TOKEN = os.getenv("MODEL_AUTH_TOKEN", "").strip()

ORT_INTRA_THREADS = int(os.getenv("ORT_INTRA_THREADS", "2"))
ORT_INTER_THREADS = int(os.getenv("ORT_INTER_THREADS", "1"))

STAGE1_SIZE = 320
STAGE2_SIZE = 320
STAGE1_THRESHOLD = float(os.getenv("STAGE1_THRESHOLD", "0.3"))
ROI_MARGIN = float(os.getenv("ROI_MARGIN", "0.18"))
MIN_WOUND_PIXELS = 16
NUM_CLASSES = 6

# ImageNet normalization constants matching training/evaluation
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

DEVICE = "cpu (onnxruntime)"


@dataclass
class TwoStageModel:
    stage1: ort.InferenceSession
    stage2: ort.InferenceSession


def _ensure_model_file(model_path: Path, model_url: str) -> Path:
    """Ensure ONNX model exists locally, downloading from URL if missing."""
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    if not model_url:
        raise FileNotFoundError(
            f"ONNX model not found at {model_path} and no download URL configured."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".onnx.tmp")

    headers = {"User-Agent": "WoundVision-AI/6.0"}
    if MODEL_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {MODEL_AUTH_TOKEN}"

    print(f"Downloading {model_path.name} from {model_url} ...", flush=True)
    req = urllib.request.Request(model_url, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp_path, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    tmp_path.replace(model_path)
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Downloaded {model_path.name} ({size_mb:.2f} MB)", flush=True)
    return model_path


def _create_session(path: Path) -> ort.InferenceSession:
    """Create a low-RAM ONNX Runtime CPU session tuned for shared VPS containers."""
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = ORT_INTRA_THREADS
    sess_options.inter_op_num_threads = ORT_INTER_THREADS
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Disable large pre-allocated CPU memory arenas so two-stage peak RSS stays ~515 MB
    sess_options.enable_mem_pattern = False
    sess_options.enable_cpu_mem_arena = False
    return ort.InferenceSession(
        str(path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


# 2. Load Stage 1 + Stage 2 ONNX sessions
def load_model() -> TwoStageModel:
    s1_path = _ensure_model_file(STAGE1_MODEL_PATH, STAGE1_MODEL_URL)
    s2_path = _ensure_model_file(STAGE2_MODEL_PATH, STAGE2_MODEL_URL)
    return TwoStageModel(
        stage1=_create_session(s1_path),
        stage2=_create_session(s2_path),
    )


# 3. Image preprocessing
def preprocess_image(image: Image.Image, size: int = STAGE1_SIZE) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # (1, 3, H, W)
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _compute_square_roi(binary_mask: np.ndarray, margin: float = ROI_MARGIN):
    ys, xs = np.where(binary_mask)
    if len(xs) < MIN_WOUND_PIXELS:
        return None
    h, w = binary_mask.shape
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    bw = x_max - x_min + 1
    bh = y_max - y_min + 1
    side = min(max(int(max(bw, bh) * (1.0 + 2.0 * margin)), 32), max(w, h))

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    x1 = max(0, min(int(round(cx - side / 2.0)), max(0, w - side)))
    y1 = max(0, min(int(round(cy - side / 2.0)), max(0, h - side)))
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    return x1, y1, x2, y2


# 4. Two-Stage Segmentation prediction
def predict(model: TwoStageModel, image: Image.Image) -> np.ndarray:
    orig = ImageOps.exif_transpose(image).convert("RGB")

    # Stage 1: Binary wound localization with sequential H-Flip TTA (low-RAM b=1)
    x1 = preprocess_image(orig, STAGE1_SIZE)
    s1_in_name = model.stage1.get_inputs()[0].name
    l1_orig = model.stage1.run(None, {s1_in_name: x1})[0]
    l1_flip = model.stage1.run(None, {s1_in_name: np.ascontiguousarray(x1[:, :, :, ::-1])})[0]
    p1 = 0.5 * (_sigmoid(l1_orig[0, 0]) + _sigmoid(l1_flip[0, 0])[:, ::-1])

    s1_bin = (
        np.asarray(
            Image.fromarray((p1 >= STAGE1_THRESHOLD).astype(np.uint8) * 255).resize(
                orig.size, Image.Resampling.NEAREST
            )
        )
        > 127
    )

    full_mask = np.zeros((orig.height, orig.width), dtype=np.uint8)
    box = _compute_square_roi(s1_bin, margin=ROI_MARGIN)
    if box is None:
        return full_mask

    # Crop square ROI (preserve aspect ratio with zero padding)
    x1b, y1b, x2b, y2b = box
    crop = orig.crop((x1b, y1b, x2b, y2b))
    cw, ch = crop.size
    sq_side = max(cw, ch)
    pad_l = (sq_side - cw) // 2
    pad_t = (sq_side - ch) // 2

    sq_img = Image.new("RGB", (sq_side, sq_side), (0, 0, 0))
    sq_img.paste(crop, (pad_l, pad_t))

    # Stage 2: High-resolution tissue classification inside Stage 1 wound mask (Mode C)
    x2 = preprocess_image(sq_img, STAGE2_SIZE)
    s2_in_name = model.stage2.get_inputs()[0].name
    l2 = model.stage2.run(None, {s2_in_name: x2})[0]
    tissue_pred = (l2[0, 1:NUM_CLASSES].argmax(axis=0) + 1).astype(np.uint8)

    sq_pred = np.asarray(
        Image.fromarray(tissue_pred).resize((sq_side, sq_side), Image.Resampling.NEAREST)
    )
    full_mask[y1b:y2b, x1b:x2b] = sq_pred[pad_t : pad_t + ch, pad_l : pad_l + cw]
    full_mask[~s1_bin] = 0
    return full_mask


# Tissue color mapping
TISSUE_COLORS = {
    1: (236, 72, 153),    # Epithelial (#EC4899)
    2: (234, 179, 8),     # Slough (#EAB308)
    3: (220, 38, 38),     # Granulation (#DC2626)
    4: (38, 38, 38),      # Necrotic (#262626)
    5: (168, 85, 247),    # Other (#A855F7)
}


def generate_overlay(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")

    if mask.shape[1] != image.width or mask.shape[0] != image.height:
        mask_np = np.asarray(
            Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)
        )
    else:
        mask_np = mask

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
    models = load_model()
    print("=== WOUNDVISION AI V6-TWOSTAGE (ONNX RUNTIME) ===")
    print("Stage 1:", STAGE1_MODEL_PATH)
    print("Stage 2:", STAGE2_MODEL_PATH)
    print("Threads (intra/inter):", ORT_INTRA_THREADS, "/", ORT_INTER_THREADS)
    print("Models loaded successfully!")
