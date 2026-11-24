from pathlib import Path
import torch
import segmentation_models_pytorch as smp
import sys
import numpy as np
from PIL import Image


# 1. Project configuration
PROJECT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_DIR / "src"))

from woundvision.evaluate import MEAN, STD

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "runs"
    / "notebook_v5_augmented"
    / "best_model.pth"
)

IMAGE_SIZE = 320
NUM_CLASSES = 6


# 2. Device configuration
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# 3. Load model
def load_model():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    return model

# Image preprocessing
def preprocess_image(image: Image.Image):

    # 1. Convert to RGB
    image = image.convert("RGB")

    # 2. Resize to model input size
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    # 3. Convert to NumPy array
    image_np = np.array(image)

    # 4. Convert to tensor and scale to 0-1
    tensor = torch.from_numpy(image_np)
    tensor = tensor.permute(2, 0, 1).float() / 255.0

    # 5. Add batch dimension
    tensor = tensor.unsqueeze(0).to(DEVICE)

    # 6. Normalization
    mean = torch.tensor(
        MEAN, device=DEVICE
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        STD, device=DEVICE
    ).view(1, 3, 1, 1)

    tensor = (tensor - mean) / std

    return tensor

# Segmentation prediction
def predict(model, image: Image.Image):

    # 1. Preprocess image
    tensor = preprocess_image(image)

    # 2. Run inference
    model.eval()

    with torch.inference_mode():
        logits = model(tensor)

    # 3. Convert logits to segmentation mask
    prediction = logits.argmax(dim=1)

    # 4. Remove batch dimension
    prediction = prediction.squeeze(0)

    # 5. Move to CPU and convert to NumPy
    mask = prediction.cpu().numpy().astype(np.uint8)

    return mask

# Tissue color mapping
TISSUE_COLORS = {
    1: (236, 72, 153),    # Epithelial
    2: (234, 179, 8),     # Slough
    3: (220, 38, 38),     # Granulation
    4: (38, 38, 38),      # Necrotic
    5: (168, 85, 247)     # Other
}


def generate_overlay(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.5
):

    # 1. Convert original image to RGB
    image = image.convert("RGB")

    # 2. Resize mask to original image size
    mask_image = Image.fromarray(mask)

    mask_image = mask_image.resize(
        image.size,
        Image.Resampling.NEAREST
    )

    mask_np = np.array(mask_image)

    # 3. Convert original image to NumPy
    original = np.array(image).astype(np.float32)

    # 4. Create overlay
    overlay = original.copy()

    # 5. Apply tissue colors
    for class_id, color in TISSUE_COLORS.items():

        tissue_pixels = mask_np == class_id

        overlay[tissue_pixels] = (
            (1 - alpha) * original[tissue_pixels]
            + alpha * np.array(color)
        )

    # 6. Convert back to PIL Image
    overlay = np.clip(
        overlay, 0, 255
    ).astype(np.uint8)

    return Image.fromarray(overlay)

# Tissue class mapping
CLASS_NAMES = {
    0: "background",
    1: "epithelial",
    2: "slough",
    3: "granulation",
    4: "necrotic",
    5: "other"
}


def calculate_tissue_percentage(mask: np.ndarray):

    # 1. Hitung total piksel
    total_pixels = mask.size

    # 2. Hitung piksel non-background
    wound_pixels = int(
        np.isin(mask, [1, 2, 3, 4, 5]).sum()
    )

    # 3. Siapkan dictionary hasil
    results = {}

    # 4. Hitung persentase setiap jaringan
    for class_id, class_name in CLASS_NAMES.items():

        if class_id == 0:
            continue

        pixels = int((mask == class_id).sum())

        percentage = (
            pixels / wound_pixels * 100
            if wound_pixels > 0
            else None
        )

        results[class_name] = {
            "pixels": pixels,
            "percentage": (
                round(percentage, 2)
                if percentage is not None
                else None
            )
        }

    # 5. Return hasil analisis
    return {
        "total_pixels": total_pixels,
        "wound_pixels": wound_pixels,
        "wound_detected": wound_pixels > 0,
        "wound_coverage": round(
            wound_pixels / total_pixels * 100, 2
        ),
        "tissues": results
    }

# 4. Test model loading
if __name__ == "__main__":

    model = load_model()

    print("=== WOUNDVISION AI ===")
    print("Model: U-Net ResNet34")
    print("Version: V5-Augmented")
    print("Device:", DEVICE)
    print("Input size:", IMAGE_SIZE)
    print("Number of classes:", NUM_CLASSES)
    print("Model loaded successfully!")