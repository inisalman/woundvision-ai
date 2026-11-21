
from pathlib import Path
import sys
import hashlib

import torch
from torch.utils.data import DataLoader
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woundvision.data import WoundDataset
from woundvision.model import build_model
from woundvision.evaluate import (
    evaluate_model,
    calculate_metrics
)


# 1. Load validation dataset
dataset = WoundDataset(
    ROOT / "data" / "LUTSeg",
    "val.txt"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
)

# 2. Load model V0.1
device = torch.device("cpu")

model = build_model()

# Pilihan checkpoint melalui terminal
parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint",
    choices=["v01", "v02"],
    default="v02",
    help="Model checkpoint to evaluate"
)

args = parser.parse_args()

checkpoint_files = {
    "v01": "woundvision_mvp_3d6d1c55.pth",
    "v02": "woundvision_v02_78189535.pth"
}

checkpoint_path = (
    ROOT / "models" / checkpoint_files[args.checkpoint]
)

print("Evaluating checkpoint:", checkpoint_path.name)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state"])
model.to(device)

# 3. Evaluate
cm = evaluate_model(model, loader, device)
metrics = calculate_metrics(cm)

fingerprint = hashlib.sha256(
    cm.float().numpy().tobytes()
).hexdigest()[:12]

# 4. Print results
print("\nWOUNDVISION LOCAL EVALUATION")
print("=" * 50)

print("Validation images:", len(dataset))
print("Confusion fingerprint:", fingerprint)

print(
    f"Pixel Accuracy: "
    f"{metrics['pixel_accuracy'] * 100:.2f}%"
)

print(
    f"Wound Dice: "
    f"{metrics['wound_dice'] * 100:.2f}%"
)

print(
    f"Macro Tissue Dice: "
    f"{metrics['macro_tissue_dice'] * 100:.2f}%"
)

print("\nPer-class Dice:")
for name, value in zip(
    ["Background", "Epithelial", "Slough",
     "Granulation", "Necrotic", "Other"],
    metrics["dice_per_class"]
):
    print(f"  {name}: {value * 100:.2f}%")
