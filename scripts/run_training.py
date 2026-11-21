import argparse
import sys
from pathlib import Path
import random
import numpy as np
import json
import uuid
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woundvision.config import (
    DATA_ROOT,
    CHECKPOINT_V02,
    CLASS_WEIGHTS
)
from woundvision.data import WoundDataset
from woundvision.model import build_model
from woundvision.train import train_one_epoch
from woundvision.evaluate import evaluate_model, calculate_metrics


# 1. Konfigurasi dari terminal
parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    default=2026
)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--max-train-samples", type=int, default=4)

parser.add_argument(
    "--necrotic-weight",
    type=float,
    default=3.0
)

args = parser.parse_args()

import math

assert math.isfinite(args.necrotic_weight)
assert args.necrotic_weight > 0

class_weights = CLASS_WEIGHTS.copy()
class_weights[4] = args.necrotic_weight

# Reproducibility
SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Batasi jumlah thread CPU
torch.set_num_threads(4)

print("Random seed:", SEED)

assert args.epochs > 0
assert args.batch_size > 0
assert args.max_train_samples >= 0


# 2. Device
device = torch.device("cpu")

# 3. Dataset
train_dataset = WoundDataset(DATA_ROOT, "train.txt")
val_dataset = WoundDataset(DATA_ROOT, "val.txt")

# Batasi data untuk pengujian pipeline
if args.max_train_samples > 0:
    n = min(args.max_train_samples, len(train_dataset))
    train_dataset = Subset(train_dataset, range(n))

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    generator=torch.Generator().manual_seed(SEED)

)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0
)


# 4. Load model V0.2
model = build_model()

checkpoint = torch.load(
    CHECKPOINT_V02,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state"])
model.to(device)


# 5. Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


# 6. Training dan evaluasi
print("WOUNDVISION TRAINING RUNNER")
print("=" * 50)
print("Device:", device)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Epochs:", args.epochs)


# 7. Experiment Tracking
run_id = (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + uuid.uuid4().hex[:6]
)

run_dir = ROOT / "runs" / run_id
run_dir.mkdir(parents=True, exist_ok=False)

# Simpan konfigurasi eksperimen
experiment_config = {
    "run_id": run_id,
    "seed": args.seed,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "train_images": len(train_dataset),
    "validation_images": len(val_dataset),
    "starting_checkpoint": CHECKPOINT_V02.name,
    "class_weights": class_weights,
    "checkpoint_metric": "clinical_3_dice",
    "device": str(device)
}

config_path = run_dir / "config.json"

config_path.write_text(
    json.dumps(experiment_config, indent=2),
    encoding="utf-8"
)

print("Experiment ID:", run_id)
print("Necrotic Class Weight:", args.necrotic_weight)



# Evaluasi checkpoint awal sebelum training
# Evaluasi checkpoint awal sebelum training
baseline_cm = evaluate_model(
    model, val_loader, device
)

baseline_metrics = calculate_metrics(baseline_cm)

# Clinical Dice: Slough, Granulation, Necrotic
best_clinical = sum(
    baseline_metrics["dice_per_class"][i]
    for i in (2, 3, 4)
) / 3

best_epoch = 0

best_checkpoint_path = run_dir / "best_model.pth"

print(
    f"Starting Clinical 3 Dice: "
    f"{best_clinical * 100:.2f}%"
)


for epoch in range(1, args.epochs + 1):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device,
        class_weights=class_weights
    )

    cm = evaluate_model(
        model,
        val_loader,
        device
    )

    metrics = calculate_metrics(cm)

    # Rata-rata Dice tiga jaringan klinis utama
    # 2 = slough, 3 = Granulation, 4 = Necrotic
    clinical_3_dice = sum(
        metrics["dice_per_class"][i]
        for i in (2, 3, 4)
    ) / 3

    print(f"\nEPOCH {epoch}")
    print("-" * 40)
    print(f"Train Loss: {train_loss:.4f}")
    print(
        "Macro Tissue Dice:",
        f"{metrics['macro_tissue_dice'] * 100:.2f}%"
    )
    print(
        "Wound Dice:",
        f"{metrics['wound_dice'] * 100:.2f}%"
    )

    print(
        "Clinical 3-Class Dice:",
        f"{clinical_3_dice * 100:.2f}%"
    )


    # Monitoring Dice setiap kelas jaringan
    class_names = [
        "Background",
        "Epithelial",
        "Slough",
        "Granulation",
        "Necrotic",
        "Other"
    ]

    print("\nPer-class Dice:")

    print("-" * 40)
    
    for name, dice in zip(
        class_names,
         metrics["dice_per_class"]
    ):
        print(f"{name:13}: {dice * 100:.2f}%")

    # Simpan metrik setiap epoch
    epoch_result = {
        "epoch": epoch,
        "train_loss": train_loss,
        "macro_tissue_dice": metrics["macro_tissue_dice"],
        "wound_dice": metrics["wound_dice"],
        "clinical_3_dice": clinical_3_dice,
        "pixel_accuracy": metrics["pixel_accuracy"],
        "dice_per_class": {
            "Background": metrics["dice_per_class"][0],
            "Epithelial": metrics["dice_per_class"][1],
            "Slough": metrics["dice_per_class"][2],
            "Granulation": metrics["dice_per_class"][3],
            "Necrotic": metrics["dice_per_class"][4],
            "Other": metrics["dice_per_class"][5]
        }
    }

    metrics_path = run_dir / "metrics.jsonl"


    # Simpan jika Macro Tissue Dice meningkat
    current_macro = metrics["macro_tissue_dice"]
    current_clinical = clinical_3_dice

    if current_clinical > best_clinical:

        best_clinical = current_clinical
        best_epoch = epoch

        checkpoint_data = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
            "macro_tissue_dice": current_macro,
            "wound_dice": metrics["wound_dice"],
            "dice_per_class": metrics["dice_per_class"],
            "train_images": len(train_dataset),
            "clinical_3_dice": current_clinical,
            "starting_checkpoint": CHECKPOINT_V02.name
        }

        # Simpan dahulu ke file sementara
        temp_path = run_dir / "best_model.tmp"

        torch.save(checkpoint_data, temp_path)

        # Ganti file final setelah penulisan selesai
        temp_path.replace(best_checkpoint_path)

        print(
            f"Best checkpoint saved! "
            f"Epoch {best_epoch} | "
            f"Clinical 3 Dice: {best_clinical * 100:.2f}%"
        )


    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(epoch_result) + "\n")

    

print("\nTRAINING RUNNER: PASS")


print("\nBEST MODEL SUMMARY")
print("-" * 45)

print("Best epoch:", best_epoch)
print(
    f"Best Clinical 3 Dice: "
    f"{best_clinical * 100:.2f}%"
)

if best_epoch > 0:
    print("Checkpoint:", best_checkpoint_path)
else:
    print("No improvement over starting checkpoint")

