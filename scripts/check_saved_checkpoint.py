from pathlib import Path
import sys
import json
import torch
from collections import defaultdict
from torch.utils.data import Subset

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woundvision.model import build_model
from woundvision.data import WoundDataset
from woundvision.evaluate import evaluate_model, calculate_metrics
from woundvision.config import CHECKPOINT_V02


# Folder eksperimen yang baru selesai

# Temukan run terbaru yang memiliki checkpoint
runs_dir = ROOT / "runs"

available_runs = sorted(
    path for path in runs_dir.iterdir()
    if path.is_dir()
    and (path / "best_model.pth").exists()
    and (path / "metrics.jsonl").exists()
)

assert available_runs, "No saved checkpoints found"


# Pilih checkpoint dari training penuh 111 gambar
RUN_DIR = (
    ROOT
    / "runs"
    / "20261121_185427_e291f3"
)

assert RUN_DIR.exists(), "Run directory not found"

print("Verifying run:", RUN_DIR.name)



checkpoint_path = RUN_DIR / "best_model.pth"
metrics_path = RUN_DIR / "metrics.jsonl"

# 1. Load checkpoint hasil training
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

# 2. Bangun model dan muat bobot
model = build_model()
model.load_state_dict(checkpoint["model_state"])

device = torch.device("cpu")
model.to(device)
model.eval()

# 3. Load validation dataset
dataset = WoundDataset(
    ROOT / "data" / "LUTSeg",
    "val.txt"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

# 4. Evaluasi ulang checkpoint
cm = evaluate_model(model, loader, device)
# Hitung piksel aktual dan prediksi per kelas
class_names = [
    "Background",
    "Epithelial",
    "Slough",
    "Granulation",
    "Necrotic",
    "Other"
]

true_pixels = cm.sum(dim=1)
pred_pixels = cm.sum(dim=0)

print("\nVALIDATION CLASS DISTRIBUTION")
print("-" * 55)

for i, name in enumerate(class_names):
    print(
        f"{name:13} | "
        f"GT: {true_pixels[i].item():>10,} | "
        f"Pred: {pred_pixels[i].item():>10,}"
    )


# Analisis kesalahan khusus kelas Necrotic
NECROTIC = 4

necrotic_row = cm[NECROTIC]
total_necrotic = necrotic_row.sum().item()

print("\nNECROTIC ERROR ANALYSIS")
print("-" * 50)

for i, name in enumerate(class_names):
    count = necrotic_row[i].item()
    percentage = 100 * count / total_necrotic

    print(
        f"GT Necrotic -> {name:13}: "
        f"{count:>7,} pixels ({percentage:.2f}%)"
    )

# Hitung precision dan recall Necrotic
tp = cm[NECROTIC, NECROTIC].item()
predicted_total = cm[:, NECROTIC].sum().item()

precision = tp / predicted_total if predicted_total else 0
recall = tp / total_necrotic if total_necrotic else 0

print("\nNECROTIC PERFORMANCE")
print("-" * 50)
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")


metrics = calculate_metrics(cm)
clinical_3_dice = sum(
    metrics["dice_per_class"][i]
    for i in (2, 3, 4)
) / 3

# 5. Baca hasil yang tersimpan saat training
records = [
    json.loads(line)
    for line in metrics_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

saved_metrics = next(
    row for row in records
    if row["epoch"] == checkpoint["epoch"]
)

print("\nCLINICAL 3 DICE VERIFICATION")
print("-" * 50)

logged_clinical = saved_metrics["clinical_3_dice"]

print(
    f"Logged   : {logged_clinical * 100:.4f}%"
)
print(
    f"Reloaded : {clinical_3_dice * 100:.4f}%"
)

assert abs(
    logged_clinical - clinical_3_dice
) < 1e-6

# 6. Bandingkan
expected = saved_metrics["macro_tissue_dice"]
actual = metrics["macro_tissue_dice"]

print("CHECKPOINT INTEGRITY TEST")

# Evaluasi berdasarkan pasien
patient_indices = defaultdict(list)

for index, (image_path, mask_path) in enumerate(
    dataset.samples
):
    patient_id = image_path.parent.name
    patient_indices[patient_id].append(index)

print("\nPER-PATIENT EVALUATION")
print("=" * 65)


# Bandingkan V0.2 dengan V0.3 pada pasien yang sama
baseline_checkpoint = torch.load(
    CHECKPOINT_V02,
    map_location="cpu",
    weights_only=True
)

baseline_model = build_model()

baseline_model.load_state_dict(
    baseline_checkpoint["model_state"]
)

baseline_model.to(device)
baseline_model.eval()

print("\nV0.2 VS V0.3 — PER-PATIENT NECROTIC DICE")
print("=" * 65)

for patient_id, indices in sorted(
    patient_indices.items()
):
    patient_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=2,
        shuffle=False
    )

    # Evaluasi model lama
    cm_v02 = evaluate_model(
        baseline_model, patient_loader, device
    )

    # Evaluasi model baru
    cm_v03 = evaluate_model(
        model, patient_loader, device
    )

    necrotic_gt = cm_v03[4, :].sum().item()

    if necrotic_gt == 0:
        print(f"{patient_id}: N/A (No Necrotic)")
        continue

    dice_v02 = (
        calculate_metrics(cm_v02)["dice_per_class"][4]
        * 100
    )

    dice_v03 = (
        calculate_metrics(cm_v03)["dice_per_class"][4]
        * 100
    )

    difference = dice_v03 - dice_v02

    print(
        f"{patient_id:12} | "
        f"V0.2: {dice_v02:6.2f}% | "
        f"V0.3: {dice_v03:6.2f}% | "
        f"Delta: {difference:+6.2f} pp"
    )


for patient_id, indices in sorted(patient_indices.items()):

    patient_dataset = Subset(dataset, indices)

    patient_loader = DataLoader(
        patient_dataset,
        batch_size=2,
        shuffle=False
    )

    patient_cm = evaluate_model(
        model,
        patient_loader,
        device
    )

    patient_metrics = calculate_metrics(patient_cm)

    # Necrotic = kelas 4
    necrotic_gt = patient_cm[4, :].sum().item()
    necrotic_tp = patient_cm[4, 4].item()

    if necrotic_gt > 0:
        necrotic_dice = (
            patient_metrics["dice_per_class"][4] * 100
        )

        necrotic_recall = (
            necrotic_tp / necrotic_gt * 100
        )

        print(
            f"{patient_id} | "
            f"Images: {len(indices)} | "
            f"Nec GT: {necrotic_gt:,} | "
            f"Dice: {necrotic_dice:.2f}% | "
            f"Recall: {necrotic_recall:.2f}%"
        )
    else:
        print(
            f"{patient_id} | "
            f"Images: {len(indices)} | "
            f"Nec GT: 0 | "
            f"Dice: N/A | Recall: N/A"
        )

print("\nTotal validation patients:", len(patient_indices))

print("=" * 50)

print("Checkpoint:", checkpoint_path.name)
print("Saved epoch:", checkpoint["epoch"])
print(f"Logged Macro Dice: {expected * 100:.4f}%")
print(f"Reloaded Macro Dice: {actual * 100:.4f}%")

assert abs(expected - actual) < 1e-6, (
    "Reloaded checkpoint metrics do not match!"
)

assert abs(
    saved_metrics["wound_dice"] - metrics["wound_dice"]
) < 1e-6

print("\nCHECKPOINT INTEGRITY: PASS")


# Evaluasi lokalisasi luka per pasien
print("\nWOUND LOCALIZATION — PER PATIENT")
print("=" * 75)

for patient_id, indices in sorted(
    patient_indices.items()
):
    patient_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=2,
        shuffle=False
    )

    cm_v02 = evaluate_model(
        baseline_model, patient_loader, device
    )

    cm_v03 = evaluate_model(
        model, patient_loader, device
    )

    # Seluruh kelas jaringan = 1 sampai 5
    gt_wound = cm_v03[1:6, :].sum().item()

    # GT luka yang diprediksi sebagai jaringan luka
    detected_v02 = cm_v02[1:6, 1:6].sum().item()
    detected_v03 = cm_v03[1:6, 1:6].sum().item()

    recall_v02 = (
        100 * detected_v02 / gt_wound
        if gt_wound > 0 else 0
    )

    recall_v03 = (
        100 * detected_v03 / gt_wound
        if gt_wound > 0 else 0
    )

    delta = recall_v03 - recall_v02

    print(
        f"{patient_id:12} | "
        f"V0.2: {recall_v02:6.2f}% | "
        f"V0.3: {recall_v03:6.2f}% | "
        f"Delta: {delta:+7.2f} pp"
    )
