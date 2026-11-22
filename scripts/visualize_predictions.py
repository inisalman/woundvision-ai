
from pathlib import Path
import sys
import numpy as np


import torch
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woundvision.config import DATA_ROOT, CHECKPOINT_V02
from woundvision.data import WoundDataset
from woundvision.model import build_model
from woundvision.evaluate import MEAN, STD


# 1. Konfigurasi
device = torch.device("cpu")

CHECKPOINT_V03 = (
    ROOT / "runs"
    / "20261121_185427_e291f3"
    / "best_model.pth"
)

PATIENTS = ["Patient_20", "Patient_6"]


# 2. Fungsi memuat model
def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model = build_model()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model


model_v02 = load_model(CHECKPOINT_V02)
model_v03 = load_model(CHECKPOINT_V03)


# 3. Dataset validation
dataset = WoundDataset(DATA_ROOT, "val.txt")

mean = torch.tensor(MEAN).view(1, 3, 1, 1)
std = torch.tensor(STD).view(1, 3, 1, 1)


# 4. Fungsi prediksi
def predict(model, image):
    x = image.unsqueeze(0)
    x = (x - mean) / std

    with torch.inference_mode():
        logits = model(x)
        prediction = logits.argmax(dim=1)

    return prediction.squeeze(0).numpy()


# 5. Pilih gambar dengan Necrotic terbanyak
def select_image(patient_id):
    indices = [
        i for i, (image_path, _) in enumerate(
            dataset.samples
        )
        if image_path.parent.name == patient_id
    ]

    if not indices:
        raise ValueError(f"Patient not found: {patient_id}")

    return max(
        indices,
        key=lambda i: (dataset[i][1] == 4).sum().item()
    )


# 6. Warna masing-masing kelas
class_names = [
    "Background", "Epithelial", "Slough",
    "Granulation", "Necrotic", "Other"
]

colors = [
    "#1C1C1C", "#F7B6AE", "#F2C94C",
    "#E44D5B", "#7C3AED", "#2CB67D"
]

cmap = ListedColormap(colors)


# 7. Visualisasi
fig, axes = plt.subplots(
    len(PATIENTS), 4,
    figsize=(16, 8)
)

for row, patient_id in enumerate(PATIENTS):
    index = select_image(patient_id)
    image, mask = dataset[index]

    original = image.permute(1, 2, 0).numpy()
    gt = mask.numpy()

    pred_v02 = predict(model_v02, image)
    pred_v03 = predict(model_v03, image)

    panels = [
        original, gt, pred_v02, pred_v03
    ]

    titles = [
        f"Original - {patient_id}",
        "Ground Truth",
        "Prediction V0.2",
        "Prediction V0.3"
    ]

    for col, (panel, title) in enumerate(
        zip(panels, titles)
    ):
        ax = axes[row, col]

        if col == 0:
            ax.imshow(panel)
        else:
            ax.imshow(
                panel,
                cmap=cmap,
                vmin=0,
                vmax=5,
                interpolation="nearest"
            )

        ax.set_title(title)
        ax.axis("off")


# 8. Legenda
legend = [
    Patch(color=color, label=name)
    for name, color in zip(class_names, colors)
]

fig.legend(
    handles=legend,
    loc="lower center",
    ncol=6
)

fig.suptitle(
    "WoundVision AI | V0.2 vs V0.3",
    fontsize=16
)

plt.tight_layout(rect=[0, 0.07, 1, 0.94])


# 9. Simpan gambar
output_dir = ROOT / "runs" / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "v02_vs_v03.png"

plt.savefig(
    output_path,
    dpi=160,
    bbox_inches="tight"
)

plt.close()

print("VISUALIZATION: PASS")
print("Saved to:", output_path)

# Bandingkan overlay anotasi dan prediksi
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

for row, patient_id in enumerate(PATIENTS):

    index = select_image(patient_id)
    image, mask = dataset[index]

    original = image.permute(1, 2, 0).numpy()
    gt = mask.numpy()
    pred = predict(model_v03, image)

    panels = [
        ("Original", None),
        ("GT Necrotic Overlay", gt),
        ("V0.3 Necrotic Overlay", pred)
    ]

    for col, (title, segmentation) in enumerate(panels):

        ax = axes[row, col]
        ax.imshow(original)

        if segmentation is not None:
            overlay = np.zeros(
                (*gt.shape, 4),
                dtype=np.float32
            )

            # Ungu transparan untuk Necrotic
            overlay[segmentation == 4] = [
                0.49, 0.23, 0.93, 0.55
            ]

            ax.imshow(overlay)

        ax.set_title(f"{patient_id} - {title}")
        ax.axis("off")

    gt_pixels = int((gt == 4).sum())
    pred_pixels = int((pred == 4).sum())

    print(
        f"{patient_id}: "
        f"GT Necrotic={gt_pixels}, "
        f"Predicted Necrotic={pred_pixels}"
    )

plt.tight_layout()

overlay_path = output_dir / "necrotic_overlay.png"

plt.savefig(
    overlay_path,
    dpi=160,
    bbox_inches="tight"
)

plt.close()

print("OVERLAY VISUALIZATION: PASS")
print("Saved to:", overlay_path)


# Diagnosis: Wound localization vs tissue classification

print("\nWOUND LOCALIZATION ANALYSIS")
print("=" * 55)

for patient_id in PATIENTS:

    index = select_image(patient_id)
    image, mask = dataset[index]

    pred = predict(model_v03, image)

    gt = mask.numpy()

    # Seluruh jaringan luka = kelas 1 sampai 5
    gt_wound = (gt >= 1) & (gt <= 5)
    pred_wound = (pred >= 1) & (pred <= 5)

    # Necrotic = kelas 4
    gt_necrotic = (gt == 4)
    pred_necrotic = (pred == 4)

    # Berapa banyak piksel luka GT yang ditemukan?
    wound_detected = (
        gt_wound & pred_wound
    ).sum()

    wound_total = gt_wound.sum()

    wound_recall = (
        100 * wound_detected / wound_total
        if wound_total > 0 else 0
    )

    print(f"\n{patient_id}")
    print("-" * 40)

    print("GT Wound Pixels:", gt_wound.sum())
    print("Predicted Wound Pixels:", pred_wound.sum())

    print(f"Wound Pixel Recall: {wound_recall:.2f}%")

    print("GT Necrotic Pixels:", gt_necrotic.sum())
    print(
        "Predicted Necrotic Pixels:",
        pred_necrotic.sum()
    )

    print("\nPredictions inside GT Necrotic:")

    for class_id, class_name in enumerate(class_names):
        count = (
            pred[gt_necrotic] == class_id
        ).sum()

        print(f"{class_name:13}: {count}")


# Diagnosis probabilitas Background dan Necrotic

print("\nMODEL CONFIDENCE ANALYSIS")
print("=" * 55)

for patient_id in PATIENTS:

    index = select_image(patient_id)
    image, mask = dataset[index]

    x = image.unsqueeze(0)
    x = (x - mean) / std

    # Ground truth untuk analisis saja
    gt_wound = (mask >= 1) & (mask <= 5)
    gt_necrotic = mask == 4

    print(f"\n{patient_id}")
    print("-" * 40)

    for model_name, model in [
        ("V0.2", model_v02),
        ("V0.3", model_v03)
    ]:

        with torch.inference_mode():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]

        # Probabilitas background
        prob_bg = probs[0]

        # Probabilitas necrotic
        prob_nec = probs[4]

        # Probabilitas jaringan non-background
        prob_wound = 1.0 - prob_bg

        avg_wound = (
            prob_wound[gt_wound].mean().item()
        )

        avg_necrotic = (
            prob_nec[gt_necrotic].mean().item()
        )

        confident_bg = (
            (prob_bg[gt_wound] > 0.90)
            .float()
            .mean()
            .item()
        )

        print(f"\nModel {model_name}")
        print(
            f"Avg wound probability: "
            f"{avg_wound * 100:.2f}%"
        )
        print(
            f"Avg necrotic probability: "
            f"{avg_necrotic * 100:.2f}%"
        )
        print(
            f"GT wound predicted BG >90%: "
            f"{confident_bg * 100:.2f}%"
        )

print("\nCONFIDENCE ANALYSIS: PASS")

