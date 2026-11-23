
from pathlib import Path
import sys
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from woundvision.data import WoundDataset


dataset_root = project_root / "data" / "LUTSeg"

train = WoundDataset(dataset_root, "train.txt")
val = WoundDataset(dataset_root, "val.txt")

# Periksa jumlah gambar
assert len(train) == 111
assert len(val) == 30

# Periksa apakah pasien train dan val terpisah
train_patients = {
    image_path.parent.name
    for image_path, _ in train.samples
}

val_patients = {
    image_path.parent.name
    for image_path, _ in val.samples
}

assert train_patients.isdisjoint(val_patients), (
    "DATA LEAKAGE: Pasien train dan validation overlap!"
)

# Periksa satu sampel
image, mask = train[0]

assert image.shape == (3, 320, 320)
assert mask.shape == (320, 320)

valid_classes = set(range(6)) | {255}

assert set(torch.unique(mask).tolist()).issubset(
    valid_classes
)

print("WOUNDVISION DATASET TEST")
print("-" * 45)
print("Training samples:", len(train))
print("Validation samples:", len(val))
print("Training patients:", len(train_patients))
print("Validation patients:", len(val_patients))
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask classes:", torch.unique(mask).tolist())
print("Patient overlap:", train_patients & val_patients)
print("\nDATASET TEST: PASS")
