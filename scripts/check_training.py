
from pathlib import Path
import sys
import math
import time

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woundvision.model import build_model
from woundvision.data import WoundDataset
from woundvision.train import train_one_epoch
from woundvision.config import DATA_ROOT, CHECKPOINT_V02


# 1. Pilih CPU
device = torch.device("cpu")
torch.set_num_threads(4)

print("WOUNDVISION TRAINING SMOKE TEST")
print("=" * 50)

# 2. Load dataset
dataset = WoundDataset(
    DATA_ROOT,
    "train.txt"
)

# Ambil satu gambar saja
small_dataset = Subset(dataset, [0])

loader = DataLoader(
    small_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# 3. Bangun model
model = build_model()

# 4. Load checkpoint V0.2
checkpoint = torch.load(
    CHECKPOINT_V02,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state"])
model.to(device)

print("Checkpoint: V0.2")
print("Training samples:", len(small_dataset))
print("Device:", device)

# 5. Siapkan optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# 6. Simpan contoh bobot sebelum training
before = next(model.parameters()).detach().clone()

# 7. Jalankan training satu batch
print("\nStarting one-batch training...", flush=True)

start = time.perf_counter()

loss = train_one_epoch(
    model,
    loader,
    optimizer,
    device
)

duration = time.perf_counter() - start

# 8. Periksa apakah bobot berubah
after = next(model.parameters()).detach()

weight_change = (after - before).abs().max().item()

print("\nRESULTS")
print("-" * 50)

print(f"Training Loss: {loss:.4f}")
print(f"Duration: {duration:.2f} seconds")
print(f"Max weight change: {weight_change:.8f}")

assert math.isfinite(loss), "Loss is not finite!"
assert weight_change > 0, "Model weights did not update!"

print("\nTRAINING SMOKE TEST: PASS")
