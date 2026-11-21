
from pathlib import Path
import sys
import json
import torch

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woundvision.model import build_model
from woundvision.data import WoundDataset
from woundvision.evaluate import evaluate_model, calculate_metrics

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

RUN_DIR = available_runs[-1]

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
metrics = calculate_metrics(cm)

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

# 6. Bandingkan
expected = saved_metrics["macro_tissue_dice"]
actual = metrics["macro_tissue_dice"]

print("CHECKPOINT INTEGRITY TEST")
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
