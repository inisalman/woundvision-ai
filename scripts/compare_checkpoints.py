
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

v01_path = ROOT / "models" / "woundvision_mvp_3d6d1c55.pth"
v02_path = ROOT / "models" / "woundvision_v02_78189535.pth"

# Load kedua checkpoint
v01 = torch.load(
    v01_path,
    map_location="cpu",
    weights_only=True
)["model_state"]

v02 = torch.load(
    v02_path,
    map_location="cpu",
    weights_only=True
)["model_state"]

# Bandingkan tensor bobot
assert v01.keys() == v02.keys(), "Model keys berbeda!"

changed = 0
total = 0
max_difference = 0.0

for key in v01:
    a = v01[key]
    b = v02[key]

    if not torch.is_floating_point(a):
        continue

    total += 1

    if not torch.equal(a, b):
        changed += 1

    difference = (a - b).abs().max().item()
    max_difference = max(max_difference, difference)

print("WOUNDVISION CHECKPOINT COMPARISON")
print("=" * 45)
print("Total floating tensors:", total)
print("Changed tensors:", changed)
print("Max weight difference:", max_difference)

if changed > 0:
    print("\nCHECKPOINTS ARE DIFFERENT: PASS")
else:
    print("\nWARNING: MODEL WEIGHTS ARE IDENTICAL")
