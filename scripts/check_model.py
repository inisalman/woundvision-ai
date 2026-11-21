
from pathlib import Path
import sys
import torch

# Tambahkan folder src ke Python import path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from woundvision.model import build_model


# Gunakan CPU untuk tes kompatibilitas awal
device = torch.device("cpu")

# Buat model dari modul baru
model = build_model()

# Baca checkpoint V0.1
checkpoint_path = (
    project_root
    / "models"
    / "woundvision_mvp_3d6d1c55.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

# Masukkan bobot ke model
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

# Uji forward pass
dummy = torch.randn(1, 3, 320, 320)

with torch.inference_mode():
    output = model(dummy)

print("WOUNDVISION MODEL TEST")
print("-" * 40)
print("Device:", device)
print("Output shape:", output.shape)

assert output.shape == (1, 6, 320, 320)

print("MODEL TEST: PASS")
