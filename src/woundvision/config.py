
from pathlib import Path

# Lokasi project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "LUTSeg"

CHECKPOINT_V02 = (
    PROJECT_ROOT
    / "models"
    / "woundvision_v02_78189535.pth"
)

# Konfigurasi model
NUM_CLASSES = 6
IMAGE_SIZE = 320

# Konfigurasi training
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

CLASS_WEIGHTS = [
    0.25,    # Background
    1.0,     # Epithelial
    1.3412,  # Slough
    1.0135,  # Granulation
    3.0,     # Necrotic
    1.0      # Other
]
