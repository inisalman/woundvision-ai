
from pathlib import Path
from huggingface_hub import snapshot_download

# 1. Tentukan lokasi project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2. Lokasi penyimpanan dataset
DATASET_DIR = PROJECT_ROOT / "data" / "LUTSeg"

# 3. Download dataset dari Hugging Face
print("Downloading LUTSeg dataset...")

snapshot_download(
    repo_id="ksanchez84/LUTSeg",
    repo_type="dataset",
    local_dir=str(DATASET_DIR),
    allow_patterns=[
        "Images/**",
        "Masks/**",
        "train.txt",
        "val.txt",
        "README.md"
    ]
)

print("\nDATASET DOWNLOAD: COMPLETE")
print("Location:", DATASET_DIR)
