"""Upload WoundVision AI V6 Two-Stage ONNX models and model card to Hugging Face."""

import argparse
from pathlib import Path
import sys
from huggingface_hub import HfApi

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"

V6_FILES = [
    MODELS_DIR / "woundvision_v6_stage1_wound.onnx",
    MODELS_DIR / "woundvision_v6_stage2_tissue.onnx",
]

MODEL_CARD = """---
license: mit
datasets:
  - ksanchez84/LUTSeg
  - praveen202/DFU_Datasets
  - rupeshsah/wound
tags:
  - wound-segmentation
  - medical-imaging
  - unet
  - resnet18
  - resnet34
  - onnx
  - computer-vision
---

# WoundVision AI — V6-TwoStage & V5-Augmented (ONNX)

Experimental wound tissue segmentation models exported to **ONNX FP32** (`opset 17`) for lightweight CPU inference with ONNX Runtime.

## Available ONNX Artifacts
1. **`V6-TwoStage` (Recommended for Research / Demo)**:
   - `woundvision_v6_stage1_wound.onnx` (`54.66 MB`) — Stage 1 `U-Net ResNet18` Binary Wound Detector (`320×320`)
   - `woundvision_v6_stage2_tissue.onnx` (`93.22 MB`) — Stage 2 `U-Net ResNet34` ROI 6-Class Tissue Segmenter (`320×320`)
2. **`V5-Augmented` (Single-Stage, 100% CC BY 4.0 LUTSeg only)**:
   - `woundvision_v5_augmented.onnx` (`93.22 MB`) — Single-stage `U-Net ResNet34` (`320×320`)

## Validation Metrics (LUTSeg, 30 images, patient-level split)
| Metric | `V5-Augmented` | **`V6-TwoStage`** | Delta |
| :--- | :---: | :---: | :---: |
| Pixel Accuracy | `90.54%` | **`92.19%`** | `+1.65%` |
| Wound Dice (Binary) | `81.18%` | **`85.94%`** | `+4.76%` |
| Clinical-3 Macro Dice (Slough / Granulation / Necrotic) | `59.45%` | **`64.04%`** | `+4.59%` |
| Macro Tissue Dice (5 classes) | `43.16%` | **`47.04%`** | `+3.88%` |
| Granulation Dice | `74.89%` | **`78.02%`** | `+3.13%` |
| Necrotic Dice | `60.91%` | **`65.91%`** | `+5.00%` |
| Slough Dice | `42.54%` | **`48.18%`** | `+5.64%` |
| Epithelial Dice | `37.43%` | **`43.09%`** | `+5.66%` |
| Normal Skin False Positive Rate | — | **`0.0%`** | — |

## Dataset, Attribution & Citation
- **Primary Multi-Class Dataset**: **[LUTSeg](https://huggingface.co/datasets/ksanchez84/LUTSeg)** (*LUTSeg: A Longitudinal Multi-Expert Dataset for Ulcer Tissue Segmentation*, Eleventh ISIC Skin Image Analysis Workshop @ MICCAI 2026), released under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** ([Paper](https://huggingface.co/papers/2608.25866)). Source images originate from **[CO2Wounds](https://data.mendeley.com/datasets/nkw5gx57hw/1)** ([DOI: 10.1016/j.compbiomed.2023.107753](https://doi.org/10.1016/j.compbiomed.2023.107753)).
- **V6 Stage 1 Auxiliary Datasets (Research Track)**:
  - [praveen202/DFU_Datasets](https://huggingface.co/datasets/praveen202/DFU_Datasets) (FUSeg 2021 & DFUC2022 reference cohorts — intended for research/non-commercial demo use)
  - [rupeshsah/wound](https://huggingface.co/datasets/rupeshsah/wound) (`normal_*.png` healthy skin hard negatives, Apache 2.0)

## Disclaimer
Experimental research prototype. Not clinically validated, not a medical device, and not intended for diagnosis or clinical treatment decisions.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload V6 ONNX models and card to Hugging Face.")
    parser.add_argument("--repo-id", type=str, default=None, help="Target Model repo ID, e.g. salmanid/woundvision-ai")
    parser.add_argument("--card-only", action="store_true", help="Upload only README.md model card")
    args = parser.parse_args()

    if not args.card_only:
        for p in V6_FILES:
            if not p.exists():
                print(f"ERROR: Missing ONNX file {p}", file=sys.stderr)
                sys.exit(1)

    api = HfApi()
    username = api.whoami()["name"]
    repo_id = args.repo_id or f"{username}/woundvision-ai"

    print(f"Authenticated as: {username} -> {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print("Uploading README.md model card...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card with V6-TwoStage benchmarks and attributions",
    )

    if not args.card_only:
        for p in V6_FILES:
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"Uploading {p.name} ({size_mb:.2f} MB) ...")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=p.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {p.name} (V6-TwoStage ONNX FP32, opset 17)",
            )

    print("\nUpload complete!")
    print(f"Repo page : https://huggingface.co/{repo_id}")
    for p in V6_FILES:
        print(f"URL       : https://huggingface.co/{repo_id}/resolve/main/{p.name}")


if __name__ == "__main__":
    main()
