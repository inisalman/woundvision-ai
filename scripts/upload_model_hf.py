"""Upload WoundVision AI V5-Augmented ONNX model and model card to Hugging Face."""

import argparse
from pathlib import Path
import sys
from huggingface_hub import HfApi

PROJECT_DIR = Path(__file__).resolve().parent.parent
ONNX_PATH = PROJECT_DIR / "models" / "woundvision_v5_augmented.onnx"

MODEL_CARD = """---
license: mit
datasets:
  - ksanchez84/LUTSeg
tags:
  - wound-segmentation
  - medical-imaging
  - unet
  - resnet34
  - onnx
  - computer-vision
---

# WoundVision AI — V5-Augmented (ONNX)

Experimental wound tissue segmentation model (**U-Net ResNet34**, `V5-Augmented`) exported to **ONNX FP32** (`opset 17`) for lightweight CPU inference with ONNX Runtime.

## Specifications
- **Input**: `input` — float32 tensor `(batch, 3, 320, 320)`, ImageNet normalized (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Output**: `logits` — float32 tensor `(batch, 6, 320, 320)`
- **Classes**: `0: background`, `1: epithelial`, `2: slough`, `3: granulation`, `4: necrotic`, `5: other`
- **Parity vs PyTorch**: `100.00%` pixel agreement (`max_logit_diff = 0.000027`)

## Validation Metrics (LUTSeg, 30 images, patient-level split)
| Metric | Score |
| :--- | :---: |
| Pixel Accuracy | `90.54%` |
| Wound Dice (Binary) | `81.18%` |
| Clinical-3 Macro Dice (Slough / Granulation / Necrotic) | `59.45%` |
| Macro Tissue Dice (5 tissue classes) | `43.16%` |
| Granulation Dice | `74.89%` |
| Necrotic Dice | `60.91%` |
| Slough Dice | `42.54%` |
| Epithelial Dice | `37.43%` |

## Dataset, Attribution & Citation
This model was trained and evaluated on the **[LUTSeg](https://huggingface.co/datasets/ksanchez84/LUTSeg)** dataset, released under the **[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**.

- **Dataset**: [ksanchez84/LUTSeg](https://huggingface.co/datasets/ksanchez84/LUTSeg) — *LUTSeg: A Longitudinal Multi-Expert Dataset for Ulcer Tissue Segmentation*, Eleventh ISIC Skin Image Analysis Workshop @ MICCAI 2026
- **Paper**: [https://huggingface.co/papers/2608.25866](https://huggingface.co/papers/2608.25866)
- **Source Images**: [CO2Wounds Dataset](https://data.mendeley.com/datasets/nkw5gx57hw/1) ([DOI: 10.1016/j.compbiomed.2023.107753](https://doi.org/10.1016/j.compbiomed.2023.107753))
- **Dataset License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Disclaimer
Experimental research prototype. Not clinically validated, not a medical device, and not intended for diagnosis or clinical treatment decisions.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload ONNX model to Hugging Face Model Repo.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Target Model repo ID, e.g. salmanid/woundvision-ai",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the model repo as private (default: public)",
    )
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="Upload only the README.md model card without re-uploading the ONNX binary",
    )
    args = parser.parse_args()

    if not args.card_only and not ONNX_PATH.exists():
        print(f"ERROR: ONNX model not found at {ONNX_PATH}", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = args.repo_id or f"{username}/woundvision-ai"

    print(f"Authenticated as: {username}")
    print(f"Creating/updating HF Model Repo: {repo_id} (private={args.private}) ...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print("Uploading README.md model card...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add LUTSeg (CC BY 4.0) & CO2Wounds attribution and full validation metrics",
    )

    if not args.card_only:
        size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
        print(f"Uploading {ONNX_PATH.name} ({size_mb:.2f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(ONNX_PATH),
            path_in_repo="woundvision_v5_augmented.onnx",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload V5-Augmented ONNX model (FP32, opset 17)",
        )

    direct_url = f"https://huggingface.co/{repo_id}/resolve/main/woundvision_v5_augmented.onnx"
    print("\nUpload complete!")
    print(f"Repo page  : https://huggingface.co/{repo_id}")
    print(f"MODEL_URL  : {direct_url}")


if __name__ == "__main__":
    main()
