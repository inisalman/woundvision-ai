"""Download Stage 1 wound datasets from Hugging Face and build verified train/val manifests."""

import csv
from pathlib import Path
import random
from PIL import Image
import numpy as np
from huggingface_hub import snapshot_download

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
LUTSEG_DIR = DATA_DIR / "LUTSeg"
DFU_DIR = DATA_DIR / "DFU_Datasets"
NORMAL_DIR = DATA_DIR / "normal_skin"

TRAIN_MANIFEST = DATA_DIR / "stage1_train.csv"
VAL_MANIFEST = DATA_DIR / "stage1_val.csv"


def download_datasets() -> None:
    print("[1/3] Downloading praveen202/DFU_Datasets (500 binary wound images + masks)...")
    snapshot_download(
        repo_id="praveen202/DFU_Datasets",
        repo_type="dataset",
        local_dir=str(DFU_DIR),
        allow_patterns=["external_fuseg/**", "internal_dfuc2022/**", "README.md"],
    )

    print("[2/3] Downloading normal skin hard-negative images from rupeshsah/wound...")
    snapshot_download(
        repo_id="rupeshsah/wound",
        repo_type="dataset",
        local_dir=str(NORMAL_DIR),
        allow_patterns=["images/normal_*.png", "README.md"],
    )

    print("[3/3] Ensuring LUTSeg Wound_Masks are present...")
    snapshot_download(
        repo_id="ksanchez84/LUTSeg",
        repo_type="dataset",
        local_dir=str(LUTSEG_DIR),
        allow_patterns=["Wound_Masks/**"],
    )


def _find_mask_for_image(img_path: Path, masks_dir: Path) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        cand = masks_dir / f"{img_path.stem}{ext}"
        if cand.exists():
            return cand
    return None


def _verify_pair(img_path: Path, mask_path: Path | None) -> tuple[bool, str]:
    try:
        with Image.open(img_path) as im:
            im.verify()
        if mask_path is not None:
            with Image.open(mask_path) as mk:
                arr = np.array(mk)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                if arr.size == 0:
                    return False, "empty mask"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def build_manifests() -> None:
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    # 1. LUTSeg train & val (strict patient-level split, ZERO leakage)
    lutseg_train_lines = [l.strip() for l in (LUTSEG_DIR / "train.txt").read_text().splitlines() if l.strip()]
    lutseg_val_lines = [l.strip() for l in (LUTSEG_DIR / "val.txt").read_text().splitlines() if l.strip()]

    val_image_set = set()
    for line in lutseg_val_lines:
        img_rel, mask_rel = line.split()
        img_path = LUTSEG_DIR / img_rel
        mask_path = LUTSEG_DIR / mask_rel
        val_image_set.add(str(img_path.resolve()))
        ok, err = _verify_pair(img_path, mask_path)
        assert ok, f"Invalid LUTSeg val pair {img_path}: {err}"
        val_rows.append({
            "image_path": str(img_path.relative_to(PROJECT_DIR)),
            "mask_path": str(mask_path.relative_to(PROJECT_DIR)),
            "mask_type": "lutseg_multiclass",
            "source": "lutseg_val",
        })

    for line in lutseg_train_lines:
        img_rel, mask_rel = line.split()
        img_path = LUTSEG_DIR / img_rel
        mask_path = LUTSEG_DIR / mask_rel
        assert str(img_path.resolve()) not in val_image_set, f"Leakage detected: {img_path}"
        ok, err = _verify_pair(img_path, mask_path)
        assert ok, f"Invalid LUTSeg train pair {img_path}: {err}"
        train_rows.append({
            "image_path": str(img_path.relative_to(PROJECT_DIR)),
            "mask_path": str(mask_path.relative_to(PROJECT_DIR)),
            "mask_type": "lutseg_multiclass",
            "source": "lutseg_train",
        })

    # 2. DFU_Datasets (external_fuseg + internal_dfuc2022) -> train only
    for sub, src_name in [("external_fuseg", "fuseg"), ("internal_dfuc2022", "dfuc2022")]:
        imgs_dir = DFU_DIR / sub / "images"
        masks_dir = DFU_DIR / sub / "masks"
        if not imgs_dir.exists():
            print(f"WARNING: Missing {imgs_dir}")
            continue
        matched = 0
        for img_path in sorted(imgs_dir.iterdir()):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            mask_path = _find_mask_for_image(img_path, masks_dir)
            if mask_path is None:
                continue
            ok, err = _verify_pair(img_path, mask_path)
            if not ok:
                print(f"Skipping bad pair {img_path}: {err}")
                continue
            train_rows.append({
                "image_path": str(img_path.relative_to(PROJECT_DIR)),
                "mask_path": str(mask_path.relative_to(PROJECT_DIR)),
                "mask_type": "binary",
                "source": src_name,
            })
            matched += 1
        print(f"Matched {matched} valid pairs from {src_name}")

    # 3. Normal skin hard negatives (80% train, 20% val for FPR measurement)
    normal_imgs = sorted((NORMAL_DIR / "images").glob("normal_*.png")) if (NORMAL_DIR / "images").exists() else []
    rng = random.Random(42)
    normal_imgs_shuffled = list(normal_imgs)
    rng.shuffle(normal_imgs_shuffled)
    val_count = max(1, int(len(normal_imgs_shuffled) * 0.2)) if normal_imgs_shuffled else 0
    normal_val = normal_imgs_shuffled[:val_count]
    normal_train = normal_imgs_shuffled[val_count:]

    for p in normal_train:
        ok, _ = _verify_pair(p, None)
        if ok:
            train_rows.append({
                "image_path": str(p.relative_to(PROJECT_DIR)),
                "mask_path": "",
                "mask_type": "empty",
                "source": "normal_skin_train",
            })
    for p in normal_val:
        ok, _ = _verify_pair(p, None)
        if ok:
            val_rows.append({
                "image_path": str(p.relative_to(PROJECT_DIR)),
                "mask_path": "",
                "mask_type": "empty",
                "source": "normal_skin_val",
            })

    fieldnames = ["image_path", "mask_path", "mask_type", "source"]
    with open(TRAIN_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(train_rows)

    with open(VAL_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(val_rows)

    # Summary stats
    from collections import Counter
    print("\n=== STAGE 1 MANIFEST SUMMARY ===")
    print(f"Train ({len(train_rows)}): {dict(Counter(r['source'] for r in train_rows))}")
    print(f"Val   ({len(val_rows)}): {dict(Counter(r['source'] for r in val_rows))}")
    print(f"Saved: {TRAIN_MANIFEST.relative_to(PROJECT_DIR)}, {VAL_MANIFEST.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    download_datasets()
    build_manifests()
