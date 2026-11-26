"""Train V6 Stage 2 (ROI Multi-Class Tissue Segmenter) and run end-to-end Two-Stage evaluation."""

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
from woundvision.evaluate import calculate_metrics, NUM_CLASSES

LUTSEG_DIR = PROJECT_DIR / "data" / "LUTSeg"
RUNS_DIR = PROJECT_DIR / "runs"

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ["background", "epithelial", "slough", "granulation", "necrotic", "other"]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_square_roi(
    binary_mask: np.ndarray,
    margin: float = 0.18,
    jitter_shift: float = 0.0,
) -> tuple[int, int, int, int] | None:
    """Compute square ROI (x1, y1, x2, y2) around foreground mask pixels."""
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None
    h, w = binary_mask.shape
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    bw = x_max - x_min + 1
    bh = y_max - y_min + 1
    side = int(max(bw, bh) * (1.0 + 2.0 * margin))
    side = max(side, 32)
    side = min(side, max(w, h))

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    if jitter_shift > 0.0:
        cx += random.uniform(-jitter_shift, jitter_shift) * side
        cy += random.uniform(-jitter_shift, jitter_shift) * side

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side

    # Shift inside image bounds where possible
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 = max(0, x1 - (x2 - w))
        x2 = w
    if y2 > h:
        y1 = max(0, y1 - (y2 - h))
        y2 = h

    return x1, y1, x2, y2


def crop_and_pad_square(img: Image.Image, mask: Image.Image | None, box: tuple[int, int, int, int]):
    """Crop box and pad to square so aspect ratio is preserved before resize."""
    x1, y1, x2, y2 = box
    crop_img = img.crop((x1, y1, x2, y2))
    cw, ch = crop_img.size
    side = max(cw, ch)
    pad_l = (side - cw) // 2
    pad_t = (side - ch) // 2

    sq_img = Image.new("RGB", (side, side), (0, 0, 0))
    sq_img.paste(crop_img, (pad_l, pad_t))

    sq_mask = None
    if mask is not None:
        crop_mask = mask.crop((x1, y1, x2, y2))
        sq_mask = Image.new("L", (side, side), 0)
        sq_mask.paste(crop_mask, (pad_l, pad_t))

    meta = (x1, y1, x2, y2, pad_l, pad_t, side)
    return sq_img, sq_mask, meta


class ROITissueDataset(Dataset):
    def __init__(self, split_file: str, image_size: int = 320, augment: bool = False):
        self.image_size = image_size
        self.augment = augment
        lines = [l.strip() for l in (LUTSEG_DIR / split_file).read_text().splitlines() if l.strip()]
        self.samples = [(LUTSEG_DIR / l.split()[0], LUTSEG_DIR / l.split()[1]) for l in lines]

    def __len__(self) -> int:
        return len(self.samples)

    def _color_preserving_augment(self, img: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if random.random() < 0.5:
            k = random.choice([90, 180, 270])
            img = TF.rotate(img, k)
            mask = TF.rotate(mask, k)
        if random.random() < 0.6:
            angle = random.uniform(-20.0, 20.0)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        # Color-preserving: gentle brightness/contrast ONLY (NO hue shift!)
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.90, 1.10))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.90, 1.10))
        return img, mask

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        with Image.open(img_path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        with Image.open(mask_path) as mk:
            mask = mk.copy()

        mask_np = np.array(mask)
        bin_mask = ((mask_np > 0) & (mask_np < NUM_CLASSES)).astype(np.uint8)

        margin = random.uniform(0.10, 0.28) if self.augment else 0.18
        jitter = 0.08 if self.augment else 0.0
        box = compute_square_roi(bin_mask, margin=margin, jitter_shift=jitter)
        if box is None:
            box = (0, 0, img.width, img.height)

        sq_img, sq_mask, _ = crop_and_pad_square(img, mask, box)
        sq_img = sq_img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        sq_mask = sq_mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        if self.augment:
            sq_img, sq_mask = self._color_preserving_augment(sq_img, sq_mask)

        img_t = torch.from_numpy(np.array(sq_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(np.array(sq_mask, dtype=np.int64))
        return img_t, mask_t


def evaluate_oracle_roi(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    cm = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)

    with torch.inference_mode():
        for images, masks in loader:
            images = (images.to(device) - mean) / std
            preds = model(images).argmax(dim=1).cpu()
            valid = (masks != 255) & (masks >= 0) & (masks < NUM_CLASSES)
            idx = masks[valid] * NUM_CLASSES + preds[valid]
            cm += torch.bincount(idx.long(), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)

    m = calculate_metrics(cm)
    d = m["dice_per_class"]
    m["clinical_3_dice"] = (d[2] + d[3] + d[4]) / 3.0
    return m


def evaluate_end_to_end(
    stage1_model: nn.Module,
    stage1_size: int,
    stage1_th: float,
    stage2_model: nn.Module,
    stage2_size: int,
    device: torch.device,
    margin: float = 0.18,
    hard_mask_with_stage1: bool = False,
    eval_size: int = 320,
) -> dict:
    """Exact end-to-end Two-Stage eval on LUTSeg/val.txt matching V5 protocol."""
    stage1_model.eval()
    stage2_model.eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    lines = [l.strip() for l in (LUTSEG_DIR / "val.txt").read_text().splitlines() if l.strip()]
    cm = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)

    with torch.inference_mode():
        for line in lines:
            img_rel, mask_rel = line.split()
            with Image.open(LUTSEG_DIR / img_rel) as im:
                orig_img = ImageOps.exif_transpose(im).convert("RGB")
            with Image.open(LUTSEG_DIR / mask_rel) as mk:
                gt_320 = np.array(mk.resize((eval_size, eval_size), Image.Resampling.NEAREST), dtype=np.int64)

            # 1. Stage 1 binary prediction on full image
            s1_in = orig_img.resize((stage1_size, stage1_size), Image.Resampling.BILINEAR)
            s1_t = torch.from_numpy(np.array(s1_in, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            s1_t = (s1_t - mean) / std
            s1_prob = torch.sigmoid(stage1_model(s1_t)).squeeze().cpu().numpy()
            s1_bin_orig = np.array(
                Image.fromarray((s1_prob >= stage1_th).astype(np.uint8) * 255).resize(orig_img.size, Image.Resampling.NEAREST)
            ) > 127

            full_pred = np.zeros((orig_img.height, orig_img.width), dtype=np.uint8)
            box = compute_square_roi(s1_bin_orig.astype(np.uint8), margin=margin, jitter_shift=0.0)

            if box is not None:
                sq_img, _, meta = crop_and_pad_square(orig_img, None, box)
                x1, y1, x2, y2, pad_l, pad_t, side = meta

                # 2. Stage 2 tissue prediction on ROI crop
                s2_in = sq_img.resize((stage2_size, stage2_size), Image.Resampling.BILINEAR)
                s2_t = torch.from_numpy(np.array(s2_in, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                s2_t = (s2_t - mean) / std
                s2_pred = stage2_model(s2_t).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                # Paste back from square crop to original image coordinates
                s2_sq = np.array(Image.fromarray(s2_pred).resize((side, side), Image.Resampling.NEAREST))
                cw, ch = x2 - x1, y2 - y1
                s2_crop = s2_sq[pad_t : pad_t + ch, pad_l : pad_l + cw]
                full_pred[y1:y2, x1:x2] = s2_crop

                if hard_mask_with_stage1:
                    full_pred[~s1_bin_orig] = 0

            pred_320 = np.array(Image.fromarray(full_pred).resize((eval_size, eval_size), Image.Resampling.NEAREST), dtype=np.int64)

            gt_t = torch.from_numpy(gt_320)
            pr_t = torch.from_numpy(pred_320)
            valid = (gt_t != 255) & (gt_t >= 0) & (gt_t < NUM_CLASSES)
            idx = gt_t[valid] * NUM_CLASSES + pr_t[valid]
            cm += torch.bincount(idx.long(), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)

    m = calculate_metrics(cm)
    d = m["dice_per_class"]
    m["clinical_3_dice"] = (d[2] + d[3] + d[4]) / 3.0
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--init-v5", action="store_true", help="Initialize weights from V5-Augmented resnet34 checkpoint")
    parser.add_argument("--stage1-ckpt", type=str, default="runs/v6_stage1_effb2/best_stage1.pth")
    parser.add_argument("--stage1-th", type=float, default=0.5)
    parser.add_argument("--out-name", type=str, default="v6_stage2_resnet34")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = get_device()
    out_dir = RUNS_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Stage 2 ROI Training | encoder={args.encoder} | size={args.image_size} | init_v5={args.init_v5} | device={device}")

    train_ds = ROITissueDataset("train.txt", args.image_size, augment=True)
    val_ds = ROITissueDataset("val.txt", args.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None if args.init_v5 else "imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    if args.init_v5:
        v5_path = RUNS_DIR / "notebook_v5_augmented" / "best_model.pth"
        v5_ckpt = torch.load(v5_path, map_location="cpu", weights_only=True)
        model.load_state_dict(v5_ckpt["model_state"], strict=True)
        print(f"Initialized from {v5_path}")
    model = model.to(device)

    # Load Stage 1 for live end-to-end validation if available
    s1_ckpt_path = PROJECT_DIR / args.stage1_ckpt
    stage1_model = None
    stage1_size = 384
    if s1_ckpt_path.exists():
        s1_ckpt = torch.load(s1_ckpt_path, map_location="cpu", weights_only=True)
        stage1_size = s1_ckpt["image_size"]
        stage1_model = smp.Unet(
            encoder_name=s1_ckpt["encoder"],
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        stage1_model.load_state_dict(s1_ckpt["model_state"], strict=True)
        stage1_model = stage1_model.to(device)
        stage1_model.eval()
        print(f"Loaded Stage 1 ({s1_ckpt['encoder']} @ {stage1_size}px, Wound Dice={s1_ckpt['lutseg_wound_dice']*100:.2f}%)")

    class_weights = torch.tensor([0.35, 1.1, 1.8, 1.0, 2.5, 0.2], device=device, dtype=torch.float32)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True, ignore_index=255)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    best_score = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = (images.to(device) - mean) / std
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        train_loss = running_loss / len(train_loader)
        roi_m = evaluate_oracle_roi(model, val_loader, device)
        e2e_m = None
        if stage1_model is not None:
            e2e_m = evaluate_end_to_end(
                stage1_model, stage1_size, args.stage1_th, model, args.image_size, device, hard_mask_with_stage1=False
            )

        elapsed = time.perf_counter() - t0
        key_score = e2e_m["clinical_3_dice"] if e2e_m is not None else roi_m["clinical_3_dice"]
        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "roi_clinical_3": round(roi_m["clinical_3_dice"], 5),
            "e2e_wound_dice": round(e2e_m["wound_dice"], 5) if e2e_m else None,
            "e2e_clinical_3": round(e2e_m["clinical_3_dice"], 5) if e2e_m else None,
            "e2e_macro_5": round(e2e_m["macro_tissue_dice"], 5) if e2e_m else None,
            "sec": round(elapsed, 1),
        }
        history.append(rec)

        e2e_str = (
            f" | E2E Wound={e2e_m['wound_dice']*100:.2f}% C3={e2e_m['clinical_3_dice']*100:.2f}% M5={e2e_m['macro_tissue_dice']*100:.2f}%"
            if e2e_m
            else ""
        )
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} | ROI C3={roi_m['clinical_3_dice']*100:.2f}%{e2e_str} | {elapsed:.1f}s", flush=True)

        if key_score > best_score:
            best_score = key_score
            torch.save(
                {
                    "encoder": args.encoder,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "clinical_3_dice": key_score,
                    "model_state": model.state_dict(),
                },
                out_dir / "best_stage2.pth",
            )

    # Final ablation Mode A (hard mask) vs Mode B (soft refinement) on best checkpoint
    best_ckpt = torch.load(out_dir / "best_stage2.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(best_ckpt["model_state"])

    final_report = {"args": vars(args), "best_epoch": best_ckpt["epoch"], "history": history}
    if stage1_model is not None:
        print("\n=== END-TO-END ABLATION ON LUTSeg/val.txt (30 images) ===")
        for mode_name, hard_mask in [("Mode A (Hard Mask Stage1)", True), ("Mode B (Soft Refinement)", False)]:
            m = evaluate_end_to_end(
                stage1_model, stage1_size, args.stage1_th, model, args.image_size, device, hard_mask_with_stage1=hard_mask
            )
            d = m["dice_per_class"]
            print(
                f"{mode_name:26s} | PixelAcc={m['pixel_accuracy']*100:.2f}% | "
                f"Wound={m['wound_dice']*100:.2f}% | C3={m['clinical_3_dice']*100:.2f}% | "
                f"M5={m['macro_tissue_dice']*100:.2f}% | "
                f"[Epi={d[1]*100:.1f} Slo={d[2]*100:.1f} Gra={d[3]*100:.1f} Nec={d[4]*100:.1f}]"
            )
            final_report[mode_name] = m

    (out_dir / "metrics.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"\nSaved best Stage 2 checkpoint to {out_dir / 'best_stage2.pth'}")


if __name__ == "__main__":
    main()
