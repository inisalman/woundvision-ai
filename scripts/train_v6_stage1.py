"""Train and evaluate V6 Stage 1: Binary Wound Segmentation (with Hard Negatives)."""

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RUNS_DIR = PROJECT_DIR / "runs"

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BinaryWoundDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        image_size: int = 384,
        augment: bool = False,
        lutseg_repeat: int = 1,
    ):
        self.image_size = image_size
        self.augment = augment
        self.rows: list[dict] = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                reps = lutseg_repeat if row["source"] == "lutseg_train" else 1
                for _ in range(reps):
                    self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_mask(self, row: dict, size: tuple[int, int]) -> Image.Image:
        mtype = row["mask_type"]
        if mtype == "empty" or not row["mask_path"]:
            return Image.new("L", size, 0)

        with Image.open(PROJECT_DIR / row["mask_path"]) as mk:
            arr = np.array(mk)
            if arr.ndim == 3:
                arr = arr[..., 0]
            if mtype == "lutseg_multiclass":
                bin_arr = ((arr > 0) & (arr < 6)).astype(np.uint8) * 255
            else:
                thresh = 127 if arr.max() > 1 else 0
                bin_arr = (arr > thresh).astype(np.uint8) * 255
            return Image.fromarray(bin_arr, mode="L")

    def _apply_augment(self, img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
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
        if random.random() < 0.7:
            angle = random.uniform(-25.0, 25.0)
            scale = random.uniform(0.85, 1.15)
            max_dx = int(0.08 * img.width)
            max_dy = int(0.08 * img.height)
            trans = [random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy)]
            img = TF.affine(img, angle=angle, translate=trans, scale=scale, shear=[0.0, 0.0], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=angle, translate=trans, scale=scale, shear=[0.0, 0.0], interpolation=TF.InterpolationMode.NEAREST)
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
        return img, mask

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(PROJECT_DIR / row["image_path"]) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        mask = self._load_mask(row, img.size)

        img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        if self.augment:
            img, mask = self._apply_augment(img, mask)

        img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy((np.array(mask) > 127).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t, row["source"]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict:
    model.eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    lut_tp = lut_fp = lut_fn = 0.0
    norm_total = norm_fp_images = 0
    norm_fp_covs: list[float] = []

    with torch.inference_mode():
        for images, masks, sources in loader:
            images = (images.to(device) - mean) / std
            probs = torch.sigmoid(model(images)).cpu()
            preds = (probs >= threshold).float()

            for i, src in enumerate(sources):
                p = preds[i]
                g = masks[i]
                if src == "lutseg_val":
                    lut_tp += float((p * g).sum().item())
                    lut_fp += float((p * (1.0 - g)).sum().item())
                    lut_fn += float(((1.0 - p) * g).sum().item())
                elif src == "normal_skin_val":
                    norm_total += 1
                    cov = float(p.mean().item()) * 100.0
                    norm_fp_covs.append(cov)
                    if cov > 1.0:
                        norm_fp_images += 1

    wound_dice = (2.0 * lut_tp) / max(2.0 * lut_tp + lut_fp + lut_fn, 1.0)
    fpr = (norm_fp_images / norm_total * 100.0) if norm_total > 0 else 0.0
    mean_fp_cov = float(np.mean(norm_fp_covs)) if norm_fp_covs else 0.0
    return {
        "lutseg_wound_dice": round(wound_dice, 5),
        "normal_fpr_pct": round(fpr, 2),
        "normal_mean_fp_cov_pct": round(mean_fp_cov, 3),
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="efficientnet-b2")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lutseg-repeat", type=int, default=3)
    parser.add_argument("--out-name", type=str, default="v6_stage1_effb2")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = get_device()
    out_dir = RUNS_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Stage 1 Training | encoder={args.encoder} | size={args.image_size} | device={device}")

    train_ds = BinaryWoundDataset(DATA_DIR / "stage1_train.csv", args.image_size, augment=True, lutseg_repeat=args.lutseg_repeat)
    val_ds = BinaryWoundDataset(DATA_DIR / "stage1_val.csv", args.image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    bce = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    best_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        for step, (images, masks, _) in enumerate(train_loader, 1):
            images = (images.to(device) - mean) / std
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = 0.5 * bce(logits, masks) + 0.5 * dice(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step == 1 or step % 25 == 0 or step == len(train_loader):
                print(f"  [E{epoch:02d}] step {step:03d}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        scheduler.step()

        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device, threshold=0.5)
        elapsed = time.perf_counter() - t0
        record = {"epoch": epoch, "train_loss": round(train_loss, 5), **val_metrics, "sec": round(elapsed, 1)}
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} | "
            f"LUTSeg Wound Dice={val_metrics['lutseg_wound_dice']*100:.2f}% | "
            f"Normal FPR={val_metrics['normal_fpr_pct']:.1f}% ({val_metrics['normal_mean_fp_cov_pct']:.2f}% px) | {elapsed:.1f}s",
            flush=True,
        )

        if val_metrics["lutseg_wound_dice"] > best_dice:
            best_dice = val_metrics["lutseg_wound_dice"]
            torch.save(
                {
                    "encoder": args.encoder,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "lutseg_wound_dice": best_dice,
                    "normal_fpr_pct": val_metrics["normal_fpr_pct"],
                    "model_state": model.state_dict(),
                },
                out_dir / "best_stage1.pth",
            )

    # Threshold sweep on best model
    best_ckpt = torch.load(out_dir / "best_stage1.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(best_ckpt["model_state"])
    print("\nThreshold sweep on best Stage 1 checkpoint:")
    best_t, best_t_dice = 0.5, -1.0
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        m = evaluate(model, val_loader, device, threshold=th)
        print(f"  th={th:.1f} -> Wound Dice={m['lutseg_wound_dice']*100:.2f}%, Normal FPR={m['normal_fpr_pct']:.1f}%")
        if m["lutseg_wound_dice"] > best_t_dice:
            best_t, best_t_dice = th, m["lutseg_wound_dice"]

    summary = {
        "args": vars(args),
        "best_lutseg_wound_dice_th05": best_dice,
        "best_threshold": best_t,
        "best_lutseg_wound_dice": best_t_dice,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved best Stage 1 checkpoint to {out_dir / 'best_stage1.pth'} (best Wound Dice={best_t_dice*100:.2f}% @ th={best_t})")


if __name__ == "__main__":
    main()
