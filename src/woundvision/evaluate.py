
import torch


NUM_CLASSES = 6

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def evaluate_model(model, val_loader, device):
    """Evaluate all validation images."""

    model.eval()

    cm = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.long
    )

    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    with torch.inference_mode():
        for images, masks in val_loader:

            # Same preprocessing as Google Colab
            images = images.to(device)
            images = (images - mean) / std

            logits = model(images)
            predictions = logits.argmax(dim=1).cpu()

            masks = masks.cpu()

            # Ignore unlabeled pixels
            valid = (
                (masks != 255)
                & (masks >= 0)
                & (masks < NUM_CLASSES)
            )

            indices = (
                masks[valid] * NUM_CLASSES
                + predictions[valid]
            )

            cm += torch.bincount(
                indices.long(),
                minlength=NUM_CLASSES ** 2
            ).reshape(NUM_CLASSES, NUM_CLASSES)

    return cm


def calculate_metrics(cm):
    """Calculate segmentation metrics."""

    cm = cm.float()

    tp = cm.diag()
    true_pixels = cm.sum(dim=1)
    pred_pixels = cm.sum(dim=0)

    dice = (
        2 * tp
        / (true_pixels + pred_pixels).clamp_min(1)
    )

    pixel_accuracy = tp.sum() / cm.sum()

    # Five tissue classes, excluding Background
    macro_tissue_dice = dice[1:6].mean()

    # Binary wound segmentation: 0 vs 1-5
    wound_tp = cm[1:6, 1:6].sum()
    wound_fn = cm[1:6, 0].sum()
    wound_fp = cm[0, 1:6].sum()

    wound_dice = (
        2 * wound_tp
        / (2 * wound_tp + wound_fn + wound_fp).clamp_min(1)
    )

    return {
        "pixel_accuracy": pixel_accuracy.item(),
        "macro_tissue_dice": macro_tissue_dice.item(),
        "wound_dice": wound_dice.item(),
        "dice_per_class": dice.tolist()
    }
