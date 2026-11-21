
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from .config import CLASS_WEIGHTS
from .evaluate import MEAN, STD


def train_one_epoch(model, loader, optimizer, device):
    """Train model for one epoch."""

    model.train()

    # Normalisasi ImageNet
    mean = torch.tensor(
        MEAN, device=device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        STD, device=device
    ).view(1, 3, 1, 1)

    # Weighted Cross Entropy
    weights = torch.tensor(
        CLASS_WEIGHTS, device=device
    )

    ce_loss = nn.CrossEntropyLoss(
        weight=weights,
        ignore_index=255
    )

    # Dice Loss
    dice_loss = smp.losses.DiceLoss(
        mode="multiclass",
        from_logits=True,
        ignore_index=255
    )

    total_loss = 0.0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        images = (images - mean) / std

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Hitung loss
        loss = (
            ce_loss(outputs, masks)
            + dice_loss(outputs, masks)
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        # Backpropagation
        loss.backward()

        # Update parameter model
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
