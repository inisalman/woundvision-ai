from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

class WoundDataset(Dataset):
    def __init__(self, root_dir, split_file, image_size=320):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.samples = []

        # Baca daftar pasangan gambar dan mask
        lines = (
            (self.root_dir / split_file)
            .read_text(encoding="utf-8")
            .splitlines()
        )

        for line in lines:
            
            if not line.strip():
                continue

            parts = line.strip().split()

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid dataset entry: {line}"
                )

            image_path, mask_path = parts

            self.samples.append((
                self.root_dir / image_path,
                self.root_dir / mask_path
            ))

    def __len__(self):
         return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        # Baca gambar RGB
        with Image.open(image_path) as img:
            image = img.convert("RGB").resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR
            )

         # Baca indexed segmentation mask
        with Image.open(mask_path) as mask:
            mask = mask.resize(
                (self.image_size, self.image_size),
                Image.Resampling.NEAREST
                )

            mask_array = np.array(
                mask, dtype=np.int64
            )

            # Konversi menjadi tensor
            image_array = np.array(image).copy()

            image_tensor = (
                torch.from_numpy(image_array)
                .permute(2, 0, 1)
                .float() / 255.0
            )

            mask_tensor = torch.from_numpy(
                mask_array.copy()
            ).long()

            return image_tensor, mask_tensor