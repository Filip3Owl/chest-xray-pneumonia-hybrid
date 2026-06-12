"""Dataset classes and DataLoader builders for Chest X-Ray data."""
import os
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}


def get_transforms(split: str, image_size: int = 224) -> A.Compose:
    """Returns augmentation pipeline appropriate for each data split."""
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3),
            # CLAHE: standard preprocessing for chest X-rays
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_tta_transforms(image_size: int = 224) -> List[A.Compose]:
    """Test-Time Augmentation transforms for inference uncertainty estimation."""
    base = [
        A.Resize(image_size, image_size),
        A.CLAHE(clip_limit=2.0, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.Rotate(limit=10, p=1.0)] + base),
        A.Compose([A.RandomBrightnessContrast(0.1, 0.1, p=1.0)] + base),
        A.Compose([A.CLAHE(clip_limit=4.0, p=1.0)] + base[1:]),  # stronger CLAHE
    ]


class ChestXRayDataset(Dataset):
    """
    Chest X-Ray Dataset (NORMAL / PNEUMONIA).

    Supports both classification (returns label) and feature extraction
    (returns raw numpy array for expert system).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[A.Compose] = None,
        return_path: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform or get_transforms(split, image_size)
        self.return_path = return_path

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        split_dir = self.root_dir / self.split
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.jpeg")) + sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")):
                self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Load as RGB — X-rays need 3-channel for ImageNet pretrained backbone
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        tensor = augmented["image"]

        if self.return_path:
            return tensor, label, str(img_path)
        return tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for imbalanced dataset."""
        counts = np.bincount([s[1] for s in self.samples])
        weights = 1.0 / counts
        return torch.tensor(weights / weights.sum(), dtype=torch.float32)

    def get_sample_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler."""
        counts = np.bincount([s[1] for s in self.samples])
        class_weights = 1.0 / counts
        return [float(class_weights[label]) for _, label in self.samples]

    @property
    def class_distribution(self) -> dict:
        counts = np.bincount([s[1] for s in self.samples])
        return {IDX_TO_CLASS[i]: int(c) for i, c in enumerate(counts)}


def build_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    The official val split has only 16 images per class — we rely on
    use_weighted_sampler to balance the much larger training set.
    """
    train_ds = ChestXRayDataset(data_root, "train", image_size)
    val_ds   = ChestXRayDataset(data_root, "val",   image_size, get_transforms("val",  image_size))
    test_ds  = ChestXRayDataset(data_root, "test",  image_size, get_transforms("test", image_size))

    sampler = None
    if use_weighted_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
