"""Dataset classes and DataLoader builders for Chest X-Ray data."""
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Disable albumentations' online version check before importing it — the SSL
# request fails on macOS (CERTIFICATE_VERIFY_FAILED) and only emits a noisy
# warning. Must be set prior to the import to take effect. Recent albumentations
# uses NO_ALBUMENTATIONS_UPDATE; the older name is kept for compatibility.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
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
        samples: Optional[Sequence[Tuple[Path, int]]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform or get_transforms(split, image_size)
        self.return_path = return_path

        # If an explicit sample list is provided (e.g. from a stratified
        # re-split) use it directly; otherwise scan the split directory.
        if samples is not None:
            self.samples: List[Tuple[Path, int]] = [(Path(p), int(l)) for p, l in samples]
        else:
            self.samples = []
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


def _patient_key(img_path: Path) -> str:
    """
    Group key for patient-level splitting.

    PNEUMONIA files are named like ``person123_bacteria_4.jpeg`` — every image
    of the same ``personN`` is the SAME patient, so they must never be split
    across train/val (that would leak patient-specific cues). NORMAL files have
    no consistent patient id, so each image is treated as its own group.
    """
    match = re.match(r"(person\d+)", img_path.stem, flags=re.IGNORECASE)
    return match.group(1).lower() if match else img_path.stem


def _scan_samples(root: Path, split: str) -> List[Tuple[Path, int]]:
    """Scan one split directory into a list of (path, label)."""
    samples: List[Tuple[Path, int]] = []
    split_dir = root / split
    for class_name, label in CLASS_TO_IDX.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for img_path in (sorted(class_dir.glob("*.jpeg"))
                         + sorted(class_dir.glob("*.jpg"))
                         + sorted(class_dir.glob("*.png"))):
            samples.append((img_path, label))
    return samples


def _build_patient_groups(root: Path, splits: Sequence[str]) -> Tuple[dict, dict]:
    """Group all images in the given splits by patient key.

    Returns (group_label, group_paths) where group_label maps patient_key->label
    and group_paths maps patient_key->list[Path].
    """
    group_label: dict = {}
    group_paths: dict = defaultdict(list)
    for split in splits:
        for img_path, label in _scan_samples(root, split):
            key = _patient_key(img_path)
            group_label[key] = label
            group_paths[key].append(img_path)
    return group_label, group_paths


def _expand_groups(keys: Sequence[str], group_label: dict, group_paths: dict) -> List[Tuple[Path, int]]:
    """Expand patient keys back into a flat list of (path, label)."""
    out: List[Tuple[Path, int]] = []
    for k in keys:
        for p in group_paths[k]:
            out.append((p, group_label[k]))
    return out


def _pool_splits(include_official_val: bool, source_splits: Sequence[str]) -> List[str]:
    splits = list(source_splits)
    if include_official_val and "val" not in splits:
        splits.append("val")
    return splits


def make_stratified_splits(
    data_root: str,
    val_split: float = 0.15,
    seed: int = 42,
    include_official_val: bool = True,
    source_splits: Sequence[str] = ("train",),
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Build a reproducible, patient-aware, class-stratified train/val split.

    The official Kaggle val split has only 16 images, making it useless for
    model selection or threshold tuning. We instead carve a proper validation
    set out of the training data. The ``test`` split is never touched here, so
    there is no leakage into final evaluation.

    Splitting happens at the *patient* level (see ``_patient_key``) to avoid
    leaking images of the same patient across train and val. Because each
    patient belongs to a single class, we can stratify the groups by their
    label and still preserve the overall class balance.

    Returns (train_samples, val_samples) as lists of (Path, label).
    """
    from sklearn.model_selection import train_test_split

    root = Path(data_root)
    splits = _pool_splits(include_official_val, source_splits)
    group_label, group_paths = _build_patient_groups(root, splits)

    keys = sorted(group_paths)  # deterministic ordering before the seeded split
    labels = [group_label[k] for k in keys]

    train_keys, val_keys = train_test_split(
        keys, test_size=val_split, stratify=labels, random_state=seed,
    )

    return (_expand_groups(train_keys, group_label, group_paths),
            _expand_groups(val_keys, group_label, group_paths))


def make_calibration_splits(
    data_root: str,
    val_split: float = 0.15,
    calib_split: float = 0.15,
    seed: int = 42,
    include_official_val: bool = True,
    source_splits: Sequence[str] = ("train",),
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Patient-aware, class-stratified **3-way** split: train / val / calibration.

    Roles (test is never touched here, kept for final evaluation only):
      - **train**       — neural-network training
      - **val**         — model selection, early stopping, threshold tuning
      - **calibration** — meta-learner training, kept SEPARATE from val so the
                          hybrid fusion is never tuned on data used to select
                          the network (removes the stacking leakage caveat).

    Splitting is done at the patient level and stratified by class, identical in
    spirit to :func:`make_stratified_splits`. Using the same ``seed`` and
    fractions guarantees the partition is reproducible across notebooks.

    Returns (train_samples, val_samples, calib_samples).
    """
    from sklearn.model_selection import train_test_split

    root = Path(data_root)
    splits = _pool_splits(include_official_val, source_splits)
    group_label, group_paths = _build_patient_groups(root, splits)

    keys = sorted(group_paths)
    labels = [group_label[k] for k in keys]

    holdout_frac = val_split + calib_split
    if not 0 < holdout_frac < 1:
        raise ValueError(f"val_split + calib_split must be in (0, 1), got {holdout_frac}")

    # First peel off the combined (val + calib) hold-out, then split it in two.
    train_keys, hold_keys = train_test_split(
        keys, test_size=holdout_frac, stratify=labels, random_state=seed,
    )
    hold_labels = [group_label[k] for k in hold_keys]
    val_keys, calib_keys = train_test_split(
        hold_keys, test_size=calib_split / holdout_frac,
        stratify=hold_labels, random_state=seed,
    )

    return (_expand_groups(train_keys, group_label, group_paths),
            _expand_groups(val_keys, group_label, group_paths),
            _expand_groups(calib_keys, group_label, group_paths))


def build_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    val_split: float = 0.15,
    calib_split: float = 0.15,
    seed: int = 42,
    include_official_val: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    The official val split has only 16 images, so we carve proper, patient-
    aware, class-stratified splits out of the training pool. When
    ``calib_split > 0`` a **3-way** split (train/val/calibration) is used via
    ``make_calibration_splits`` so the meta-learner (notebook 05) can be trained
    on the held-out calibration set — obtain it with the same ``seed``,
    ``val_split`` and ``calib_split`` for an identical partition. The official
    ``test`` split is loaded untouched for final evaluation.

    Shuffling/sampling is seeded (``make_generator`` + ``seed_worker``) for
    reproducibility.
    """
    from ..utils.seed import seed_worker, make_generator

    if calib_split and calib_split > 0:
        train_samples, val_samples, _ = make_calibration_splits(
            data_root, val_split=val_split, calib_split=calib_split, seed=seed,
            include_official_val=include_official_val,
        )
    else:
        train_samples, val_samples = make_stratified_splits(
            data_root, val_split=val_split, seed=seed,
            include_official_val=include_official_val,
        )

    train_ds = ChestXRayDataset(data_root, "train", image_size,
                                get_transforms("train", image_size), samples=train_samples)
    val_ds   = ChestXRayDataset(data_root, "val",   image_size,
                                get_transforms("val",   image_size), samples=val_samples)
    test_ds  = ChestXRayDataset(data_root, "test",  image_size, get_transforms("test", image_size))

    generator = make_generator(seed)

    sampler = None
    if use_weighted_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
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
