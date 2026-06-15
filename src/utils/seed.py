"""Reproducibility helpers — global seeding for torch / numpy / random.

The README advertises "fixed seeds (42)", but training previously seeded
nothing in torch, so runs were not reproducible. ``set_seed`` fixes that, and
``seed_worker`` + a seeded ``Generator`` make DataLoader workers deterministic.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG that affects training for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` — derive each worker's seed from torch's.

    PyTorch sets a unique base seed per worker; we propagate it to numpy and
    random so augmentations are deterministic given the global seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """A seeded ``torch.Generator`` for DataLoader shuffling/sampling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
