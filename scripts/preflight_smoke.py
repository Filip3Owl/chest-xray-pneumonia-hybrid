"""Pre-flight smoke test for the training pipeline.

Runs the FULL Trainer (dataloaders → model → 2 epochs incl. the Stage 1→2
backbone unfreeze → checkpoint → MLflow) on a tiny balanced subset, on CPU,
in ~1-2 min. Its job is to catch integration bugs BEFORE committing to the
real multi-hour run — not to produce a good model.

Usage:
    NO_ALBUMENTATIONS_UPDATE=1 python scripts/preflight_smoke.py
Exits non-zero if any sanity check fails.
"""
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import torch

from src.data.dataset import ChestXRayDataset, get_transforms, make_calibration_splits
from src.models import build_model
from src.training import Trainer
from src.utils import set_seed

DATA_ROOT = "chest_xray"
N_PER_CLASS = 4  # tiny subset per class


def _take_balanced(samples, n_per_class):
    """Grab the first n samples of each label for a small balanced subset."""
    buckets = {0: [], 1: []}
    for path, label in samples:
        if len(buckets[label]) < n_per_class:
            buckets[label].append((path, label))
    return buckets[0] + buckets[1]


def main() -> int:
    set_seed(42)
    device = "cpu"  # smoke test is intentionally hardware-agnostic

    train_all, val_all, _ = make_calibration_splits(DATA_ROOT, seed=42)
    train_sub = _take_balanced(train_all, N_PER_CLASS)
    val_sub = _take_balanced(val_all, N_PER_CLASS)
    print(f"subset -> train {Counter(l for _, l in train_sub)}, val {Counter(l for _, l in val_sub)}")

    train_ds = ChestXRayDataset(DATA_ROOT, "train", 224, get_transforms("train", 224), samples=train_sub)
    val_ds = ChestXRayDataset(DATA_ROOT, "val", 224, get_transforms("val", 224), samples=val_sub)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    model = build_model(backbone="densenet121", num_classes=2, dropout=0.5,
                        pretrained=False, device=device)  # pretrained=False: no network, faster

    with tempfile.TemporaryDirectory() as tmp:
        mlflow.set_tracking_uri(f"file:{tmp}/mlruns")
        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader, device=device,
            learning_rate=1e-4, epochs=2, freeze_backbone_epochs=1,  # epoch 2 unfreezes
            class_weights=train_ds.get_class_weights(),
            early_stopping_patience=99, use_amp=False,
            checkpoint_dir=f"{tmp}/ckpt", experiment_name="preflight-smoke",
            run_name="smoke",
        )
        trainer.train(params_to_log={"smoke": True})

        # --- Sanity checks ---
        checks = {
            "ran 2 epochs": len(trainer.history["epoch"]) == 2,
            "loss is finite": all(torch.isfinite(torch.tensor(h["loss"])) for h in trainer.history["train"]),
            "checkpoint saved": trainer.best_model_path and Path(trainer.best_model_path).exists(),
            "best AUC recorded": trainer.best_auc >= 0.0,
            "stage-2 LR < stage-1 LR": trainer.history["lr"][1] < trainer.history["lr"][0],
            "backbone unfrozen at end": any(p.requires_grad for p in model.features.parameters()),
        }

    print("\n--- pre-flight checks ---")
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed

    print("\nPRE-FLIGHT", "OK ✅" if ok else "FAILED ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
