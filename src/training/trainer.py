"""Training loop with MLflow integration, AMP, and clinical metrics tracking."""
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    precision_score, recall_score,
)

from ..models.neural_net import ChestXRayNet
from ..utils.metrics import compute_clinical_metrics


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Full training pipeline for ChestXRayNet with:
      - Automatic Mixed Precision (AMP)
      - Weighted loss for class imbalance
      - Stage-wise fine-tuning (freeze backbone → unfreeze)
      - MLflow experiment tracking
      - Clinical metrics (sensitivity, specificity, AUC)
      - Model checkpointing (best AUC)
    """

    def __init__(
        self,
        model: ChestXRayNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        epochs: int = 30,
        freeze_backbone_epochs: int = 3,
        class_weights: Optional[torch.Tensor] = None,
        scheduler_type: str = "cosine_annealing",
        early_stopping_patience: int = 7,
        gradient_clip: float = 1.0,
        use_amp: bool = True,
        checkpoint_dir: str = "models_saved",
        experiment_name: str = "chest-xray-pneumonia",
        run_name: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp and (device == "cuda")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss with class-imbalance weighting
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        if scheduler_type == "cosine_annealing":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-7)
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=3
            )

        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.scaler = GradScaler() if self.use_amp else None

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.best_auc = 0.0
        self.best_model_path: Optional[str] = None

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            total_loss += loss.item()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

        metrics = compute_clinical_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )
        metrics["loss"] = total_loss / len(self.train_loader)
        return metrics

    @torch.no_grad()
    def _val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            total_loss += loss.item()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

        metrics = compute_clinical_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    def train(self, params_to_log: Optional[Dict] = None) -> Dict[str, float]:
        """Full training loop with MLflow tracking."""
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.run_name) as run:
            print(f"\nMLflow Run ID: {run.info.run_id}")

            # Log hyperparameters
            mlflow.log_params({
                "backbone": self.model.backbone_name,
                "epochs": self.epochs,
                "lr": self.optimizer.param_groups[0]["lr"],
                "batch_size": self.train_loader.batch_size,
                "freeze_backbone_epochs": self.freeze_backbone_epochs,
                "device": self.device,
                **(params_to_log or {}),
            })

            # Stage 1: Freeze backbone, train only classifier head
            print(f"\n{'='*60}")
            print("Stage 1: Training classifier head (backbone frozen)")
            print(f"{'='*60}")
            self.model.freeze_backbone(True)

            best_metrics = {}
            for epoch in range(1, self.epochs + 1):
                # Unfreeze backbone after warm-up phase
                if epoch == self.freeze_backbone_epochs + 1:
                    print(f"\nStage 2: Fine-tuning full network (epoch {epoch})")
                    self.model.freeze_backbone(False)
                    # Lower LR for backbone layers
                    for pg in self.optimizer.param_groups:
                        pg["lr"] *= 0.1

                t0 = time.time()
                train_metrics = self._train_epoch()
                val_metrics   = self._val_epoch()

                elapsed = time.time() - t0

                # Log metrics to MLflow
                log_dict = {
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}":   v for k, v in val_metrics.items()},
                }
                mlflow.log_metrics(log_dict, step=epoch)

                auc = val_metrics.get("auc_roc", 0.0)
                print(
                    f"Epoch {epoch:3d}/{self.epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_auc={auc:.4f} | "
                    f"val_sens={val_metrics.get('sensitivity', 0):.4f} | "
                    f"val_spec={val_metrics.get('specificity', 0):.4f} | "
                    f"{elapsed:.1f}s"
                )

                # Save best model
                if auc > self.best_auc:
                    self.best_auc = auc
                    best_metrics = val_metrics.copy()
                    ckpt_path = self.checkpoint_dir / "best_model.pth"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "backbone": self.model.backbone_name,
                    }, ckpt_path)
                    self.best_model_path = str(ckpt_path)
                    mlflow.log_artifact(str(ckpt_path))
                    print(f"  ✓ Best model saved (AUC={auc:.4f})")

                # Scheduler step
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(auc)
                else:
                    self.scheduler.step()

                # Early stopping
                if self.early_stopping.step(auc):
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

            # Log final model with MLflow Model Registry
            if self.best_model_path:
                self.model.load_state_dict(
                    torch.load(self.best_model_path, map_location=self.device)["model_state_dict"]
                )
                mlflow.pytorch.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name="ChestXRayNet",
                )

            mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
            print(f"\nTraining complete. Best Val AUC: {self.best_auc:.4f}")
            return best_metrics
