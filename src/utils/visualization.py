"""Visualization utilities: Grad-CAM, confusion matrix, ROC, dataset exploration."""
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


def plot_gradcam(
    model: nn.Module,
    tensor: torch.Tensor,
    raw_image: np.ndarray,
    target_layer,
    label: Optional[str] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate and display Grad-CAM heatmap for model explainability.

    Grad-CAM highlights the regions the model focuses on — critical for
    medical AI transparency and trust.
    """
    if not GRADCAM_AVAILABLE:
        print("grad-cam not installed. Run: pip install grad-cam")
        return raw_image

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=tensor)[0]  # (H, W)

    # Normalize raw image for overlay
    img_rgb = cv2.resize(raw_image, (224, 224))
    img_float = img_rgb.astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb, cmap="gray")
    axes[0].set_title("Original X-Ray", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(visualization)
    axes[2].set_title(f"Overlay{f' — {label}' if label else ''}", fontsize=12)
    axes[2].axis("off")

    plt.suptitle("Model Explainability: Grad-CAM", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return visualization


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    if classes is None:
        classes = ["NORMAL", "PNEUMONIA"]

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Clinical annotations
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fig.text(0.5, -0.05, f"Sensitivity={sensitivity:.3f}  |  Specificity={specificity:.3f}",
             ha="center", fontsize=11, color="navy")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: dict,
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot ROC curves for multiple models on the same axis.

    Args:
        y_probs: dict of {"model_name": probabilities_array}
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    for (name, probs), color in zip(y_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_dataset_distribution(
    class_counts: dict,
    split_name: str = "Dataset",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart showing class distribution with imbalance annotation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ["#66BB6A", "#EF5350"]

    bars = ax1.bar(classes, counts, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{count:,}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax1.set_title(f"{split_name} — Class Distribution", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Number of Images", fontsize=11)
    ax1.set_ylim(0, max(counts) * 1.15)
    ax1.grid(axis="y", alpha=0.3)

    ax2.pie(counts, labels=classes, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 12})
    ax2.set_title(f"{split_name} — Class Proportion", fontsize=13, fontweight="bold")

    imbalance_ratio = max(counts) / min(counts)
    fig.suptitle(
        f"Total: {sum(counts):,} images  |  Imbalance ratio: {imbalance_ratio:.2f}:1",
        fontsize=11, color="gray",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(
    history: dict,
    best_epoch: Optional[int] = None,
    stage_transition_epoch: Optional[int] = None,
    target_sensitivity: float = 0.95,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the training dynamics produced by ``Trainer.history``.

    Renders a 6-panel dashboard (loss, AUC-ROC, sensitivity, specificity,
    F1-score, learning rate) with train/val curves. Clinical context is
    overlaid: the best-AUC epoch, the Stage 1 -> Stage 2 (backbone unfreeze)
    transition, and the clinical sensitivity target.

    Args:
        history: dict with keys ``epoch``, ``lr``, ``train``, ``val`` where
            ``train``/``val`` are lists of per-epoch metric dicts
            (as built by :class:`src.training.trainer.Trainer`).
        best_epoch: epoch (1-indexed) of the saved best model. Highlighted
            with a vertical marker on every panel.
        stage_transition_epoch: epoch where the backbone was unfrozen.
        target_sensitivity: clinical sensitivity floor (drawn on the
            sensitivity panel). Defaults to 0.95.
        save_path: optional path to write the figure (PNG, 150 dpi).
    """
    epochs = history.get("epoch") or list(range(1, len(history["val"]) + 1))
    train_metrics = history["train"]
    val_metrics = history["val"]
    lrs = history.get("lr")

    panels = [
        ("loss",        "Loss",                 False),
        ("auc_roc",     "AUC-ROC",              True),
        ("sensitivity", "Sensitivity (Recall)", True),
        ("specificity", "Specificity",          True),
        ("f1_score",    "F1-Score",             True),
    ]

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    train_color, val_color = "#1976D2", "#E53935"

    def _annotate_markers(ax):
        if stage_transition_epoch is not None and stage_transition_epoch <= max(epochs):
            ax.axvline(stage_transition_epoch, color="#7E57C2", ls="--", lw=1.3, alpha=0.8)
        if best_epoch is not None:
            ax.axvline(best_epoch, color="#43A047", ls=":", lw=1.6, alpha=0.9)

    for ax, (key, title, is_score) in zip(axes, panels):
        train_vals = [m.get(key, np.nan) for m in train_metrics]
        val_vals   = [m.get(key, np.nan) for m in val_metrics]

        ax.plot(epochs, train_vals, "-o", color=train_color, markersize=4,
                lw=2, label="Train")
        ax.plot(epochs, val_vals, "-o", color=val_color, markersize=4,
                lw=2, label="Val")

        _annotate_markers(ax)

        # Highlight the best-epoch value on the validation curve
        if best_epoch is not None and best_epoch in epochs:
            idx = epochs.index(best_epoch)
            best_val = val_vals[idx]
            ax.scatter([best_epoch], [best_val], s=120, facecolors="none",
                       edgecolors="#43A047", linewidths=2, zorder=5)
            ax.annotate(f"{best_val:.3f}", (best_epoch, best_val),
                        textcoords="offset points", xytext=(6, 8),
                        fontsize=9, color="#2E7D32", fontweight="bold")

        if key == "sensitivity":
            ax.axhline(target_sensitivity, color="#FB8C00", ls="-.", lw=1.3,
                       alpha=0.8, label=f"Target ≥ {target_sensitivity:.2f}")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)
        if is_score:
            ax.set_ylim(0, 1.02)

    # Last panel: learning rate schedule
    ax_lr = axes[5]
    if lrs:
        ax_lr.plot(epochs, lrs, "-o", color="#00897B", markersize=4, lw=2)
        ax_lr.set_yscale("log")
        _annotate_markers(ax_lr)
        ax_lr.set_title("Learning Rate", fontsize=12, fontweight="bold")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("LR (log scale)")
        ax_lr.grid(alpha=0.3, which="both")
    else:
        ax_lr.axis("off")

    # Shared legend entries for the vertical markers
    legend_handles = []
    if stage_transition_epoch is not None:
        legend_handles.append(mpatches.Patch(color="#7E57C2",
                              label=f"Stage 2 start (epoch {stage_transition_epoch})"))
    if best_epoch is not None:
        legend_handles.append(mpatches.Patch(color="#43A047",
                              label=f"Best model (epoch {best_epoch})"))
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper center", ncol=2,
                   fontsize=10, bbox_to_anchor=(0.5, 0.965), frameon=True)

    plt.suptitle("Training History — DenseNet-121", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
