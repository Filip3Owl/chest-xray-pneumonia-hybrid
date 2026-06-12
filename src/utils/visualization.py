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
    train_metrics: List[dict],
    val_metrics: List[dict],
    save_path: Optional[str] = None,
) -> None:
    """Plot loss, AUC, sensitivity, specificity across epochs."""
    epochs = range(1, len(train_metrics) + 1)
    metric_keys = ["loss", "auc_roc", "sensitivity", "specificity"]
    titles = ["Loss", "AUC-ROC", "Sensitivity (Recall)", "Specificity"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, key, title in zip(axes, metric_keys, titles):
        train_vals = [m.get(key, 0) for m in train_metrics]
        val_vals   = [m.get(key, 0) for m in val_metrics]
        ax.plot(epochs, train_vals, "b-o", markersize=3, label="Train")
        ax.plot(epochs, val_vals,   "r-o", markersize=3, label="Val")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
        if key != "loss":
            ax.set_ylim(0, 1.05)

    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
