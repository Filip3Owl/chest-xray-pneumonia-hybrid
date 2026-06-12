"""
Clinical metrics for medical AI evaluation.

Standard accuracy/F1 are insufficient for medical diagnosis:
  - Sensitivity (recall for disease) = avoid false negatives
  - Specificity = avoid unnecessary treatment
  - AUC-ROC = discrimination ability
  - Youden's J = optimal operating point
"""
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    average_precision_score, roc_curve,
)


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    pos_label: int = 1,
) -> Dict[str, float]:
    """
    Computes the full suite of clinical classification metrics.

    Args:
        y_true: Ground truth labels (0=NORMAL, 1=PNEUMONIA)
        y_pred: Binary predictions
        y_prob: Predicted probabilities for positive class
        pos_label: Positive class index

    Returns:
        Dictionary of metric name → value
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0            # precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # Youden's J statistic: optimal threshold selection
    youdens_j = sensitivity + specificity - 1

    metrics = {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),  # most critical in pneumonia detection
        "specificity": float(specificity),
        "ppv":         float(ppv),
        "npv":         float(npv),
        "f1_score":    float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "youdens_j":   float(youdens_j),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    if y_prob is not None:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "youden",
    target_sensitivity: float = 0.95,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the probability threshold that optimizes a clinical criterion.

    Strategies:
      - "youden"      : maximizes sensitivity + specificity
      - "sensitivity" : ensures sensitivity >= target, maximizes specificity
      - "f1"          : maximizes F1 score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    if strategy == "youden":
        idx = np.argmax(tpr - fpr)
        best_threshold = float(thresholds[idx])

    elif strategy == "sensitivity":
        # Find lowest threshold that achieves target sensitivity
        valid = np.where(tpr >= target_sensitivity)[0]
        idx = valid[np.argmax(1 - fpr[valid])] if len(valid) > 0 else np.argmax(tpr)
        best_threshold = float(thresholds[idx])

    elif strategy == "f1":
        f1_scores = []
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            f1_scores.append(f1_score(y_true, preds, zero_division=0))
        idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[idx])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    preds = (y_prob >= best_threshold).astype(int)
    metrics = compute_clinical_metrics(y_true, preds, y_prob)
    return best_threshold, metrics


def print_clinical_report(metrics: Dict[str, float], title: str = "Clinical Metrics Report") -> None:
    """Pretty-print clinical metrics in a radiologist-readable format."""
    bar = "=" * 55
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    print(f"  {'Accuracy':<25} {metrics.get('accuracy', 0):.4f}")
    print(f"  {'AUC-ROC':<25} {metrics.get('auc_roc', 0):.4f}")
    print(f"  {'Avg Precision (AP)':<25} {metrics.get('avg_precision', 0):.4f}")
    print(f"  {'F1 Score':<25} {metrics.get('f1_score', 0):.4f}")
    youdens_label = "Youden's J"
    print(f"  {youdens_label:<25} {metrics.get('youdens_j', 0):.4f}")
    print(f"  {'-'*45}")
    print(f"  {'Sensitivity (Recall)':<25} {metrics.get('sensitivity', 0):.4f}  ← minimize FN")
    print(f"  {'Specificity':<25} {metrics.get('specificity', 0):.4f}  ← minimize FP")
    print(f"  {'PPV (Precision)':<25} {metrics.get('ppv', 0):.4f}")
    print(f"  {'NPV':<25} {metrics.get('npv', 0):.4f}")
    print(f"  {'-'*45}")
    print(f"  {'TP':<10} {metrics.get('tp', 0):>6}   {'FP':<10} {metrics.get('fp', 0):>6}")
    print(f"  {'FN':<10} {metrics.get('fn', 0):>6}   {'TN':<10} {metrics.get('tn', 0):>6}")
    print(f"{bar}\n")
