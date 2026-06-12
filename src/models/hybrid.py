"""
Hybrid Neuro-Symbolic System.

Combines the neural network's learned representation with the expert
system's interpretable radiological rules using a meta-learner approach.

Architecture:
  ┌─────────────────┐     ┌──────────────────┐
  │  DenseNet-121   │     │  Expert System   │
  │  (deep feats)   │     │  (symbolic rules)│
  └────────┬────────┘     └────────┬─────────┘
           │  P(pneumonia)         │  expert_score
           │  nn_prob              │  + 4 sub-scores
           └──────────┬────────────┘
                      │
              ┌───────▼────────┐
              │  Meta-Learner  │  (Logistic Regression or
              │  (calibrated)  │   Platt scaling)
              └───────┬────────┘
                      │
              Final Prediction
             + Confidence + Report
"""
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from .neural_net import ChestXRayNet
from .expert_system import ChestExpertSystem, ExpertFindings


class DempsterShaferCombiner:
    """
    Dempster-Shafer evidence combination for uncertainty-aware fusion.

    Treats NN and expert as independent evidence sources.
    Handles cases where they disagree (high conflict) with reduced confidence.
    """

    def combine(self, nn_prob: float, expert_score: float) -> Tuple[float, float]:
        """
        Returns (combined_belief, uncertainty).

        Based on D-S basic belief assignments:
          - m1: evidence from neural network
          - m2: evidence from expert system
        """
        m1_pneumonia = nn_prob
        m1_normal    = 1 - nn_prob
        m2_pneumonia = expert_score
        m2_normal    = 1 - expert_score

        # Orthogonal sum (Dempster's rule)
        k = m1_pneumonia * m2_normal + m1_normal * m2_pneumonia  # conflict mass

        if k >= 0.99:  # high conflict → return average
            return (nn_prob + expert_score) / 2, 0.5

        belief_pneumonia = (m1_pneumonia * m2_pneumonia) / (1 - k)
        belief_normal    = (m1_normal * m2_normal) / (1 - k)

        # Plausibility = belief + uncommitted mass
        total = belief_pneumonia + belief_normal
        if total == 0:
            return 0.5, 0.5

        combined = belief_pneumonia / total
        uncertainty = 1 - abs(belief_pneumonia - belief_normal) / total
        return float(combined), float(np.clip(uncertainty, 0, 1))


class HybridNeuroSymbolicSystem:
    """
    Production-grade hybrid system combining NN + Expert System.

    Three combination strategies:
      1. "weighted"        — fixed weighted average
      2. "dempster_shafer" — evidence-theoretic combination
      3. "meta_learner"    — trained logistic regression on [nn_prob, expert_features]
    """

    def __init__(
        self,
        neural_net: ChestXRayNet,
        expert_system: ChestExpertSystem,
        combination_method: str = "meta_learner",
        nn_weight: float = 0.65,
        expert_weight: float = 0.35,
        device: str = "cpu",
    ):
        self.neural_net = neural_net
        self.expert_system = expert_system
        self.method = combination_method
        self.nn_weight = nn_weight
        self.expert_weight = expert_weight
        self.device = device

        self.meta_learner: Optional[CalibratedClassifierCV] = None
        self.scaler: Optional[StandardScaler] = None
        self._ds_combiner = DempsterShaferCombiner()

        self.neural_net.eval()

    # ─────────────────────────────────────────────────────────────────────
    # Neural Network inference
    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _nn_predict(self, tensor: torch.Tensor) -> np.ndarray:
        """Returns softmax probabilities for a batch tensor."""
        tensor = tensor.to(self.device)
        logits = self.neural_net(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs  # shape (N, 2)

    # ─────────────────────────────────────────────────────────────────────
    # Combination methods
    # ─────────────────────────────────────────────────────────────────────
    def _combine_weighted(
        self, nn_probs: np.ndarray, expert_scores: np.ndarray
    ) -> np.ndarray:
        pneumonia_prob = (
            self.nn_weight * nn_probs[:, 1] +
            self.expert_weight * expert_scores
        )
        return np.clip(pneumonia_prob, 0, 1)

    def _combine_dempster_shafer(
        self, nn_probs: np.ndarray, expert_scores: np.ndarray
    ) -> np.ndarray:
        results = []
        for nn_p, exp_s in zip(nn_probs[:, 1], expert_scores):
            combined, _ = self._ds_combiner.combine(float(nn_p), float(exp_s))
            results.append(combined)
        return np.array(results)

    def _combine_meta_learner(self, feature_matrix: np.ndarray) -> np.ndarray:
        if self.meta_learner is None:
            raise RuntimeError("Meta-learner not trained. Call train_meta_learner() first.")
        X = self.scaler.transform(feature_matrix)
        return self.meta_learner.predict_proba(X)[:, 1]

    # ─────────────────────────────────────────────────────────────────────
    # Meta-learner training
    # ─────────────────────────────────────────────────────────────────────
    def train_meta_learner(
        self,
        tensors: List[torch.Tensor],
        raw_images: List[np.ndarray],
        labels: List[int],
        cv_folds: int = 5,
    ) -> Dict[str, float]:
        """
        Trains meta-learner on validation / held-out data.

        Feature matrix = [nn_prob_normal, nn_prob_pneumonia,
                          opacity, texture, density, consolidation, expert_final]
        """
        print("Extracting NN predictions...")
        nn_preds = []
        for t in tensors:
            if t.dim() == 3:
                t = t.unsqueeze(0)
            probs = self._nn_predict(t)
            nn_preds.append(probs[0])
        nn_preds = np.array(nn_preds)

        print("Running expert system analysis...")
        expert_features = []
        for img in raw_images:
            findings = self.expert_system.analyze(img)
            expert_features.append(findings.feature_vector)
        expert_features = np.array(expert_features)

        feature_matrix = np.concatenate([nn_preds, expert_features], axis=1)
        labels_arr = np.array(labels)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(feature_matrix)

        base_lr = LogisticRegressionCV(
            Cs=10, cv=cv_folds, max_iter=1000,
            class_weight="balanced", random_state=42,
        )
        self.meta_learner = CalibratedClassifierCV(base_lr, cv=cv_folds, method="isotonic")
        self.meta_learner.fit(X, labels_arr)

        # Training performance
        preds = (self.meta_learner.predict_proba(X)[:, 1] >= 0.5).astype(int)
        from sklearn.metrics import accuracy_score, roc_auc_score
        acc = accuracy_score(labels_arr, preds)
        auc = roc_auc_score(labels_arr, self.meta_learner.predict_proba(X)[:, 1])
        print(f"Meta-learner trained — train accuracy={acc:.4f}, AUC={auc:.4f}")
        return {"accuracy": acc, "auc_roc": auc}

    # ─────────────────────────────────────────────────────────────────────
    # Main inference
    # ─────────────────────────────────────────────────────────────────────
    def predict(
        self,
        tensor: torch.Tensor,
        raw_image: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Full hybrid prediction for a single image.

        Args:
            tensor: preprocessed torch tensor (1×C×H×W or C×H×W)
            raw_image: original numpy array for expert system
            threshold: decision threshold (tune for sensitivity/specificity tradeoff)

        Returns:
            dict with prediction, confidence, nn_prob, expert_score, report
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        nn_probs = self._nn_predict(tensor)  # (1, 2)
        findings = self.expert_system.analyze(raw_image)

        nn_prob_pneumonia = float(nn_probs[0, 1])
        expert_score = findings.final_score

        if self.method == "weighted":
            final_prob = self._combine_weighted(nn_probs, np.array([expert_score]))[0]
            uncertainty = abs(final_prob - 0.5) * -2 + 1  # approx

        elif self.method == "dempster_shafer":
            final_prob, uncertainty = self._ds_combiner.combine(nn_prob_pneumonia, expert_score)

        elif self.method == "meta_learner":
            expert_feats = findings.feature_vector
            feature_row = np.concatenate([nn_probs[0], expert_feats]).reshape(1, -1)
            final_prob = self._combine_meta_learner(feature_row)[0]
            uncertainty = 1 - abs(final_prob - 0.5) * 2

        else:
            raise ValueError(f"Unknown method: {self.method}")

        prediction = int(final_prob >= threshold)
        label_name = "PNEUMONIA" if prediction == 1 else "NORMAL"

        # Expert override: if expert is very confident, trust it
        if findings.is_high_confidence and self.method != "meta_learner":
            expert_pred = findings.prediction
            if expert_pred != prediction and findings.confidence >= 0.90:
                prediction = expert_pred
                label_name = "PNEUMONIA" if prediction == 1 else "NORMAL"
                final_prob = findings.final_score
                uncertainty = 0.15  # high confidence override

        return {
            "prediction": prediction,
            "label": label_name,
            "final_probability": round(final_prob, 4),
            "confidence": round(1 - uncertainty, 4),
            "nn_probability": round(nn_prob_pneumonia, 4),
            "expert_score": round(expert_score, 4),
            "expert_findings": findings.findings,
            "combination_method": self.method,
        }

    def predict_batch(
        self,
        tensors: torch.Tensor,
        raw_images: List[np.ndarray],
        threshold: float = 0.5,
    ) -> List[Dict]:
        """Batch inference for evaluation."""
        results = []
        for i in range(len(raw_images)):
            t = tensors[i] if tensors.dim() == 4 else tensors[i].unsqueeze(0)
            results.append(self.predict(t, raw_images[i], threshold))
        return results

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────
    def save_meta_learner(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"meta_learner": self.meta_learner, "scaler": self.scaler}, f)

    def load_meta_learner(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.meta_learner = data["meta_learner"]
        self.scaler = data["scaler"]
