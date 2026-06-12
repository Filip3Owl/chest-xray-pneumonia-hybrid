"""
Rule-based Expert System for Chest X-Ray Analysis.

Encodes radiological knowledge as computable rules:
  - Lung opacity / consolidation detection
  - Texture heterogeneity (GLCM features)
  - Lung density distribution
  - Opacification pattern classification

Each rule returns a score in [0, 1] where higher = more likely pneumonia.
The final expert score is a weighted combination of all rules.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, opening
from skimage.measure import regionprops, label as sk_label


@dataclass
class ExpertFindings:
    """Structured report from the expert system."""
    opacity_score: float = 0.0
    texture_score: float = 0.0
    density_score: float = 0.0
    consolidation_score: float = 0.0
    final_score: float = 0.0
    prediction: int = 0               # 0=NORMAL, 1=PNEUMONIA
    confidence: float = 0.0
    findings: List[str] = field(default_factory=list)
    feature_vector: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.85


class ChestExpertSystem:
    """
    Symbolic expert system that mimics radiologist diagnostic criteria.

    Rules are based on:
    - ACR/RSNA pneumonia diagnosis guidelines
    - Opacity: air-space opacification in lung fields
    - Consolidation: replacement of air by fluid/tissue
    - Texture: heterogeneous vs homogeneous lung parenchyma
    """

    def __init__(
        self,
        opacity_threshold: float = 0.35,
        glcm_distances: List[int] = None,
        glcm_angles: List[float] = None,
        opacity_weight: float = 0.40,
        texture_weight: float = 0.35,
        density_weight: float = 0.25,
        expert_override_threshold: float = 0.85,
    ):
        self.opacity_threshold = opacity_threshold
        self.glcm_distances = glcm_distances or [1, 3, 5]
        self.glcm_angles = glcm_angles or [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.opacity_weight = opacity_weight
        self.texture_weight = texture_weight
        self.density_weight = density_weight
        self.override_threshold = expert_override_threshold

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Standardize image for rule evaluation."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # CLAHE: enhance local contrast (standard in chest X-ray processing)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Normalize to [0, 1]
        return enhanced.astype(np.float32) / 255.0

    def extract_lung_roi(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rough lung segmentation using adaptive thresholding + morphology.

        Returns (roi_image, lung_mask).
        """
        gray_uint8 = (gray * 255).astype(np.uint8)

        # Otsu threshold to separate lung field from surrounding tissue
        thresh = threshold_otsu(gray_uint8)
        binary = gray_uint8 < thresh  # lungs are darker than bones/mediastinum

        # Morphological operations to clean up lung mask
        cleaned = closing(binary, disk(5))
        cleaned = opening(cleaned, disk(3))

        # Keep only the two largest connected components (left + right lung)
        labeled = sk_label(cleaned)
        props = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)
        lung_mask = np.zeros_like(binary, dtype=bool)
        for prop in props[:2]:  # top 2 regions = lungs
            lung_mask[labeled == prop.label] = True

        # Fallback: if segmentation fails, use center crop as ROI
        if lung_mask.sum() < 0.05 * gray.size:
            h, w = gray.shape
            lung_mask[h//6:5*h//6, w//6:5*w//6] = True

        roi = gray.copy()
        roi[~lung_mask] = 0.0
        return roi, lung_mask

    # ──────────────────────────────────────────────────────────────────────
    # RULE 1 — Opacity Rule
    # Pneumonia causes air-space opacification: lung regions become brighter
    # than expected. Elevated mean intensity within lung ROI → opacity.
    # ──────────────────────────────────────────────────────────────────────
    def _opacity_rule(self, roi: np.ndarray, mask: np.ndarray) -> Tuple[float, str]:
        lung_pixels = roi[mask]
        if len(lung_pixels) == 0:
            return 0.5, "Could not evaluate opacity (mask empty)"

        mean_intensity = float(lung_pixels.mean())
        std_intensity = float(lung_pixels.std())

        # High mean + low std → consolidation (fluid filling air spaces)
        opacity_score = np.clip(mean_intensity / 0.6, 0, 1)
        homogeneity_penalty = np.clip(1 - std_intensity / 0.3, 0, 0.3)
        score = np.clip(opacity_score + homogeneity_penalty, 0, 1)

        finding = (
            f"Lung opacity: mean={mean_intensity:.3f}, std={std_intensity:.3f} → "
            f"{'ELEVATED (consolidation suspected)' if score > 0.5 else 'NORMAL range'}"
        )
        return float(score), finding

    # ──────────────────────────────────────────────────────────────────────
    # RULE 2 — Texture Heterogeneity Rule (GLCM)
    # Pneumonic consolidation alters parenchymal texture:
    # lower correlation, higher contrast, lower homogeneity.
    # ──────────────────────────────────────────────────────────────────────
    def _texture_rule(self, roi: np.ndarray, mask: np.ndarray) -> Tuple[float, str]:
        lung_pixels = roi.copy()
        lung_pixels[~mask] = 0

        # Quantize to 64 levels for GLCM (reduces noise)
        quantized = (lung_pixels * 63).astype(np.uint8)

        glcm = graycomatrix(
            quantized,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=64,
            symmetric=True,
            normed=True,
        )

        contrast     = float(graycoprops(glcm, "contrast").mean())
        homogeneity  = float(graycoprops(glcm, "homogeneity").mean())
        energy       = float(graycoprops(glcm, "energy").mean())
        correlation  = float(graycoprops(glcm, "correlation").mean())

        # Pneumonia: high contrast, low homogeneity, low energy
        # Normalize each feature to [0,1] contribution
        contrast_score    = np.clip(contrast / 500.0, 0, 1)
        homogeneity_score = np.clip(1 - homogeneity, 0, 1)
        energy_score      = np.clip(1 - energy * 10, 0, 1)
        correlation_score = np.clip(1 - (correlation + 1) / 2, 0, 1)

        score = (contrast_score + homogeneity_score + energy_score + correlation_score) / 4

        finding = (
            f"Texture — contrast={contrast:.1f}, homogeneity={homogeneity:.3f}, "
            f"energy={energy:.4f}, correlation={correlation:.3f} → "
            f"{'ABNORMAL (heterogeneous)' if score > 0.5 else 'NORMAL'}"
        )
        return float(np.clip(score, 0, 1)), finding

    # ──────────────────────────────────────────────────────────────────────
    # RULE 3 — Density Distribution Rule
    # Pneumonia creates focal/lobar density increases.
    # Asymmetry between upper and lower lung zones indicates pathology.
    # ──────────────────────────────────────────────────────────────────────
    def _density_rule(self, roi: np.ndarray, mask: np.ndarray) -> Tuple[float, str]:
        h, w = roi.shape
        # Divide lung into quadrants
        upper = roi[:h//2, :]
        lower = roi[h//2:, :]
        upper_mask = mask[:h//2, :]
        lower_mask = mask[h//2:, :]

        upper_mean = float(upper[upper_mask].mean()) if upper_mask.any() else 0
        lower_mean = float(lower[lower_mask].mean()) if lower_mask.any() else 0

        # High asymmetry → focal consolidation
        asymmetry = abs(upper_mean - lower_mean)
        overall_density = (upper_mean + lower_mean) / 2

        # Basal pneumonia is most common → lower zone higher than upper
        basal_predominance = max(0, lower_mean - upper_mean)

        score = np.clip(
            asymmetry * 3 + overall_density * 0.5 + basal_predominance * 2,
            0, 1
        )

        finding = (
            f"Density distribution — upper_zone={upper_mean:.3f}, lower_zone={lower_mean:.3f}, "
            f"asymmetry={asymmetry:.3f} → "
            f"{'FOCAL DENSITY (basal consolidation suspected)' if score > 0.5 else 'NORMAL distribution'}"
        )
        return float(score), finding

    # ──────────────────────────────────────────────────────────────────────
    # RULE 4 — Consolidation Pattern Rule
    # Uses Otsu thresholding to identify high-density focal regions
    # inconsistent with normal aeration.
    # ──────────────────────────────────────────────────────────────────────
    def _consolidation_rule(self, roi: np.ndarray, mask: np.ndarray) -> Tuple[float, str]:
        lung_pixels = roi[mask]
        if len(lung_pixels) < 100:
            return 0.5, "Insufficient lung area for consolidation analysis"

        # Pixels above 60th percentile of lung intensity = dense regions
        high_thresh = np.percentile(lung_pixels, 60)
        dense_region = (roi > high_thresh) & mask
        consolidation_ratio = dense_region.sum() / mask.sum()

        # Very high ratio (>0.35) with high intensity = consolidation
        score = np.clip(consolidation_ratio / 0.35, 0, 1) * float(roi[mask].mean() > 0.3)

        finding = (
            f"Consolidation — dense_ratio={consolidation_ratio:.3f}, "
            f"mean_lung_intensity={float(roi[mask].mean()):.3f} → "
            f"{'CONSOLIDATION PATTERN' if score > 0.5 else 'NORMAL aeration'}"
        )
        return float(np.clip(score, 0, 1)), finding

    def analyze(self, image: np.ndarray) -> ExpertFindings:
        """
        Run full expert system analysis on a single image.

        Args:
            image: RGB or grayscale numpy array (H×W or H×W×3), uint8 [0,255]

        Returns:
            ExpertFindings with per-rule scores and final diagnosis
        """
        gray = self.preprocess(image)
        roi, mask = self.extract_lung_roi(gray)

        opacity_score,       f1 = self._opacity_rule(roi, mask)
        texture_score,       f2 = self._texture_rule(roi, mask)
        density_score,       f3 = self._density_rule(roi, mask)
        consolidation_score, f4 = self._consolidation_rule(roi, mask)

        final_score = (
            self.opacity_weight  * opacity_score +
            self.texture_weight  * texture_score +
            self.density_weight  * (density_score * 0.5 + consolidation_score * 0.5)
        )
        final_score = float(np.clip(final_score, 0, 1))

        prediction = int(final_score >= 0.5)
        confidence = abs(final_score - 0.5) * 2  # distance from decision boundary

        feature_vector = np.array([
            opacity_score, texture_score, density_score,
            consolidation_score, final_score,
        ], dtype=np.float32)

        return ExpertFindings(
            opacity_score=opacity_score,
            texture_score=texture_score,
            density_score=density_score,
            consolidation_score=consolidation_score,
            final_score=final_score,
            prediction=prediction,
            confidence=confidence,
            findings=[f1, f2, f3, f4],
            feature_vector=feature_vector,
        )

    def analyze_batch(self, images: List[np.ndarray]) -> List[ExpertFindings]:
        return [self.analyze(img) for img in images]

    def get_feature_names(self) -> List[str]:
        return [
            "opacity_score", "texture_score", "density_score",
            "consolidation_score", "final_expert_score",
        ]
