# Chest X-Ray Pneumonia Detection
## Hybrid Neuro-Symbolic System

> **Binary classification** of chest radiographs as `NORMAL` or `PNEUMONIA`  
> using a novel combination of deep learning (DenseNet-121) and a rule-based expert system.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Notebooks](#notebooks)
7. [Expert System Rules](#expert-system-rules)
8. [Hybrid Fusion Strategies](#hybrid-fusion-strategies)
9. [Metrics & Results](#metrics--results)
10. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
11. [Good Practices for Medical AI](#good-practices-for-medical-ai)
12. [References](#references)

---

## Project Overview

This project implements a **hybrid neuro-symbolic system** for automatic pneumonia detection in chest X-rays. It combines:

- **Neural Network (DenseNet-121)**: Learns complex visual patterns from data, following the landmark CheXNet paper.
- **Expert System**: Encodes radiological knowledge as interpretable rules (opacity detection, GLCM texture analysis, density distribution).
- **Meta-Learner Fusion**: A calibrated logistic regression that learns the optimal combination of both systems from data.

### Why Hybrid?

| Approach | Strength | Weakness |
|----------|----------|----------|
| Neural Network | High accuracy, learns subtle patterns | Black-box, no explicit reasoning |
| Expert System | Interpretable, domain-knowledge encoded | Limited to coded rules, brittle |
| **Hybrid** | **Best of both: accuracy + interpretability** | More complex to implement |

In medical AI, interpretability is critical — clinicians need to understand *why* a system made a decision. Grad-CAM visualizations (notebook 06) show the neural network's focus regions, while the expert system provides explicit rule-based justification.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT: Chest X-Ray                     │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
┌─────────────────┐       ┌──────────────────────┐
│   DenseNet-121  │       │   Expert System       │
│   (CheXNet)     │       │                       │
│                 │       │  Rule 1: Opacity       │
│  • ImageNet     │       │  Rule 2: GLCM Texture  │
│    pretrained   │       │  Rule 3: Density Dist  │
│  • Stage-wise   │       │  Rule 4: Consolidation │
│    fine-tuning  │       │                        │
│  • Grad-CAM     │       │  + Lung Segmentation   │
│    explainability│      │  + CLAHE Preprocessing │
└────────┬────────┘       └──────────┬─────────────┘
         │                           │
         │  P(pneumonia)             │  expert_score
         │  nn_prob                  │  + 4 sub-scores
         └──────────────┬────────────┘
                        │
               ┌────────▼────────┐
               │   Meta-Learner  │
               │  (Calibrated    │
               │   Logistic      │
               │   Regression)   │
               └────────┬────────┘
                        │
               ┌────────▼────────┐
               │  Final Output   │
               │  • Prediction   │
               │  • Confidence   │
               │  • Report       │
               └─────────────────┘
```

---

## Dataset

**Chest X-Ray Images (Pneumonia)** — [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

| Split | NORMAL | PNEUMONIA | Total | Imbalance |
|-------|--------|-----------|-------|-----------|
| Train | 1,341  | 3,875     | 5,216 → used: 10,432 | 2.89:1 |
| Val   | 8      | 8         | 16    | 1:1 |
| Test  | 234    | 390       | 624 → 1,248 | 1.67:1 |

> **Note on imbalance**: The training set has ~3× more PNEUMONIA images.  
> We address this with `WeightedRandomSampler` + inverse-frequency class weights in the loss.

---

## Project Structure

```
chest2/
├── chest_xray/                     # Dataset (train/val/test splits)
│   ├── train/
│   │   ├── NORMAL/                 # 1,341 images
│   │   └── PNEUMONIA/              # 3,875 images
│   ├── val/
│   └── test/
│
├── notebooks/                      # Jupyter notebooks (run in order)
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing_augmentation.ipynb
│   ├── 03_expert_system.ipynb      # Standalone expert system evaluation
│   ├── 04_neural_network_training.ipynb   # DenseNet-121 training
│   ├── 05_hybrid_system.ipynb      # Hybrid fusion + comparison
│   ├── 06_explainability_gradcam.ipynb    # Grad-CAM + TTA
│   └── 07_mlflow_model_registry.ipynb     # MLflow model management
│
├── src/                            # Source code (importable modules)
│   ├── data/
│   │   └── dataset.py             # Dataset class + DataLoader builder
│   ├── models/
│   │   ├── neural_net.py          # DenseNet-121, EfficientNet-B3, ResNet-50
│   │   ├── expert_system.py       # Rule-based radiological expert system
│   │   └── hybrid.py             # Hybrid fusion (3 strategies)
│   ├── training/
│   │   └── trainer.py            # Training loop + MLflow integration
│   └── utils/
│       ├── metrics.py            # Clinical metrics (sensitivity, specificity, AUC)
│       └── visualization.py      # Grad-CAM, ROC, confusion matrix plots
│
├── configs/
│   └── config.yaml               # All hyperparameters and paths
│
├── mlruns/                        # MLflow experiment tracking (auto-generated)
├── models_saved/                  # Model checkpoints (auto-generated)
├── reports/                       # Generated plots and reports
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── setup_env.sh                   # Environment setup script
└── README.md
```

---

## Quick Start

### 1. Create and Activate Virtual Environment

```bash
cd /path/to/chest2

# Option A: Use the setup script (recommended)
chmod +x setup_env.sh
./setup_env.sh

# Option B: Manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name=chest-xray --display-name="Python (chest-xray)"
```

### 2. Launch Jupyter

```bash
source .venv/bin/activate
jupyter notebook notebooks/
```

### 3. Run Notebooks in Order

| Notebook | Description | Expected Duration |
|----------|-------------|-------------------|
| `01_eda.ipynb` | Data exploration and visualization | 5 min |
| `02_preprocessing_augmentation.ipynb` | Validate preprocessing pipeline | 3 min |
| `03_expert_system.ipynb` | Expert system evaluation | 20 min |
| `04_neural_network_training.ipynb` | **Train DenseNet-121** | 2-4h (GPU) / 8-12h (CPU) |
| `05_hybrid_system.ipynb` | Train meta-learner + evaluate hybrid | 30 min |
| `06_explainability_gradcam.ipynb` | Grad-CAM visualizations | 10 min |
| `07_mlflow_model_registry.ipynb` | Model registry and versioning | 5 min |

### 4. View MLflow Dashboard

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri ./mlruns --port 5000
# Open http://localhost:5000
```

---

## Notebooks

### 01 — Exploratory Data Analysis
- Class distribution across splits
- Image property analysis (size, aspect ratio, pixel statistics)
- Visual gallery of NORMAL vs PNEUMONIA X-rays
- Mean pixel histogram comparison
- **Key finding**: Pneumonia images show elevated mean intensity (opacity) and higher standard deviation (heterogeneous texture) — directly motivates expert system rules.

### 02 — Preprocessing & Augmentation
- CLAHE (Contrast Limited Adaptive Histogram Equalization) effect visualization
- Full augmentation pipeline: HFlip, Rotation, Brightness, Grid Distortion, CoarseDropout
- WeightedRandomSampler validation (confirms ~50/50 class distribution per batch)

### 03 — Expert System
- Rule-by-rule analysis with score visualization
- Lung segmentation visualization
- Standalone expert system performance on test set
- Feature distribution comparison by class

### 04 — Neural Network Training
- Stage-wise fine-tuning (freeze backbone → full network)
- Training curves (loss, AUC, sensitivity, specificity)
- Test set evaluation with optimal threshold selection
- Comparison at threshold=0.5 vs sensitivity-maximizing threshold

### 05 — Hybrid System
- Meta-learner training on validation data
- Comparison of all 3 fusion strategies
- Single patient prediction demo with full diagnostic report
- System performance comparison table

### 06 — Explainability
- Grad-CAM++ gallery (NORMAL, bacterial pneumonia, viral pneumonia)
- Test-Time Augmentation uncertainty estimation
- Expert system rules + Grad-CAM side-by-side overlay

### 07 — MLflow Model Registry
- List all experiment runs
- Identify and register best model
- Load model from registry for inference
- Export model card

---

## Expert System Rules

The expert system encodes **4 radiological rules** based on ACR/RSNA diagnostic criteria:

### Rule 1: Opacity Detection
```
Pneumonia causes air-space opacification: affected lung regions appear 
brighter than normal air-filled parenchyma.

Score = clip(mean_lung_intensity / 0.6, 0, 1)
      + homogeneity_penalty (high mean + low std → consolidation)
```

### Rule 2: GLCM Texture Heterogeneity
```
Pneumonic consolidation alters parenchymal texture.
Uses Gray-Level Co-occurrence Matrix (GLCM) features:
  - Contrast:    high in pneumonia
  - Homogeneity: low in pneumonia  
  - Energy:      low in pneumonia
  - Correlation: low in pneumonia

Distances: [1, 3, 5] pixels | Angles: 0°, 45°, 90°, 135°
```

### Rule 3: Density Distribution
```
Pneumonia often shows basal predominance (lower lobes more affected).
Measures upper/lower lung zone asymmetry.

asymmetry = |upper_zone_mean - lower_zone_mean|
basal_predominance = max(0, lower_mean - upper_mean)
```

### Rule 4: Consolidation Pattern
```
Pixels above the 60th percentile of lung intensity = dense regions.
High consolidation_ratio + high mean intensity → pneumonia.

consolidation_ratio = dense_pixels / total_lung_pixels
```

**Final expert score** = weighted combination:
```
score = 0.40 × opacity + 0.35 × texture + 0.25 × (0.5 × density + 0.5 × consolidation)
```

---

## Hybrid Fusion Strategies

### Strategy 1: Weighted Average
```python
final_prob = 0.65 × nn_prob + 0.35 × expert_score
```
Simple and interpretable. Weights are fixed based on expected performance.

### Strategy 2: Dempster-Shafer Evidence Theory
```
Treats NN and expert as independent evidence sources.
Handles disagreement (conflict) explicitly with reduced confidence.
High-conflict cases trigger uncertainty flag.
```
Principled probabilistic combination with uncertainty quantification.

### Strategy 3: Meta-Learner (Recommended)
```
Feature vector = [nn_prob_normal, nn_prob_pneumonia,
                  opacity_score, texture_score, density_score,
                  consolidation_score, expert_final_score]

Trained: CalibratedClassifierCV(LogisticRegressionCV)
```
Learns optimal weights from data. Isotonic calibration ensures well-calibrated probabilities.

---

## Metrics & Results

### Clinical Metrics (Primary)

| Metric | Description | Why Important |
|--------|-------------|---------------|
| **Sensitivity** | TP / (TP + FN) | Minimize missed pneumonia cases |
| **Specificity** | TN / (TN + FP) | Minimize unnecessary treatment |
| **AUC-ROC** | Area under ROC curve | Overall discrimination ability |
| **Youden's J** | Sensitivity + Specificity - 1 | Optimal threshold selection |
| NPV | TN / (TN + FN) | Confidence in negative result |

> **Target**: Sensitivity ≥ 0.95 (missing pneumonia is more dangerous than a false alarm)

### Expected Performance (after training)

| System | AUC-ROC | Sensitivity | Specificity |
|--------|---------|-------------|-------------|
| Expert System only | ~0.72 | ~0.75 | ~0.68 |
| DenseNet-121 only | ~0.96 | ~0.93 | ~0.90 |
| **Hybrid (Meta-Learner)** | **~0.97** | **~0.95** | **~0.92** |

---

## MLflow Experiment Tracking

Every training run automatically logs:

**Parameters**: backbone, epochs, learning_rate, batch_size, augmentation strategy, class weighting

**Metrics** (per epoch): train_loss, val_loss, train_auc, val_auc, train_sensitivity, val_sensitivity, val_specificity

**Artifacts**: model checkpoint, confusion matrix, ROC curve, Grad-CAM samples

**Model Registry**: Best models registered as `ChestXRayNet` with version tracking

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000
```

---

## Good Practices for Medical AI

This project follows best practices for responsible medical AI:

1. **Interpretability**: Grad-CAM + Expert System rules provide dual-channel explanation
2. **Calibration**: Probability outputs are calibrated (CalibratedClassifierCV)
3. **Clinical Threshold**: Threshold tuned for ≥0.95 sensitivity (not just accuracy)
4. **Uncertainty Quantification**: TTA-based uncertainty + Dempster-Shafer conflict detection
5. **Imbalance Handling**: WeightedRandomSampler + class-weighted loss
6. **No Data Leakage**: Test set never used during training or hyperparameter tuning
7. **Reproducibility**: Fixed seeds, version-controlled code, MLflow-tracked experiments
8. **Preprocessing**: CLAHE following medical imaging literature standards
9. **Augmentation**: Clinically realistic augmentations only (no color jitter in grayscale)
10. **Documentation**: Model card exported with each experiment

> ⚠️ **Disclaimer**: This system is for research purposes only. It is not validated for clinical use. All medical decisions must be made by qualified healthcare professionals.

---

## References

1. **CheXNet** — Rajpurkar et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*. arXiv:1711.05225
2. **Grad-CAM** — Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV 2017.
3. **DenseNet** — Huang et al. (2017). *Densely Connected Convolutional Networks*. CVPR 2017.
4. **Dempster-Shafer** — Shafer, G. (1976). *A Mathematical Theory of Evidence*. Princeton University Press.
5. **GLCM Texture** — Haralick et al. (1973). *Textural Features for Image Classification*. IEEE Transactions on Systems, Man, and Cybernetics.
6. **Albumentations** — Buslaev et al. (2020). *Albumentations: Fast and Flexible Image Augmentations*. Information 2020.
7. **MLflow** — Zaharia et al. (2018). *Accelerating the Machine Learning Lifecycle with MLflow*. VLDB 2018.

---

## License

This project is for educational and research purposes.  
Dataset: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) — Paul Mooney (CC BY 4.0)
