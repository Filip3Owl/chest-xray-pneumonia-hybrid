# CLAUDE.md — Chest X-Ray Pneumonia Detection

Guia de contexto para o agente Claude Code trabalhar neste projeto.

---

## Visão Geral do Projeto

Sistema híbrido neuro-simbólico para detecção de pneumonia em radiografias torácicas.  
Combina rede neural profunda (DenseNet-121) com sistema especialista baseado em regras radiológicas.

**Repositório:** https://github.com/Filip3Owl/chest-xray-pneumonia-hybrid  
**Dataset:** Chest X-Ray Images (Pneumonia) — Kaggle (local em `chest_xray/`)

---

## Ambiente

- **Python:** 3.11 (obrigatório — PyTorch não suporta Python 3.14)
- **Virtualenv:** `.venv/` — sempre ativar antes de qualquer comando
- **Ativar:** `source .venv/bin/activate`
- **Instalar dependências:** `pip install -r requirements.txt`
- **Instalar projeto:** `pip install -e .`
- **Variável necessária:** `ALBUMENTATIONS_DISABLE_VERSION_CHECK=1` (evita warning de SSL no macOS)

### Restrições de dependências
- `numpy<2.0` — NumPy 2.x quebra o PyTorch instalado
- `opencv-python-headless>=4.8,<4.11` — versões mais novas exigem NumPy>=2

---

## Dataset

```
chest_xray/
├── train/
│   ├── NORMAL/      1.341 imagens
│   └── PNEUMONIA/   3.875 imagens  (imbalance 2.89:1)
├── val/
│   ├── NORMAL/      8 imagens
│   └── PNEUMONIA/   8 imagens
└── test/
    ├── NORMAL/      234 imagens
    └── PNEUMONIA/   390 imagens
```

- Imagens em `.jpeg`, dimensões variáveis (até 2916×2583)
- Redimensionadas para 224×224 no pipeline
- Subtipos de pneumonia: bacterial (`*bacteria*`) e viral (`*virus*`)
- O dataset **não é versionado no git** (listado no `.gitignore`)

---

## Arquitetura do Sistema

### 1. Rede Neural — `src/models/neural_net.py`
- **Backbone:** DenseNet-121 (padrão CheXNet — Rajpurkar et al., 2017)
- Backbones alternativos disponíveis: `efficientnet_b3`, `resnet50`
- **Fine-tuning em 2 estágios:**
  - Stage 1 (epochs 1-3): backbone congelado, treina só o classificador
  - Stage 2 (epoch 4+): rede completa com LR reduzido 10×
- Output: 2 classes via softmax (NORMAL=0, PNEUMONIA=1)
- Dropout: 0.5 no classificador

### 2. Sistema Especialista — `src/models/expert_system.py`
Codifica conhecimento radiológico como regras computáveis:

| Regra | Descrição | Peso |
|-------|-----------|------|
| **Opacity** | Intensidade média do campo pulmonar — opacificação → consolidação | 0.40 |
| **Texture (GLCM)** | Matriz de co-ocorrência de níveis de cinza — heterogeneidade da textura | 0.35 |
| **Density Distribution** | Predominância basal — assimetria entre zonas superior/inferior | 0.125 |
| **Consolidation Pattern** | Regiões focais de alta densidade (>60° percentil) | 0.125 |

Pipeline do especialista:
1. Pré-processamento CLAHE
2. Segmentação do pulmão (Otsu + morfologia)
3. Avaliação das 4 regras no ROI pulmonar
4. Score final em [0, 1] — threshold 0.5

### 3. Sistema Híbrido — `src/models/hybrid.py`
Três estratégias de fusão:

| Estratégia | Descrição |
|-----------|-----------|
| `weighted` | Média ponderada fixa: 0.65×NN + 0.35×Expert |
| `dempster_shafer` | Teoria de evidências — quantifica conflito entre as fontes |
| `meta_learner` | **Recomendado** — Regressão logística calibrada treinada em [nn_probs + expert_features] |

**Feature vector do meta-learner:**
`[nn_prob_normal, nn_prob_pneumonia, opacity, texture, density, consolidation, expert_final]`

---

## Pipeline de Dados — `src/data/dataset.py`

- **Classe:** `ChestXRayDataset` — carrega imagens como RGB, aplica transforms Albumentations
- **Augmentação treino:** HFlip, Rotation±15°, BrightnessContrast, GaussianBlur, GridDistortion, CoarseDropout, CLAHE
- **Inferência:** apenas Resize + CLAHE + Normalize (ImageNet mean/std)
- **TTA:** 5 transforms para estimativa de incerteza em inferência
- **Balanceamento:** `WeightedRandomSampler` + `CrossEntropyLoss(weight=class_weights)`
- **Builder:** `build_dataloaders(data_root, image_size, batch_size, num_workers)`

---

## Treinamento — `src/training/trainer.py`

- **Otimizador:** AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler:** CosineAnnealingLR (ou ReduceLROnPlateau)
- **AMP:** Mixed Precision automático (só em CUDA)
- **Early Stopping:** patience=7 epochs por Val AUC-ROC
- **Gradient Clipping:** max_norm=1.0
- **Checkpoint:** salvo em `models_saved/best_model.pth` (melhor Val AUC)
- **MLflow:** todo experimento logado automaticamente

---

## Métricas — `src/utils/metrics.py`

Métricas clínicas são prioritárias sobre métricas padrão de ML:

| Métrica | Prioridade | Motivo |
|---------|-----------|--------|
| **Sensitivity** | ★★★ Crítica | Minimizar falsos negativos (pneumonia não detectada) |
| **AUC-ROC** | ★★★ Primária | Discriminação geral |
| **Specificity** | ★★ Alta | Evitar tratamento desnecessário |
| Youden's J | ★★ Alta | Seleção de threshold ótimo |
| F1, Accuracy | ★ Secundária | Referência geral |

**Target clínico:** Sensitivity ≥ 0.95  
**Função:** `find_optimal_threshold(y_true, y_prob, strategy='sensitivity', target_sensitivity=0.95)`

---

## MLflow

- **URI:** `./mlruns`
- **Experimento:** `chest-xray-pneumonia`
- **Runs nomeados:** `expert_system_baseline`, `densenet121_baseline`, `hybrid_systems_comparison`
- **Lançar UI:** `mlflow ui --backend-store-uri ./mlruns --port 5000`
- **Modelo registrado:** `ChestXRayNet` no Model Registry

---

## Notebooks (executar em ordem)

| # | Arquivo | Conteúdo |
|---|---------|----------|
| 01 | `01_eda.ipynb` | EDA — distribuição, propriedades das imagens, histogramas |
| 02 | `02_preprocessing_augmentation.ipynb` | CLAHE, pipeline de augmentação, validação do sampler |
| 03 | `03_expert_system.ipynb` | Avaliação standalone do sistema especialista |
| 04 | `04_neural_network_training.ipynb` | Treinamento DenseNet-121 + avaliação no test set |
| 05 | `05_hybrid_system.ipynb` | Treinamento meta-learner + comparação dos 3 sistemas |
| 06 | `06_explainability_gradcam.ipynb` | Grad-CAM++, TTA, overlay especialista + NN |
| 07 | `07_mlflow_model_registry.ipynb` | Gestão de versões e model registry |

---

## Estrutura de Arquivos

```
chest2/
├── chest_xray/          # Dataset (não versionado)
├── notebooks/           # 7 notebooks Jupyter
├── src/
│   ├── data/
│   │   └── dataset.py   # ChestXRayDataset, build_dataloaders, transforms
│   ├── models/
│   │   ├── neural_net.py    # ChestXRayNet, build_model
│   │   ├── expert_system.py # ChestExpertSystem, ExpertFindings
│   │   └── hybrid.py        # HybridNeuroSymbolicSystem, DempsterShaferCombiner
│   ├── training/
│   │   └── trainer.py   # Trainer, EarlyStopping
│   └── utils/
│       ├── metrics.py        # compute_clinical_metrics, find_optimal_threshold
│       └── visualization.py  # plot_gradcam, plot_confusion_matrix, plot_roc_curve
├── configs/
│   └── config.yaml      # Todos os hiperparâmetros e caminhos
├── mlruns/              # Artefatos MLflow (não versionado)
├── models_saved/        # Checkpoints (não versionados)
├── reports/             # Plots gerados pelos notebooks
├── .venv/               # Virtualenv Python 3.11 (não versionado)
├── requirements.txt
├── setup.py
├── setup_env.sh         # Script de setup do ambiente
└── README.md
```

---

## Comandos Úteis

```bash
# Ativar ambiente
source .venv/bin/activate

# Jupyter
jupyter notebook notebooks/

# MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000

# Rodar smoke tests
ALBUMENTATIONS_DISABLE_VERSION_CHECK=1 python -c "
import sys; sys.path.insert(0, '.')
from src.models.neural_net import ChestXRayNet
from src.models.expert_system import ChestExpertSystem
from src.utils.metrics import compute_clinical_metrics
print('OK')
"

# Git
git add . && git commit -m "mensagem" && git push
```

---

## Boas Práticas Seguidas

1. **Interpretabilidade dupla** — Grad-CAM (NN) + regras explícitas (especialista)
2. **Calibração** — `CalibratedClassifierCV` com isotonic regression
3. **Threshold clínico** — otimizado para Sensitivity ≥ 0.95, não accuracy
4. **Incerteza** — TTA (5 transforms) + Dempster-Shafer conflict detection
5. **Sem data leakage** — test set nunca usado em treino ou tuning
6. **Reprodutibilidade** — seeds fixos (42), MLflow, código versionado
7. **Augmentações realistas** — apenas transformações clinicamente plausíveis
8. **Ambiente isolado** — virtualenv Python 3.11

> ⚠️ Este sistema é para fins de pesquisa. Não validado para uso clínico.
