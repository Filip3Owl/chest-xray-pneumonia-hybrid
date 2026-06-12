"""
Neural network backbone for chest X-ray classification.

Uses DenseNet-121 following CheXNet (Rajpurkar et al., 2017) — the
landmark paper showing radiologist-level pneumonia detection.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class ChestXRayNet(nn.Module):
    """
    Fine-tuned DenseNet-121 for binary chest X-ray classification.

    Architecture mirrors CheXNet but adapted for binary output with
    calibrated probability output via sigmoid (not softmax).
    """

    def __init__(
        self,
        backbone: str = "densenet121",
        num_classes: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone == "densenet121":
            base = models.densenet121(weights=weights)
            in_features = base.classifier.in_features
            self.features = base.features
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout / 2),
                nn.Linear(512, num_classes),
            )

        elif backbone == "efficientnet_b3":
            base = models.efficientnet_b3(weights=weights)
            in_features = base.classifier[1].in_features
            self.features = base.features
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout / 2),
                nn.Linear(512, num_classes),
            )

        elif backbone == "resnet50":
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout / 2),
                nn.Linear(512, num_classes),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        # DenseNet features output needs ReLU before pooling
        if self.backbone_name == "densenet121":
            features = torch.relu(features)
        pooled = self.gap(features)
        flat = pooled.view(pooled.size(0), -1)
        return self.classifier(flat)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, feature_map) for Grad-CAM and expert fusion."""
        feat_map = self.features(x)
        if self.backbone_name == "densenet121":
            feat_map = torch.relu(feat_map)
        pooled = self.gap(feat_map)
        flat = pooled.view(pooled.size(0), -1)
        return self.classifier(flat), feat_map

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Stage-wise fine-tuning: freeze backbone, train only classifier head."""
        for param in self.features.parameters():
            param.requires_grad = not freeze

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def build_model(
    backbone: str = "densenet121",
    num_classes: int = 2,
    dropout: float = 0.5,
    pretrained: bool = True,
    device: Optional[str] = None,
) -> ChestXRayNet:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    model = ChestXRayNet(backbone, num_classes, dropout, pretrained)
    return model.to(device)
