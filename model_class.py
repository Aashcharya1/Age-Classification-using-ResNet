import torch
import torch.nn as nn
from torchvision import models


class AgeClassifier(nn.Module):
    """
    ResNet-18 with heavy classification head.
    Built-in Test Time Augmentation (horizontal flip) during inference.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        # ResNet-18 backbone (trained from scratch, no pretrained weights)
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Heavy head: Linear(512→256) → BN → GELU → Dropout → Linear(256→2)
        self.feature_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)

    def extract_features(self, x):
        """Backbone + projection → 256-dim features."""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.feature_proj(x)

    def classify(self, features):
        """Dropout + final linear → logits."""
        return self.classifier(self.dropout(features))

    def forward(self, x, return_features=False):
        if not self.training:
            # --- TTA: average original + horizontally flipped predictions ---
            logits = self.classify(self.extract_features(x))
            logits_flip = self.classify(self.extract_features(torch.flip(x, [3])))
            return (logits + logits_flip) / 2

        # --- Training: normal forward pass ---
        features = self.extract_features(x)
        logits = self.classify(features)
        return (logits, features) if return_features else logits


def build_model(num_classes=2):
    """Build AgeClassifier (ResNet-18 backbone + heavy head + TTA)."""
    return AgeClassifier(num_classes=num_classes)
