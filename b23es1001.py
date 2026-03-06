import torch.nn as nn
from torchvision import models

class MyAgeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),   # Normalizes features from the backbone
            nn.Dropout(0.4),            # Lowered from 0.6
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)