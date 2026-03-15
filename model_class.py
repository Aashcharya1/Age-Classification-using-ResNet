import torch
import torch.nn as nn
from torchvision import models

class MyAgeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        resnet = models.resnet18(weights=None)
        
        # Keep everything up to avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        
        # Squeeze-and-Excitation (SE) block right after global average pooling
        reduction = 16
        self.se = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs // reduction, num_ftrs, bias=False),
            nn.Sigmoid()
        )
        
        # Deep bottleneck classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Feature representation
        
        # Apply channel attention explicitly
        weight = self.se(x)
        x = x * weight
        
        return self.classifier(x)
