# -*- coding: utf-8 -*-
"""Shared model architectures used by both training and serving (single source of truth)."""
import torch
import torch.nn as nn
import torchvision


class SmallXRayCNN(nn.Module):
    """Compact CNN for single-channel X-ray classification (pneumonia)."""
    def __init__(self, num_classes: int = 1, in_ch: int = 1):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(in_ch, 32), block(32, 64), block(64, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_brain_resnet(num_classes: int, pretrained: bool = True, dropout: float = 0.0):
    """ResNet-18 backbone fine-tuned for brain-MRI tumor classification (RGB input).

    dropout=0 keeps the bare `fc` Linear so v1 checkpoints (saved with that layout)
    still load; dropout>0 wraps it in a Sequential, which renames the state_dict keys
    to fc.1.* — so the checkpoint records its dropout and serving rebuilds it the same way.
    """
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    if dropout and dropout > 0:
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, num_classes))
    else:
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def build_pneumonia_resnet(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
    """ResNet-18 with a dropout head — dropout regularizes the classifier against the
    small (4.7k) pneumonia training set. Same factory used at train and serve time."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, num_classes))
    return net
