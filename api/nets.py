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


# --- MedMNIST backbones ---------------------------------------------------------------
# TRAINING_LOG session 1 assumed ConvNeXt / EfficientNet / ResNet-50 "do not fit in 4 GB
# VRAM" and routed them to Colab. Session 6 measured it instead, at 224px fp32 on the
# GTX 1650 Max-Q (4096 MiB): resnet18 bs32 -> 897 MB, efficientnet_b0 bs24 -> 2146 MB,
# resnet50 bs24 -> 2370 MB, convnext_tiny bs16 -> 2027 MB. All of them fit. The assumption
# was wrong, so the stronger backbones are available locally.
#
# Every entry returns (model, head_prefix). head_prefix is the state_dict prefix of the
# classifier, which the trainer needs to freeze the backbone in stage A — it is "fc" for
# ResNets and "classifier" for EfficientNet/ConvNeXt, so it cannot be hardcoded.
MEDMNIST_ARCHS = ("resnet18", "resnet50", "efficientnet_b0", "convnext_tiny")

# Measured-safe batch size per arch on a 4 GB card at 224px, fp32 (AMP is off on this GPU —
# see the nan note in train_medmnist_v2.py). Used as the default when BATCH is not set.
ARCH_MAX_BATCH = {"resnet18": 32, "resnet50": 24, "efficientnet_b0": 24, "convnext_tiny": 16}


def build_medmnist_backbone(arch: str, num_classes: int, pretrained: bool = True,
                            dropout: float = 0.4):
    """Build one of MEDMNIST_ARCHS with a dropout classifier head.

    Returns (model, head_prefix). The resnet18 branch delegates to build_brain_resnet so it
    stays byte-identical to the layout every existing v1/v2 checkpoint was saved with —
    changing it here would silently break loading of the twelve models already shipped.

    ResNet-50 uses IMAGENET1K_V2 weights (80.9% ImageNet top-1) rather than V1 (76.1%): the
    point of moving off resnet18 (69.8%) is a better feature extractor, and V2 is free.
    """
    import torch.nn as _nn
    if arch not in MEDMNIST_ARCHS:
        raise ValueError("unknown arch %r; expected one of %s" % (arch, ", ".join(MEDMNIST_ARCHS)))

    if arch == "resnet18":
        return build_brain_resnet(num_classes, pretrained=pretrained, dropout=dropout), "fc"

    if arch == "resnet50":
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = torchvision.models.resnet50(weights=w)
        net.fc = _nn.Sequential(_nn.Dropout(dropout), _nn.Linear(net.fc.in_features, num_classes))
        return net, "fc"

    if arch == "efficientnet_b0":
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.efficientnet_b0(weights=w)
        net.classifier = _nn.Sequential(
            _nn.Dropout(dropout), _nn.Linear(net.classifier[1].in_features, num_classes))
        return net, "classifier"

    w = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.convnext_tiny(weights=w)
    net.classifier[2] = _nn.Linear(net.classifier[2].in_features, num_classes)
    return net, "classifier"
