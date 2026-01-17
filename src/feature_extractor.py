# src/feature_extractor.py
import torch
import torch.nn as nn
from torchvision import models


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone="resnet18", layers=("layer2", "layer3")):
        super().__init__()

        if backbone == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.layers = layers

        # Freeze weights (PatchCore ไม่เทรน backbone)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        feats = {}

        x = self.stem(x)
        x = self.layer1(x)

        x = self.layer2(x)
        if "layer2" in self.layers:
            feats["layer2"] = x

        x = self.layer3(x)
        if "layer3" in self.layers:
            feats["layer3"] = x

        x = self.layer4(x)
        if "layer4" in self.layers:
            feats["layer4"] = x

        return feats