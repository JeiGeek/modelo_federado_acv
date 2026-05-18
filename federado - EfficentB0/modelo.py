# modelo.py
# EfficientNet-B0 para clasificación binaria STROKE/CONTROL
# Entrada: 2 canales (DWI + ADC) | Salida: logit (usar sigmoid para probabilidad)

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0(nn.Module):
    def __init__(self):
        super().__init__()

        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Adaptar primera capa de 3 a 2 canales (DWI + ADC)
        old_weight = base.features[0][0].weight.data
        new_weight = old_weight[:, :2, :, :].clone() * (3 / 2)

        new_conv = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False)
        new_conv.weight = nn.Parameter(new_weight)
        base.features[0][0] = new_conv

        self.features = base.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Linear(1280, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        return x


def crear_modelo():
    return EfficientNetB0()