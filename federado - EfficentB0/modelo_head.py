# modelo_head.py
# EfficientNet-B0 para aprendizaje federado con cabeza separada
# Encoder: se agrega federadamente (pesos globales)
# ClassificationHead: se entrena local, NO se agrega

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


#########################################################################################################
# ENCODER — este es el que se agrega federadamente

class EfficientNetB0Encoder(nn.Module):
    """
    Backbone EfficientNet-B0 sin cabeza de clasificación.
    Retorna un embedding de 1280 dimensiones.
    Este módulo es el que viaja entre centros y se agrega con FedAvg.
    """
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

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)  # [batch, 1280]
        return x


#########################################################################################################
# CABEZA DE CLASIFICACIÓN — local por centro, NO se agrega

class ClassificationHead(nn.Module):
    """
    Cabeza de clasificación binaria (STROKE / CONTROL).
    Se entrena localmente en cada centro.
    Sus pesos NO se incluyen en FedAvg — se guardan por separado.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1280, 1)

    def forward(self, x):
        return self.fc(x)  # [batch, 1] — sin sigmoid, BCEWithLogitsLoss lo maneja


#########################################################################################################

def crear_encoder():
    return EfficientNetB0Encoder()

def crear_cabeza():
    return ClassificationHead()