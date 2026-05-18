# modelo.py
# Modelo ResNet18 + CBAM para aprendizaje federado
# Encoder: se agrega federadamente (pesos globales)
# ClassificationHead: se entrena local, NO se agrega

import torch
import torch.nn as nn
from torchvision.models import resnet18

#########################################################################################################
# CBAM

class ChannelAttention(nn.Module):
    def __init__(self, canales, reduccion=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(canales, canales // reduccion),
            nn.ReLU(),
            nn.Linear(canales // reduccion, canales)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg  = self.mlp(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.mlp(self.max_pool(x).squeeze(-1).squeeze(-1))
        return self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1) * x


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg  = x.mean(dim=1, keepdim=True)
        max_ = x.max(dim=1, keepdim=True).values
        mapa = torch.cat([avg, max_], dim=1)
        return self.sigmoid(self.conv(mapa)) * x


class CBAM(nn.Module):
    def __init__(self, canales):
        super().__init__()
        self.channel = ChannelAttention(canales)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

#########################################################################################################
# ENCODER — este es el que se agrega federadamente

class ResNet18CBAMEncoder(nn.Module):
    """
    Backbone ResNet18 + CBAM sin cabeza de clasificación.
    Retorna un embedding de 512 dimensiones.
    Este módulo es el que viaja entre centros y se agrega con FedAvg.
    """
    def __init__(self):
        super().__init__()

        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool

        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x).squeeze(-1).squeeze(-1)  # [batch, 512]
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
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.fc(x)  # [batch, 1] — sin sigmoid, BCEWithLogitsLoss lo maneja

#########################################################################################################

def crear_encoder():
    return ResNet18CBAMEncoder()

def crear_cabeza():
    return ClassificationHead()