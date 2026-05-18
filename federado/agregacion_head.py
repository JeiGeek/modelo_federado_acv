# agregacion_head.py
# Métodos de agregación para aprendizaje federado
# FedAvg SOLO para backbone global
# La cabeza (head) permanece local

import copy
import torch


# FEDAVG SIN HEAD
def fedavg(pesos_centros: list, muestras_centros: list):
    """
    Promedia únicamente el backbone global.
    La cabeza de clasificación NO se agrega.

    muestras_centros : número de muestras de cada centro
    """

    total = sum(muestras_centros)

    # Inicializar pesos globales
    pesos_globales = copy.deepcopy(pesos_centros[0])

    for key in pesos_globales:

        # NO agregamos la cabeza local
        if key.startswith("head"):
            continue

        pesos_globales[key] = torch.zeros_like(
            pesos_globales[key],
            dtype=torch.float32
        )


    # Promedio ponderado SOLO del backbone
    for pesos, n in zip(pesos_centros, muestras_centros):

        proporcion = n / total

        for key in pesos_globales:

            # Saltamos head local
            if key.startswith("head"):
                continue

            pesos_globales[key] += pesos[key].float() * proporcion

    return pesos_globales


# FUNCIÓN GENERAL
def agregar_pesos(
    pesos_centros,
    muestras_centros,
    val_losses_anteriores=None,
    val_losses_nuevos=None
):
    """
    Usa FedAvg sin agregar la head local.
    """

    return fedavg(pesos_centros, muestras_centros)