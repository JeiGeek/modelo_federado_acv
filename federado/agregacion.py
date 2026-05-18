# Métodos de agregación para aprendizaje federado
# FedAvg
# Para cambiar método: reemplazar la función en `agregar_pesos`

import copy
import torch
import math


#########################################################################################################

# FedAvg: promedio ponderado por número de muestras

def fedavg(pesos_centros: list, muestras_centros: list):
    """
    Promedia los pesos de todos los centros ponderado
    por el número de muestras de cada uno.

        pesos_centros    : lista de state_dict de cada centro
        muestras_centros : número de muestras de cada centro
    """
    total = sum(muestras_centros)

    # Inicializamos pesos globales en cero
    pesos_globales = copy.deepcopy(pesos_centros[0])
    for key in pesos_globales:
        pesos_globales[key] = torch.zeros_like(pesos_globales[key], dtype=torch.float32)

    # Acumulamos contribución ponderada de cada centro
    for pesos, n in zip(pesos_centros, muestras_centros):
        proporcion = n / total
        for key in pesos_globales:
            pesos_globales[key] += pesos[key].float() * proporcion

    return pesos_globales

#########################################################################################################

def agregar_pesos(pesos_centros, muestras_centros,
                  val_losses_anteriores=None, val_losses_nuevos=None):
    """
    Si se pasan val_losses usa fedavg_adaptativo.
    Si no, usa fedavg normal.
    """
    if val_losses_anteriores is not None and val_losses_nuevos is not None:
        return fedavg_adaptativo(pesos_centros, muestras_centros, val_losses_anteriores, val_losses_nuevos)
    return fedavg(pesos_centros, muestras_centros)