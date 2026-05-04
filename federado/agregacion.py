# Métodos de agregación para aprendizaje federado
# FedAvg
# Para cambiar método: reemplazar la función en `agregar_pesos`

import copy
import torch


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

def agregar_pesos(pesos_centros: list, muestras_centros: list):
    # Para otros metódos de agregación, reemplazar esta función por la correspondiente
    return fedavg(pesos_centros, muestras_centros)