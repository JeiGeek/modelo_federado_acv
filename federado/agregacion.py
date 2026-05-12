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


# FedAvg adaptativo: pondera por número de muestras y mejora relativa en validación

def fedavg_adaptativo(pesos_centros, muestras_centros, val_losses_anteriores, val_losses_nuevos, eps=1e-8):
    """
    Agregación ponderada por datos + mejora relativa.
    
    Δk = (val_old - val_new) / (val_old + ε)
    Ik = √nk × (1 + Δk)
    αk = Ik / ΣIj
    """

    importancias = []

    for n, val_old, val_new in zip(muestras_centros, val_losses_anteriores, val_losses_nuevos):
        delta = (val_old - val_new) / (val_old + eps)
        ik    = math.sqrt(n) * (1 + delta)
        importancias.append(ik)

    total = sum(importancias)

    # Pesos normalizados
    pesos_globales = copy.deepcopy(pesos_centros[0])
    for key in pesos_globales:
        pesos_globales[key] = torch.zeros_like(pesos_globales[key], dtype=torch.float32)

    for pesos, alpha in zip(pesos_centros, importancias):
        alpha_norm = alpha / total
        for key in pesos_globales:
            pesos_globales[key] += pesos[key].float() * alpha_norm

    return pesos_globales

#########################################################################################################
def q_fedavg(pesos_centros, pesos_efectivos):
    """
    Realiza la agregación de q-FedAvg utilizando los pesos efectivos 
    calculados previamente en el script de entrenamiento.
    """
    # 1. Calculamos la suma total de los pesos efectivos para normalizar
    suma_pesos = sum(pesos_efectivos)
    
    # 2. Inicializamos el modelo global con ceros siguiendo la estructura del primer centro
    pesos_globales = copy.deepcopy(pesos_centros[0])
    for key in pesos_globales:
        pesos_globales[key] = torch.zeros_like(pesos_globales[key], dtype=torch.float32)

<<<<<<< HEAD
    # 3. Agregación ponderada
    for pesos_locales, w_efectivo in zip(pesos_centros, pesos_efectivos):
        # Proporción basada en el peso efectivo (n y loss combinados)
        proporcion = w_efectivo / (suma_pesos + 1e-10) 
        
        for key in pesos_globales:
            # Acumulamos el aporte del centro multiplicado por su peso
            pesos_globales[key] += pesos_locales[key].float() * proporcion

    return pesos_globales
def agregar_pesos(pesos_centros: list, pesos_efectivos: list):
    """
    Función puente que llama a q-FedAvg. 
   	"""
    return q_fedavg(pesos_centros, pesos_efectivos)
=======
def agregar_pesos(pesos_centros, muestras_centros,
                  val_losses_anteriores=None, val_losses_nuevos=None):
    """
    Si se pasan val_losses usa fedavg_adaptativo.
    Si no, usa fedavg normal.
    """
    if val_losses_anteriores is not None and val_losses_nuevos is not None:
        return fedavg_adaptativo(pesos_centros, muestras_centros, val_losses_anteriores, val_losses_nuevos)
    return fedavg(pesos_centros, muestras_centros)
>>>>>>> main
