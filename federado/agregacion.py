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
def q_fedavg(pesos_centros, q_weights):
    """
    Agregación q-FedAvg usando pesos pre-calculados (n / loss^q).
    """
    suma_pesos = sum(q_weights)
    pesos_globales = copy.deepcopy(pesos_centros[0])
    
    # Inicializar en cero
    for key in pesos_globales:
        pesos_globales[key] = torch.zeros_like(pesos_globales[key], dtype=torch.float32)
        
    # Promedio ponderado por el peso efectivo q
    for pesos, w_q in zip(pesos_centros, q_weights):
        proporcion = w_q / (suma_pesos + 1e-10)
        for key in pesos_globales:
            pesos_globales[key] += pesos[key].float() * proporcion
            
    return pesos_globales

def agregar_pesos(pesos_centros, muestras_centros, 
                  val_losses_anteriores=None, val_losses_nuevos=None, 
                  q_weights=None):
    """
    Agregador dinámico:
    1. Si se pasan q_weights, usa q_fedavg.
    2. Si se pasan losses, usa fedavg_adaptativo.
    3. Por defecto, usa fedavg (promedio por muestras).
    """
    
    # Prioridad 1: q-FedAvg (Usa los pesos efectivos calculados con la pérdida y q)
    if q_weights is not None:
        print(" -> Usando agregación: q-FedAvg")
        return q_fedavg(pesos_centros, q_weights)
    
    # Prioridad 2: FedAvg Adaptativo (Basado en la evolución de la pérdida)
    if val_losses_anteriores is not None and val_losses_nuevos is not None:
        print(" -> Usando agregación: FedAvg Adaptativo")
        return fedavg_adaptativo(pesos_centros, muestras_centros, 
                                 val_losses_anteriores, val_losses_nuevos)
    
    # Prioridad 3: FedAvg Estándar (Promedio ponderado simple por n)
    print(" -> Usando agregación: FedAvg Estándar")
    return fedavg(pesos_centros, muestras_centros)
