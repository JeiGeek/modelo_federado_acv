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