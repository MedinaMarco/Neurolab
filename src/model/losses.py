"""
losses.py
Funciones de pérdida para clasificación binaria.
"""
import numpy as np


def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross-Entropy: mide qué tan lejos está la predicción de la realidad.
    Fórmula: -mean( y*log(ŷ) + (1-y)*log(1-ŷ) )
    """
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))


def mse(y_true, y_pred):
    """
    Mean Squared Error: promedio del cuadrado de los errores.
    Fórmula: mean( (y - ŷ)² )
    """
    return np.mean((y_true - y_pred) ** 2)
