"""
activations.py
Funciones de activación y sus derivadas.
Todas operan sobre arrays de NumPy.
"""
import numpy as np


def sigmoid(z):
    """Aplasta cualquier número al rango (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_deriv(a):
    """Derivada de sigmoid expresada en términos de su salida a = sigmoid(z)."""
    return a * (1.0 - a)


def relu(z):
    """Deja pasar valores positivos, bloquea negativos."""
    return np.maximum(0.0, z)


def relu_deriv(a):
    """Derivada de ReLU: 1 donde a > 0, 0 en otro caso."""
    return (a > 0).astype(float)


def tanh_fn(z):
    """Versión centrada de sigmoid: rango (-1, 1)."""
    return np.tanh(z)


def tanh_deriv(a):
    """Derivada de tanh expresada en términos de su salida a = tanh(z)."""
    return 1.0 - a ** 2


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_deriv),
    "relu":    (relu,    relu_deriv),
    "tanh":    (tanh_fn, tanh_deriv),
}

ACTIVATION_DESCRIPTIONS = {
    "sigmoid": "Sigmoid — salida entre 0 y 1, ideal para la capa final de clasificación binaria.",
    "relu":    "ReLU — rápida y eficiente, recomendada para capas ocultas.",
    "tanh":    "Tanh — salida entre -1 y 1, útil cuando querés salidas centradas.",
}
