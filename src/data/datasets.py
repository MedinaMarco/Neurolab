"""
datasets.py
Generadores de datasets para clasificación binaria.
Todos usan solo NumPy — sin sklearn para los datos principales.
"""
import numpy as np
import pandas as pd


def generar_xor(n: int = 200, ruido: float = 0.1, seed: int = 0) -> tuple:
    """
    Genera el problema XOR con ruido gaussiano.
    Clases:
      0 → puntos en los cuadrantes (-, -) y (+, +)
      1 → puntos en los cuadrantes (-, +) y (+, -)
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(float)
    X += rng.randn(n, 2) * ruido
    return X.astype(np.float32), y.astype(np.float32)


def generar_circulos(n: int = 200, ruido: float = 0.1, factor: float = 0.5, seed: int = 0) -> tuple:
    """
    Dos círculos concéntricos.
    Clase 0 → círculo externo
    Clase 1 → círculo interno
    """
    rng = np.random.RandomState(seed)
    n_inner = n // 2
    n_outer = n - n_inner

    # Círculo externo
    t_outer = rng.uniform(0, 2 * np.pi, n_outer)
    r_outer = 1.0 + rng.randn(n_outer) * ruido
    X_outer = np.column_stack([r_outer * np.cos(t_outer), r_outer * np.sin(t_outer)])
    y_outer = np.zeros(n_outer)

    # Círculo interno
    t_inner = rng.uniform(0, 2 * np.pi, n_inner)
    r_inner = factor + rng.randn(n_inner) * ruido
    X_inner = np.column_stack([r_inner * np.cos(t_inner), r_inner * np.sin(t_inner)])
    y_inner = np.ones(n_inner)

    X = np.vstack([X_outer, X_inner]).astype(np.float32)
    y = np.concatenate([y_outer, y_inner]).astype(np.float32)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def generar_espiral(n: int = 200, ruido: float = 0.2, seed: int = 0) -> tuple:
    """
    Dos espirales entrelazadas — el problema más difícil.
    Clase 0 → espiral 1
    Clase 1 → espiral 2
    """
    rng = np.random.RandomState(seed)
    n2 = n // 2
    theta = np.sqrt(rng.uniform(0, 1, n2)) * 2 * np.pi

    # Espiral 1
    r1 = -2 * theta - np.pi
    X1 = np.column_stack([r1 * np.cos(theta), r1 * np.sin(theta)])
    X1 += rng.randn(n2, 2) * ruido

    # Espiral 2
    r2 = 2 * theta + np.pi
    X2 = np.column_stack([r2 * np.cos(theta), r2 * np.sin(theta)])
    X2 += rng.randn(n2, 2) * ruido

    X = np.vstack([X1, X2]).astype(np.float32)
    y = np.concatenate([np.zeros(n2), np.ones(n2)]).astype(np.float32)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def generar_and(n: int = 200, ruido: float = 0.05, seed: int = 0) -> tuple:
    """
    Versión continua y ruidosa del problema AND lógico.
    Separable linealmente — ideal para mostrar el perceptrón.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(float)
    X += rng.randn(n, 2) * ruido
    return X.astype(np.float32), y.astype(np.float32)


DATASETS = {
    "XOR":      generar_xor,
    "Círculos": generar_circulos,
    "Espiral":  generar_espiral,
    "AND":      generar_and,
}

DATASET_DESCRIPTIONS = {
    "XOR":      "El problema clásico que un perceptrón simple NO puede resolver. Perfecto para mostrar el poder del MLP.",
    "Círculos": "Puntos dentro y fuera de un círculo. Ninguna línea recta puede separarlos.",
    "Espiral":  "El más difícil: dos espirales entrelazadas. Requiere una red con suficiente capacidad.",
    "AND":      "Separable linealmente. Un perceptrón simple lo resuelve sin problemas.",
}


def a_dataframe(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray = None) -> pd.DataFrame:
    """Convierte arrays en un DataFrame de Pandas para mostrar en la tabla."""
    df = pd.DataFrame({"x₁": X[:, 0], "x₂": X[:, 1], "clase_real": y.astype(int)})
    if y_pred is not None:
        df["predicción"] = y_pred.astype(int)
        df["correcto"] = df["clase_real"] == df["predicción"]
    return df.round(3)
