"""
mlp.py
Red neuronal multicapa (MLP) implementada desde cero con NumPy.

Arquitectura:
  input → [capas ocultas con activación elegida] → output con sigmoid

Algoritmo de entrenamiento:
  1. Forward pass: calcular predicción capa a capa
  2. Calcular pérdida (Binary Cross-Entropy)
  3. Backward pass: propagar el error hacia atrás (backpropagation)
  4. Actualizar pesos con gradient descent

Nada de sklearn, nada de TensorFlow. Solo NumPy.
"""
import numpy as np
from src.model.activations import ACTIVATIONS, sigmoid, sigmoid_deriv
from src.model.losses import binary_cross_entropy


class MLP:
    def __init__(self, layer_sizes: list, activation: str = "relu", lr: float = 0.1):
        """
        Parámetros
        ----------
        layer_sizes : lista de enteros, ej. [2, 4, 4, 1]
                      Primer elemento = nº de entradas
                      Último elemento = nº de salidas
        activation  : función de activación para las capas ocultas
                      ('relu', 'sigmoid', 'tanh')
        lr          : tasa de aprendizaje (learning rate)
        """
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.lr = lr
        self.n_layers = len(layer_sizes)

        self.act_fn, self.act_deriv = ACTIVATIONS[activation]

        # Inicialización de pesos (He para ReLU, Xavier para otros)
        self.weights = []
        self.biases = []
        self._init_weights()

        # Almacenamiento del historial de entrenamiento
        self.history = {"loss": [], "accuracy": [], "epochs": []}
        # Snapshots de pesos para animar la frontera de decisión
        self.weight_snapshots = []

    # ------------------------------------------------------------------ #
    # Inicialización de pesos                                              #
    # ------------------------------------------------------------------ #
    def _init_weights(self):
        np.random.seed(42)
        for i in range(self.n_layers - 1):
            fan_in = self.layer_sizes[i]
            if self.activation_name == "relu":
                scale = np.sqrt(2.0 / fan_in)          # He
            else:
                scale = np.sqrt(1.0 / fan_in)           # Xavier simplificado
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * scale
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    # ------------------------------------------------------------------ #
    # Forward pass                                                         #
    # ------------------------------------------------------------------ #
    def forward(self, X):
        """
        Propaga los datos de entrada a través de toda la red.

        Para cada capa calcula:
            z = X_prev · W + b       (combinación lineal)
            a = activación(z)        (no-linealidad)

        La capa final usa siempre sigmoid para clasificación binaria.

        Guarda todas las activaciones intermedias para el backward pass.
        """
        self._activations = [X]  # a[0] = entrada
        self._zs = []            # valores antes de la activación

        current = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            self._zs.append(z)

            is_output = (i == len(self.weights) - 1)
            if is_output:
                a = sigmoid(z)           # salida: siempre sigmoid
            else:
                a = self.act_fn(z)       # capas ocultas: activación elegida
            self._activations.append(a)
            current = a

        return current  # predicción final, shape (m, 1)

    # ------------------------------------------------------------------ #
    # Backward pass (backpropagation)                                      #
    # ------------------------------------------------------------------ #
    def backward(self, X, y):
        """
        Calcula el gradiente del error respecto a cada peso y bias,
        y actualiza los parámetros usando gradient descent.

        Pasos:
        1. Error en la capa de salida: δ_L = ŷ - y
           (derivada de BCE con sigmoid ya simplificada)

        2. Para cada capa l hacia atrás:
           δ_l = (δ_{l+1} · W_{l+1}^T) * f'(a_l)
           ∂L/∂W_l = a_{l-1}^T · δ_l / m
           ∂L/∂b_l = mean(δ_l, axis=0)

        3. Actualización:
           W_l ← W_l - lr * ∂L/∂W_l
           b_l ← b_l - lr * ∂L/∂b_l
        """
        m = X.shape[0]
        y = y.reshape(-1, 1)

        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Delta en la capa de salida
        # Para BCE + sigmoid: δ = ŷ - y  (fórmula simplificada)
        delta = self._activations[-1] - y

        for i in reversed(range(len(self.weights))):
            a_prev = self._activations[i]
            grads_w[i] = (a_prev.T @ delta) / m
            grads_b[i] = np.mean(delta, axis=0, keepdims=True)

            if i > 0:
                # Propagar el error hacia la capa anterior
                delta = (delta @ self.weights[i].T) * self.act_deriv(self._activations[i])

        # Actualizar todos los pesos y biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i]  -= self.lr * grads_b[i]

    # ------------------------------------------------------------------ #
    # Entrenamiento completo                                               #
    # ------------------------------------------------------------------ #
    def train(self, X, y, epochs: int, snapshot_every: int = 10):
        """
        Entrena el modelo por `epochs` iteraciones.

        En cada epoch:
        - forward  → calcula la predicción
        - loss     → calcula el error
        - accuracy → calcula la exactitud
        - backward → actualiza los pesos

        Retorna el historial para graficar la curva de pérdida.
        """
        self.history = {"loss": [], "accuracy": [], "epochs": []}
        self.weight_snapshots = []

        for epoch in range(epochs):
            output = self.forward(X)
            y_col  = y.reshape(-1, 1)

            loss = binary_cross_entropy(y_col, output)
            preds = (output >= 0.5).astype(int).flatten()
            acc   = np.mean(preds == y.flatten())

            self.history["loss"].append(float(loss))
            self.history["accuracy"].append(float(acc))
            self.history["epochs"].append(epoch)

            # Guardar snapshot para la animación de la frontera
            if epoch % snapshot_every == 0 or epoch == epochs - 1:
                self.weight_snapshots.append({
                    "epoch":   epoch,
                    "loss":    float(loss),
                    "accuracy": float(acc),
                    "weights": [w.copy() for w in self.weights],
                    "biases":  [b.copy() for b in self.biases],
                })

            self.backward(X, y)

        return self.history

    # ------------------------------------------------------------------ #
    # Predicción                                                            #
    # ------------------------------------------------------------------ #
    def predict(self, X):
        """Retorna probabilidades (0–1) para cada muestra."""
        return self.forward(X)

    def predict_classes(self, X):
        """Retorna clases binarias (0 o 1) usando umbral 0.5."""
        return (self.predict(X) >= 0.5).astype(int).flatten()

    # ------------------------------------------------------------------ #
    # Restaurar desde snapshot (para la animación)                         #
    # ------------------------------------------------------------------ #
    def load_snapshot(self, snapshot: dict):
        self.weights = [w.copy() for w in snapshot["weights"]]
        self.biases  = [b.copy() for b in snapshot["biases"]]
