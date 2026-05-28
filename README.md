<div align="center">

# 🧠 NeuroLab — Tutorial Interactivo de Redes Neuronales

**Aprendé cómo una red neuronal realmente aprende — sin magia, con matemática y código**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![NumPy](https://img.shields.io/badge/NumPy-from%20scratch-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> MLP implementado **desde cero con NumPy** (sin TensorFlow, sin PyTorch). Una app educativa en 6 actos que te lleva desde el problema XOR hasta entrenar tu propia red neuronal en tiempo real.

[Ver Demo](#) · [Reportar Bug](https://github.com/MedinaMarco/Neurolab/issues) · [GitHub](https://github.com/MedinaMarco/Neurolab)

</div>

---

## 🎯 ¿Por qué existe NeuroLab?

Las redes neuronales parecen una caja negra. La mayoría de los tutoriales arrancan con `model.fit()` y nunca explican qué pasa adentro.

NeuroLab rompe esa caja: **cada operación matemática es visible, cada peso se actualiza frente al usuario**, y podés interactuar con el modelo en cada paso del proceso.

---

## 🗺️ Recorrido narrativo (6 actos)

| Acto | Nombre | Qué aprendés |
|------|--------|--------------|
| 1 | **El Problema** | Por qué XOR no es linealmente separable — el usuario "dibuja" la línea |
| 2 | **La Neurona** | Diagrama interactivo con sliders de pesos y bias; ves la activación en tiempo real |
| 3 | **La Red** | Arquitectura MLP, cómo fluye la información (forward pass) capa por capa |
| 4 | **El Error** | Binary Cross-Entropy, backpropagation con números reales, entrenamiento en vivo |
| 5 | **La Frontera** | Perceptrón vs MLP en XOR — animación de cómo cambia la decisión epoch a epoch |
| 6 | **Tu Turno** | Panel libre: elegís arquitectura, dataset, hiperparámetros y entrenás |

---

## ✨ Funcionalidades

- 🔢 **MLP desde cero** — forward pass y backprop implementados con matrices NumPy puras
- 📐 **Visualización de gradientes** en cada capa durante el entrenamiento
- 🎛️ **Sliders interactivos** de pesos, bias, learning rate y arquitectura
- 🌀 **4 datasets** incluidos: XOR, Círculos, Espiral, AND
- 📉 **Curvas de loss y accuracy** en tiempo real durante el entrenamiento
- 🗺️ **Frontera de decisión animada** por epoch

---

## 🧪 Implementación técnica

El MLP está implementado **exclusivamente con NumPy**:

```python
# Forward pass (simplificado)
def forward(self, X):
    self.activations = [X]
    for W, b, fn in zip(self.weights, self.biases, self.activation_fns):
        Z = self.activations[-1] @ W + b
        self.activations.append(fn(Z))
    return self.activations[-1]

# Backpropagation — regla de la cadena capa por capa
def backward(self, y_true):
    delta = self.activations[-1] - y_true
    for i in reversed(range(len(self.weights))):
        dW = self.activations[i].T @ delta
        self.weights[i] -= self.lr * dW
        delta = (delta @ self.weights[i].T) * self.activation_derivs[i](self.activations[i])
```

**Funciones de activación disponibles:** Sigmoid · ReLU · Tanh (con derivadas analíticas)
**Funciones de pérdida:** Binary Cross-Entropy · MSE

---

## 🚀 Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/MedinaMarco/Neurolab.git
cd Neurolab

# 2. (Recomendado) Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
streamlit run app.py
```

---

## 📁 Estructura del proyecto

```
neurolab/
├── app.py                   ← Punto de entrada principal
├── requirements.txt
├── .streamlit/
│   └── config.toml          ← Tema oscuro
└── src/
    ├── model/
    │   ├── mlp.py            ← MLP desde cero (NumPy puro)
    │   ├── activations.py    ← Sigmoid, ReLU, Tanh + derivadas
    │   └── losses.py         ← BCE, MSE
    ├── data/
    │   └── datasets.py       ← XOR, Círculos, Espiral, AND
    ├── viz/
    │   ├── neuron.py         ← Diagrama interactivo de neurona (Plotly)
    │   ├── network.py        ← Grafo de arquitectura (Matplotlib)
    │   ├── boundary.py       ← Frontera de decisión (Plotly)
    │   └── curves.py         ← Loss/accuracy curves
    └── pages/
        ├── acto1.py … acto6.py
        └── glosario.py
```

---

## 👥 Equipo

| Integrante | GitHub |
|-----------|--------|
| Marco Medina | [@MedinaMarco](https://github.com/MedinaMarco) |
| Nicolas Mesquiatti | [@Nicolas-Mesquiatti](https://github.com/Nicolas-Mesquiatti) |
| Cristian Monzon | [@VynCrey](https://github.com/VynCrey) |

Instituto Tecnológico Beltrán — Tecnicatura Superior en Ciencia de Datos e IA
