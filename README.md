# 🧠 NeuroLab — Tutorial Interactivo de Redes Neuronales

Una aplicación web educativa que guía al usuario desde el concepto más básico
hasta entender cómo una red neuronal multicapa (MLP) aprende mediante backpropagation.

---

## 👥 Integrantes

| Nombre | Rol |
|--------|-----|
| Integrante A | Especialista en el modelo (MLP, activaciones, pérdidas) |
| Integrante B | Especialista en visualización (Plotly, Matplotlib) |
| Integrante C | Arquitecto de la app + Storytelling (Streamlit, narrativa, glosario) |

**Modelo elegido:** Opción B — MLP con backpropagation

---

## 🚀 Instalación y ejecución

### Requisitos
- Python 3.10 o superior
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd neurolab

# 2. (Opcional pero recomendado) Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

La aplicación abre automáticamente en `http://localhost:8501`

---

## 📁 Estructura del proyecto

```
neurolab/
├── app.py                  ← Punto de entrada principal
├── requirements.txt        ← Dependencias exactas
├── README.md               ← Este archivo
├── .streamlit/
│   └── config.toml         ← Tema oscuro
├── src/
│   ├── model/
│   │   ├── mlp.py          ← MLP desde cero (NumPy)
│   │   ├── activations.py  ← Sigmoid, ReLU, Tanh y derivadas
│   │   └── losses.py       ← Binary Cross-Entropy, MSE
│   ├── data/
│   │   └── datasets.py     ← Generadores: XOR, Círculos, Espiral, AND
│   ├── viz/
│   │   ├── neuron.py       ← Diagrama interactivo de neurona (Plotly)
│   │   ├── network.py      ← Grafo de arquitectura (Matplotlib)
│   │   ├── boundary.py     ← Frontera de decisión (Plotly)
│   │   └── curves.py       ← Loss y accuracy curves (Plotly)
│   └── pages/
│       ├── acto1.py        ← El Problema
│       ├── acto2.py        ← La Neurona
│       ├── acto3.py        ← La Red
│       ├── acto4.py        ← El Error y Backpropagation
│       ├── acto5.py        ← La Frontera de Decisión
│       ├── acto6.py        ← Tu Turno (experimentación libre)
│       └── glosario.py     ← Glosario de conceptos
└── informe/
    └── informe.pdf         ← Informe en PDF (agregar)
```

---

## 🗺️ Recorrido narrativo

| Acto | Página | Contenido |
|------|--------|-----------|
| 1 | El Problema | Separación lineal vs XOR — el usuario "dibuja" la línea |
| 2 | La Neurona | Diagrama interactivo con sliders de pesos y bias |
| 3 | La Red | Arquitectura MLP, forward pass paso a paso |
| 4 | El Error | BCE, backpropagation con números reales, entrenamiento en vivo |
| 5 | La Frontera | Perceptrón vs MLP en XOR, animación de epochs |
| 6 | Tu Turno | Panel libre con todos los parámetros y 3 datasets |

---

## 🧪 Implementación técnica

El MLP está implementado **desde cero usando solo NumPy**:

- **Forward pass:** multiplicación de matrices + activación capa por capa
- **Backpropagation:** regla de la cadena, gradientes capa por capa
- **Gradient descent:** actualización `w ← w − lr × ∂L/∂w`
- **Sin scikit-learn, TensorFlow ni PyTorch para el modelo**

---

## 📦 Dependencias

```
streamlit>=1.30.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.15.0
pandas>=2.0.0
```
