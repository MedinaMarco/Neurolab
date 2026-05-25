"""
matematica.py — Fundamento Matemático Completo
Cubre toda la matematica detras del MLP:
notacion matricial, derivadas, backprop completo, inicializacion de pesos.
Usa st.latex() para renderizado profesional de formulas.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem 2rem; border-radius: 12px;
                border-left: 4px solid #06B6D4; margin-bottom: 1.5rem;'>
        <h1 style='color:#E2E8F0; margin:0;'>📐 Fundamento Matemático</h1>
        <p style='color:#94A3B8; margin:0.5rem 0 0 0; font-size:1.1rem;'>
            La matemática detrás de todo lo que viste en los actos anteriores.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Esta sección es para los que quieren entender **por qué** funciona, no solo **cómo** usarlo.
    Vas a encontrar las derivaciones completas de backpropagation, la justificación de la
    inicialización de pesos, y la geometría del gradient descent.

    > No hace falta haber cursado cálculo avanzado. Cada fórmula viene explicada en lenguaje llano.
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔢 Notación Matricial",
        "📉 Función de Pérdida",
        "⛓️ Backpropagation",
        "🎲 Inicialización de Pesos",
        "⛰️ Gradient Descent",
    ])

    with tab1:
        _notacion_matricial()

    with tab2:
        _funcion_perdida()

    with tab3:
        _backpropagation_completo()

    with tab4:
        _inicializacion_pesos()

    with tab5:
        _gradient_descent()


# ── TAB 1: Notación Matricial ──────────────────────────────────────────
def _notacion_matricial():
    st.markdown("## 🔢 Notación Matricial del Forward Pass")
    st.markdown("""
    Una red neuronal es, en esencia, una cadena de operaciones con matrices.
    Entender la notación matricial te permite leer cualquier paper de deep learning.
    """)

    st.markdown("### Una sola neurona")
    st.markdown("Si tenemos **n entradas**, una neurona calcula:")
    st.latex(r"z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b")
    st.markdown("""
    <div style='background:#1E293B; padding:1rem; border-radius:8px;'>
    <ul style='color:#CBD5E1; margin:0;'>
        <li><b style='color:#DDD6FE;'>x</b> = vector columna de entradas, shape (n, 1)</li>
        <li><b style='color:#DDD6FE;'>w</b> = vector columna de pesos, shape (n, 1)</li>
        <li><b style='color:#DDD6FE;'>b</b> = escalar (bias)</li>
        <li><b style='color:#DDD6FE;'>z</b> = escalar (suma ponderada antes de activación)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Una capa completa (múltiples neuronas en paralelo)")
    st.markdown("""
    Si la capa tiene **m neuronas** y recibe **n entradas**, todas las neuronas
    se computan en paralelo con una sola multiplicación de matrices:
    """)
    st.latex(r"\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#1E293B; padding:1rem; border-radius:8px;'>
        <h4 style='color:#06B6D4;'>Shapes de cada variable</h4>
        <ul style='color:#CBD5E1; font-size:0.9rem;'>
            <li><b>W⁽ˡ⁾</b>: (m, n) → m neuronas, n entradas</li>
            <li><b>a⁽ˡ⁻¹⁾</b>: (n, 1) → salidas de la capa anterior</li>
            <li><b>b⁽ˡ⁾</b>: (m, 1) → un bias por neurona</li>
            <li><b>z⁽ˡ⁾</b>: (m, 1) → resultado para cada neurona</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#1E293B; padding:1rem; border-radius:8px;'>
        <h4 style='color:#06B6D4;'>Ejemplo: capa de 4 neuronas con 2 entradas</h4>
        <p style='color:#CBD5E1; font-size:0.9rem;'>
        W tiene shape (4, 2), x tiene shape (2, 1)<br>
        W·x tiene shape (4, 1) → 4 valores z, uno por neurona<br><br>
        Todo el computo de la capa es <b>una sola línea de NumPy:</b><br>
        <code style='color:#DDD6FE;'>z = W @ a_prev + b</code>
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### El forward pass completo como cadena de capas")
    st.markdown("Para un batch de **m ejemplos** a la vez (más eficiente):")
    st.latex(r"""
    \begin{aligned}
    \mathbf{A}^{(0)} &= \mathbf{X} \quad \text{(entrada, shape: } n \times m\text{)} \\
    \mathbf{Z}^{(l)} &= \mathbf{W}^{(l)} \mathbf{A}^{(l-1)} + \mathbf{b}^{(l)} \\
    \mathbf{A}^{(l)} &= f\!\left(\mathbf{Z}^{(l)}\right) \quad \text{(aplicada elemento a elemento)} \\
    \hat{\mathbf{Y}} &= \mathbf{A}^{(L)} \quad \text{(predicción final)}
    \end{aligned}
    """)
    st.markdown("""
    <div style='background:#0F2437; padding:1rem; border-radius:8px;
                border-left:3px solid #06B6D4;'>
    <p style='color:#7DD3FC; margin:0;'>
    💡 <b>¿Por qué usar matrices en vez de loops?</b><br>
    <span style='color:#CBD5E1;'>
    Procesar m ejemplos a la vez (batch) con matrices es mucho más rápido que
    un loop de m ejemplos. NumPy y las GPUs están optimizados para multiplicación
    de matrices. Un loop de 1000 ejemplos puede ser 100x más lento que la
    multiplicación matricial equivalente.
    </span>
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dimensiones en NeuroLab")
    st.markdown("Para la arquitectura [2, 8, 8, 1] entrenando con 200 ejemplos:")

    dims_data = {
        "Variable": ["X (entrada)", "W⁽¹⁾", "b⁽¹⁾", "Z⁽¹⁾/A⁽¹⁾", "W⁽²⁾", "b⁽²⁾", "Z⁽²⁾/A⁽²⁾", "W⁽³⁾", "b⁽³⁾", "Ŷ (salida)"],
        "Shape": ["(200, 2)", "(2, 8)", "(1, 8)", "(200, 8)", "(8, 8)", "(1, 8)", "(200, 8)", "(8, 1)", "(1, 1)", "(200, 1)"],
        "Descripción": [
            "200 ejemplos, 2 características",
            "8 neuronas reciben 2 entradas cada una",
            "1 bias por neurona (8 neuronas)",
            "200 ejemplos × 8 neuronas",
            "8 neuronas reciben 8 entradas",
            "1 bias por neurona",
            "200 ejemplos × 8 neuronas",
            "1 neurona recibe 8 entradas",
            "1 bias para la salida",
            "200 probabilidades predichas"
        ]
    }
    import pandas as pd
    df_dims = pd.DataFrame(dims_data)
    st.dataframe(df_dims, use_container_width=True, hide_index=True)


# ── TAB 2: Función de Pérdida ──────────────────────────────────────────
def _funcion_perdida():
    st.markdown("## 📉 Función de Pérdida — Binary Cross-Entropy")
    st.markdown("""
    La función de pérdida es el **termómetro del modelo**: mide qué tan equivocado está.
    El entrenamiento consiste en minimizar este número.
    """)

    st.markdown("### ¿Por qué Cross-Entropy y no Error Cuadrático Medio?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#2D1B1B; padding:1rem; border-radius:8px; border:1px solid #7F1D1D;'>
        <h4 style='color:#EF4444;'>❌ MSE para clasificación</h4>
        <p style='color:#CBD5E1; font-size:0.9rem;'>
        El error cuadrático medio penaliza igual a quien
        predice 0.4 que a quien predice 0.01, cuando la
        clase real es 1. La penalización no es proporcional
        a la "gravedad" del error.<br><br>
        Además, combinado con sigmoid produce gradientes
        muy pequeños ("vanishing gradients") al inicio
        del entrenamiento.
        </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#1A2E1A; padding:1rem; border-radius:8px; border:1px solid #14532D;'>
        <h4 style='color:#10B981;'>✅ BCE para clasificación</h4>
        <p style='color:#CBD5E1; font-size:0.9rem;'>
        La cross-entropy está derivada de la teoría de
        la información: mide los "bits" extra que necesitás
        para describir la realidad usando tu modelo como
        referencia. Penaliza exponencialmente las
        predicciones muy seguras pero equivocadas.<br><br>
        Combinada con sigmoid produce un gradiente
        limpio: simplemente (ŷ - y).
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Definición y derivación")
    st.markdown("Para un solo ejemplo:")
    st.latex(r"\mathcal{L}(\hat{y}, y) = -\left[ y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right]")
    st.markdown("Para m ejemplos (el promedio es importante para que el gradiente no dependa del tamaño del dataset):")
    st.latex(r"J = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})")

    with st.expander("📖 Ver de dónde viene esta fórmula — derivación desde probabilidad"):
        st.markdown("### Origen probabilístico de la BCE")
        st.markdown("""
        Modelamos la salida de la red como una distribución de Bernoulli:
        la red predice la probabilidad de que la clase sea 1.
        """)
        st.latex(r"P(y=1 \mid x) = \hat{y} \qquad P(y=0 \mid x) = 1 - \hat{y}")
        st.markdown("Unificando ambos casos en una sola expresión:")
        st.latex(r"P(y \mid x) = \hat{y}^y \cdot (1-\hat{y})^{1-y}")
        st.markdown("""
        Para **m ejemplos independientes**, la probabilidad conjunta
        (likelihood) es el producto de las probabilidades individuales:
        """)
        st.latex(r"\mathcal{L}_{MLE} = \prod_{i=1}^{m} \hat{y}_i^{y_i} \cdot (1-\hat{y}_i)^{1-y_i}")
        st.markdown("""
        Maximizar esto es equivalente a maximizar el log-likelihood
        (el logaritmo convierte el producto en suma, más fácil de derivar):
        """)
        st.latex(r"\log \mathcal{L}_{MLE} = \sum_{i=1}^{m} \left[ y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i) \right]")
        st.markdown("""
        **Maximizar** el log-likelihood es equivalente a **minimizar** su negativo.
        Dividimos por m para normalizar:
        """)
        st.latex(r"J = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i) \right]")
        st.success("**Eso es exactamente la Binary Cross-Entropy.** No es arbitraria: es la consecuencia natural de querer maximizar la probabilidad de los datos observados.")

    st.markdown("---")
    st.markdown("### Visualización interactiva de la pérdida")
    st.markdown("Explorá cómo cambia la pérdida según lo que predice el modelo:")

    col_s, col_g = st.columns([1, 2])
    with col_s:
        y_real   = st.radio("Clase real (y)", [1, 0], key="bce_y")
        y_hat_ex = st.slider("Predicción del modelo (ŷ)", 0.01, 0.99, 0.7, 0.01, key="bce_pred")
        if y_real == 1:
            loss_ex = -np.log(y_hat_ex)
        else:
            loss_ex = -np.log(1 - y_hat_ex)

        color_loss = "#10B981" if loss_ex < 0.5 else "#F59E0B" if loss_ex < 1.5 else "#EF4444"
        st.markdown(f"""
        <div style='background:#1E293B; padding:1rem; border-radius:8px; text-align:center;'>
        <div style='color:#94A3B8; font-size:0.85rem;'>Pérdida calculada:</div>
        <div style='color:{color_loss}; font-size:2rem; font-weight:bold;'>
            {loss_ex:.4f}
        </div>
        <div style='color:#64748B; font-size:0.8rem;'>
            {"Alta — el modelo está muy equivocado" if loss_ex > 1 else "Baja — el modelo predice bien"}
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_g:
        preds = np.linspace(0.01, 0.99, 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=preds, y=-np.log(preds),
                                  name="y_real = 1", line=dict(color="#10B981", width=2.5)))
        fig.add_trace(go.Scatter(x=preds, y=-np.log(1 - preds),
                                  name="y_real = 0", line=dict(color="#EF4444", width=2.5)))
        fig.add_vline(x=y_hat_ex, line=dict(color="#F59E0B", dash="dash", width=2))
        fig.add_scatter(x=[y_hat_ex], y=[loss_ex],
                        mode="markers", name="Tu predicción",
                        marker=dict(size=12, color="#F59E0B", symbol="star"))
        fig.update_layout(
            paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
            font=dict(color="#E2E8F0"), height=280,
            xaxis=dict(title="ŷ (predicción)", color="#94A3B8", gridcolor="#1E293B"),
            yaxis=dict(title="Pérdida L", color="#94A3B8", gridcolor="#1E293B", range=[0, 5]),
            legend=dict(bgcolor="#1E293B"),
            margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### La derivada de la BCE respecto a ŷ")
    st.markdown("Esta derivada es el punto de partida de backpropagation:")
    st.latex(r"\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}")
    st.markdown("Cuando se combina con la derivada de sigmoid (que veremos en la siguiente pestaña), se produce la simplificación más elegante del algoritmo:")
    st.latex(r"\frac{\partial \mathcal{L}}{\partial z^{(L)}} = \hat{y} - y")
    st.success("**Esta simplificación** (la derivada de BCE combinada con sigmoid da simplemente ŷ - y) es la razón por la que se usan juntas. No es casualidad, es diseño matemático.")


# ── TAB 3: Backpropagation Completo ───────────────────────────────────
def _backpropagation_completo():
    st.markdown("## ⛓️ Backpropagation — Derivación Completa")
    st.markdown("""
    Backpropagation es la **aplicación de la regla de la cadena** del cálculo diferencial
    para calcular el gradiente de la pérdida respecto a cada peso de la red.
    """)

    st.markdown("### La regla de la cadena — el ingrediente clave")
    st.markdown("""
    Si z depende de x a través de y (es decir, z = f(y) y y = g(x)), entonces:
    """)
    st.latex(r"\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}")
    st.markdown("""
    En una red neuronal, la pérdida L depende de los pesos W a través de múltiples capas.
    La regla de la cadena permite "descomponer" esa dependencia capa por capa.
    """)

    st.markdown("---")
    st.markdown("### Derivadas de las funciones de activación")
    st.markdown("Son necesarias para el backward pass. Cada una tiene una forma conveniente:")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Sigmoid**")
        st.latex(r"\sigma(z) = \frac{1}{1+e^{-z}}")
        st.latex(r"\sigma'(z) = \sigma(z)(1-\sigma(z))")
        st.markdown("""
        <div style='background:#1E293B; padding:0.7rem; border-radius:6px;'>
        <p style='color:#CBD5E1; font-size:0.8rem; margin:0;'>
        Se expresa en función de su propia salida.
        No necesitás recalcular nada: ya tenés a = σ(z) guardado.
        </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("**ReLU**")
        st.latex(r"f(z) = \max(0, z)")
        st.latex(r"f'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}")
        st.markdown("""
        <div style='background:#1E293B; padding:0.7rem; border-radius:6px;'>
        <p style='color:#CBD5E1; font-size:0.8rem; margin:0;'>
        Funciona como una compuerta:
        si la neurona estaba apagada (z≤0),
        no deja pasar el gradiente.
        </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("**Tanh**")
        st.latex(r"\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}")
        st.latex(r"\tanh'(z) = 1 - \tanh^2(z)")
        st.markdown("""
        <div style='background:#1E293B; padding:0.7rem; border-radius:6px;'>
        <p style='color:#CBD5E1; font-size:0.8rem; margin:0;'>
        También se expresa en función
        de su propia salida. Rango (-1,1)
        la hace más equilibrada que sigmoid.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### El algoritmo completo — derivación paso a paso")
    st.markdown("Consideramos una red de L capas. Queremos calcular ∂J/∂W⁽ˡ⁾ para cada capa l.")

    st.markdown("#### Paso 1 — Error en la capa de salida")
    st.latex(r"\boldsymbol{\delta}^{(L)} = \frac{\partial J}{\partial \mathbf{Z}^{(L)}} = \hat{\mathbf{Y}} - \mathbf{Y}")
    st.markdown("""
    <div style='background:#1E293B; padding:0.8rem; border-radius:8px; margin-bottom:1rem;'>
    <p style='color:#CBD5E1; margin:0; font-size:0.9rem;'>
    Esta es la simplificación de la derivada de BCE respecto a Z⁽ᴸ⁾, usando la regla de la cadena
    a través de la función sigmoid. El resultado es simplemente la diferencia entre predicción y realidad.
    Shape: igual que Ŷ, es decir (1, m) para clasificación binaria.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Paso 2 — Gradientes de los pesos de la capa de salida")
    st.latex(r"\frac{\partial J}{\partial \mathbf{W}^{(L)}} = \frac{1}{m} \boldsymbol{\delta}^{(L)} \left(\mathbf{A}^{(L-1)}\right)^T")
    st.latex(r"\frac{\partial J}{\partial \mathbf{b}^{(L)}} = \frac{1}{m} \sum_{i=1}^{m} \boldsymbol{\delta}^{(L)}")

    st.markdown("#### Paso 3 — Propagación hacia atrás (para l = L-1, L-2, ..., 1)")
    st.latex(r"\boldsymbol{\delta}^{(l)} = \left(\mathbf{W}^{(l+1)}\right)^T \boldsymbol{\delta}^{(l+1)} \odot f'\!\left(\mathbf{Z}^{(l)}\right)")
    st.markdown("""
    <div style='background:#1E293B; padding:0.8rem; border-radius:8px; margin-bottom:1rem;'>
    <p style='color:#CBD5E1; margin:0; font-size:0.9rem;'>
    El símbolo <b>⊙</b> es el producto elemento a elemento (Hadamard).
    <br><br>
    Interpretación: el error de la capa siguiente (δ⁽ˡ⁺¹⁾) se "redistribuye" hacia atrás
    usando los pesos transpuestos (W⁽ˡ⁺¹⁾)ᵀ, y luego se pondera por cuánto contribuyó
    cada neurona de la capa l (la derivada de su activación f').
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Paso 4 — Gradientes de cada capa oculta")
    st.latex(r"\frac{\partial J}{\partial \mathbf{W}^{(l)}} = \frac{1}{m} \boldsymbol{\delta}^{(l)} \left(\mathbf{A}^{(l-1)}\right)^T")
    st.latex(r"\frac{\partial J}{\partial \mathbf{b}^{(l)}} = \frac{1}{m} \sum_{i=1}^{m} \boldsymbol{\delta}^{(l)}")

    st.markdown("#### Paso 5 — Actualización de pesos")
    st.latex(r"\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \alpha \frac{\partial J}{\partial \mathbf{W}^{(l)}}")
    st.latex(r"\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \alpha \frac{\partial J}{\partial \mathbf{b}^{(l)}}")

    st.markdown("---")
    st.markdown("### Visualización del flujo de gradientes")
    _grafico_flujo_gradientes()

    st.markdown("---")
    st.markdown("### El problema del Vanishing Gradient")
    st.markdown("""
    En redes muy profundas, el gradiente se multiplica por la derivada de activación
    en cada capa. Si esa derivada es pequeña (como en sigmoid, donde máximo es 0.25),
    el gradiente se va achicando exponencialmente al propagarse hacia atrás:
    """)
    st.latex(r"\boldsymbol{\delta}^{(1)} = \boldsymbol{\delta}^{(L)} \cdot \prod_{l=2}^{L} \left[\left(\mathbf{W}^{(l)}\right)^T \odot f'\!\left(\mathbf{Z}^{(l-1)}\right)\right]")

    col_vg1, col_vg2 = st.columns(2)
    with col_vg1:
        n_capas = st.slider("Número de capas", 2, 15, 6, key="vg_capas")
        derivada_max = st.select_slider(
            "Derivada máxima de activación",
            options=[0.1, 0.25, 0.5, 1.0],
            value=0.25,
            format_func=lambda x: f"{x} ({'sigmoid' if x==0.25 else 'ReLU aprox' if x==1.0 else 'tanh' if x==0.5 else 'muy pequeña'})",
            key="vg_deriv"
        )
    with col_vg2:
        capas_arr = np.arange(1, n_capas + 1)
        grad_inicial = 1.0
        grads = [grad_inicial * (derivada_max ** k) for k in capas_arr]
        fig_vg = go.Figure()
        colores_vg = ["#EF4444" if g < 0.01 else "#F59E0B" if g < 0.1 else "#10B981" for g in grads]
        fig_vg.add_trace(go.Bar(x=[f"Capa {i}" for i in capas_arr], y=grads,
                                 marker_color=colores_vg))
        fig_vg.update_layout(
            title="Magnitud del gradiente por capa",
            paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
            font=dict(color="#E2E8F0", size=10),
            xaxis=dict(color="#94A3B8", gridcolor="#1E293B"),
            yaxis=dict(color="#94A3B8", gridcolor="#1E293B", title="Magnitud"),
            height=250, margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_vg, use_container_width=True)
        if grads[-1] < 0.001:
            st.error(f"⚠️ En la capa 1, el gradiente es {grads[-1]:.6f} — prácticamente cero. Los pesos no se actualizan.")
        elif grads[-1] < 0.01:
            st.warning(f"⚠️ El gradiente se reduce mucho: {grads[-1]:.4f}")
        else:
            st.success(f"✅ El gradiente se mantiene razonable: {grads[-1]:.4f}")

    st.info("💡 **Por eso ReLU reemplazó a sigmoid en capas ocultas.** Su derivada es 1 para z > 0, lo que evita la multiplicación por números pequeños en cada capa.")


def _grafico_flujo_gradientes():
    """Diagrama visual del flujo forward (verde) y backward (rojo) de gradientes."""
    capas = [2, 4, 4, 1]
    n_layers = len(capas)
    xs = np.linspace(0.1, 0.9, n_layers)

    fig = go.Figure()

    # Nodos
    layer_colors = ["#3B82F6", "#7C3AED", "#7C3AED", "#10B981"]
    layer_names  = ["Entrada", "Oculta 1", "Oculta 2", "Salida"]
    for i, (x, n, color, name) in enumerate(zip(xs, capas, layer_colors, layer_names)):
        ys = np.linspace(0.2, 0.8, n)
        for y in ys:
            fig.add_shape(type="circle", x0=x-0.03, y0=y-0.04, x1=x+0.03, y1=y+0.04,
                          fillcolor="#1E293B", line=dict(color=color, width=2))
        fig.add_annotation(x=x, y=0.1, text=name, showarrow=False,
                           font=dict(size=9, color=color), xanchor="center")

    # Flechas forward (verde)
    for i in range(n_layers - 1):
        fig.add_annotation(x=xs[i+1]-0.04, y=0.75, ax=xs[i]+0.04, ay=0.75,
                           arrowhead=2, arrowcolor="#10B981", arrowwidth=2.5,
                           text="", showarrow=True)
    fig.add_annotation(x=0.5, y=0.88, text="→  Forward pass: x → a⁽¹⁾ → a⁽²⁾ → ŷ",
                       showarrow=False, font=dict(size=10, color="#10B981"))

    # Flechas backward (rojo)
    for i in range(n_layers - 2, 0, -1):
        fig.add_annotation(x=xs[i-1]+0.04, y=0.25, ax=xs[i]-0.04, ay=0.25,
                           arrowhead=2, arrowcolor="#EF4444", arrowwidth=2.5,
                           text="", showarrow=True)
    fig.add_annotation(x=0.5, y=0.12, text="← Backward pass: δ⁽ᴸ⁾ → δ⁽²⁾ → δ⁽¹⁾ → ∂J/∂W",
                       showarrow=False, font=dict(size=10, color="#EF4444"))

    # Etiqueta de pérdida
    fig.add_annotation(x=1.05, y=0.5, text="J\n(Loss)", showarrow=False,
                       font=dict(size=12, color="#F59E0B"), xanchor="left")
    fig.add_annotation(x=xs[-1]+0.04, y=0.5, ax=0.98, ay=0.5,
                       arrowhead=1, arrowcolor="#F59E0B", arrowwidth=1.5,
                       text="", showarrow=True)

    fig.update_layout(
        paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
        xaxis=dict(visible=False, range=[0, 1.15]),
        yaxis=dict(visible=False, range=[0.05, 0.95]),
        height=230, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 4: Inicialización de Pesos ────────────────────────────────────
def _inicializacion_pesos():
    st.markdown("## 🎲 Inicialización de Pesos — Por qué importa")
    st.markdown("""
    Una de las preguntas más frecuentes en defensas de TP:
    **¿Por qué no inicializan los pesos en cero?**
    """)

    st.markdown("### El problema de la inicialización en ceros")
    st.markdown("""
    <div style='background:#2D1B1B; padding:1.2rem; border-radius:10px; border:1px solid #7F1D1D;'>
    <h4 style='color:#EF4444;'>❌ Si todos los pesos son 0:</h4>
    <p style='color:#CBD5E1;'>
    En el forward pass, <b>todas las neuronas de una misma capa calculan exactamente lo mismo</b>:
    z = 0·x + 0 = 0. Por lo tanto, todas tienen la misma activación y el mismo gradiente en el
    backward pass. En la siguiente actualización, todos los pesos cambian en la misma dirección
    y por la misma magnitud.
    <br><br>
    Las neuronas <b>nunca se diferencian entre sí</b>. Toda la red actúa como si tuviera
    una sola neurona por capa. Esto se llama el problema de la <b>simetría</b>.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Las soluciones: Xavier e He")
    st.markdown("La idea es inicializar con valores aleatorios pequeños, calibrados según el tamaño de la capa:")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#1E293B; padding:1rem; border-radius:8px; border-top:3px solid #3B82F6;'>
        <h4 style='color:#3B82F6;'>Xavier / Glorot</h4>
        <p style='color:#CBD5E1; font-size:0.9rem;'>Para <b>sigmoid</b> y <b>tanh</b></p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"W \sim \mathcal{N}\!\left(0,\ \frac{1}{n_{in}}\right)")
        st.markdown("""
        <div style='background:#1E293B; padding:0.8rem; border-radius:8px;'>
        <p style='color:#CBD5E1; font-size:0.85rem; margin:0;'>
        Calibra la varianza para que la señal no explote ni
        desaparezca al propagarse. Diseñado para activaciones
        cuya derivada está centrada en cero.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#1E293B; padding:1rem; border-radius:8px; border-top:3px solid #7C3AED;'>
        <h4 style='color:#7C3AED;'>He / Kaiming</h4>
        <p style='color:#CBD5E1; font-size:0.9rem;'>Para <b>ReLU</b></p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"W \sim \mathcal{N}\!\left(0,\ \frac{2}{n_{in}}\right)")
        st.markdown("""
        <div style='background:#1E293B; padding:0.8rem; border-radius:8px;'>
        <p style='color:#CBD5E1; font-size:0.85rem; margin:0;'>
        ReLU "apaga" aproximadamente la mitad de las neuronas,
        así que necesitás el doble de varianza para compensar.
        El factor 2 es exactamente esa corrección.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Demo: efecto de la inicialización en las activaciones")
    st.markdown("Observá cómo se distribuyen las activaciones según la estrategia de inicialización:")

    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        n_in      = st.slider("Neuronas de entrada (n_in)", 2, 512, 64, key="init_nin")
        init_tipo = st.selectbox("Estrategia",
                                  ["Ceros", "Muy grandes (std=10)", "Xavier (std=1/√n)", "He (std=√2/n)"],
                                  key="init_tipo")
        activacion_init = st.selectbox("Activación", ["relu", "sigmoid", "tanh"], key="init_act")

    with col_plot:
        np.random.seed(42)
        X_demo = np.random.randn(1000, n_in)

        if init_tipo == "Ceros":
            W = np.zeros((n_in, 64))
        elif "grandes" in init_tipo:
            W = np.random.randn(n_in, 64) * 10
        elif "Xavier" in init_tipo:
            W = np.random.randn(n_in, 64) * np.sqrt(1.0 / n_in)
        else:
            W = np.random.randn(n_in, 64) * np.sqrt(2.0 / n_in)

        b = np.zeros(64)
        z_demo = X_demo @ W + b

        if activacion_init == "relu":
            a_demo = np.maximum(0, z_demo)
        elif activacion_init == "sigmoid":
            a_demo = 1 / (1 + np.exp(-np.clip(z_demo, -500, 500)))
        else:
            a_demo = np.tanh(z_demo)

        a_flat = a_demo.flatten()
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=a_flat, nbinsx=50,
            marker_color="#7C3AED", opacity=0.8,
            name="Activaciones"
        ))
        fig_hist.update_layout(
            title=f"Distribución de activaciones — {init_tipo}",
            paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
            font=dict(color="#E2E8F0", size=10),
            xaxis=dict(title="Valor de activación", color="#94A3B8", gridcolor="#1E293B"),
            yaxis=dict(title="Frecuencia", color="#94A3B8", gridcolor="#1E293B"),
            height=260, margin=dict(l=50, r=20, t=40, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        std_val = a_flat.std()
        pct_cero = (a_flat == 0).mean() * 100 if activacion_init == "relu" else 0
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Std de activaciones", f"{std_val:.3f}",
                       help="Cerca de 1 = bueno. Muy grande o muy pequeño = problema.")
        if activacion_init == "relu":
            col_m2.metric("Neuronas apagadas (ReLU)", f"{pct_cero:.1f}%",
                           help="Con He init debería ser ~50%. Con ceros es 0% (todas iguales).")


# ── TAB 5: Gradient Descent ───────────────────────────────────────────
def _gradient_descent():
    st.markdown("## ⛰️ Gradient Descent — La Geometría del Aprendizaje")
    st.markdown("""
    El gradient descent es el algoritmo que usa la red para encontrar los pesos que
    minimizan la pérdida. La idea es geométrica: la pérdida es una superficie en un
    espacio de alta dimensión, y queremos llegar al punto más bajo.
    """)

    st.markdown("### La regla de actualización")
    st.latex(r"\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)")
    st.markdown("""
    <div style='background:#1E293B; padding:1rem; border-radius:8px;'>
    <ul style='color:#CBD5E1;'>
        <li><b style='color:#DDD6FE;'>θ</b> = cualquier parámetro (peso o bias)</li>
        <li><b style='color:#DDD6FE;'>α</b> = learning rate (tamaño del paso)</li>
        <li><b style='color:#DDD6FE;'>∇J</b> = gradiente (apunta hacia arriba de la superficie de pérdida)</li>
        <li>El signo <b>menos</b> hace que vayamos en la dirección <b>opuesta al gradiente</b> (hacia abajo)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Intuición geométrica del learning rate")
    st.markdown("Explorá el efecto del learning rate sobre la trayectoria de descenso:")

    col_lr, col_tray = st.columns([1, 2])
    with col_lr:
        lr_demo = st.select_slider(
            "Learning rate (α)",
            options=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
            value=0.1, key="gd_lr"
        )
        n_steps = st.slider("Número de pasos", 5, 50, 20, key="gd_steps")

    with col_tray:
        # Función de pérdida simple 2D: L(w) = w^2 (parábola)
        # Gradiente: dL/dw = 2w
        w0 = 3.0
        ws = [w0]
        for _ in range(n_steps):
            grad = 2 * ws[-1]
            ws.append(ws[-1] - lr_demo * grad)
            if abs(ws[-1]) > 10:
                break

        ws_arr = np.array(ws)
        losses = ws_arr ** 2

        w_range = np.linspace(-3.5, 3.5, 300)
        loss_range = w_range ** 2

        fig_gd = make_subplots(rows=1, cols=2,
                                subplot_titles=("Trayectoria en la superficie de pérdida",
                                               "Pérdida vs Pasos"))
        fig_gd.add_trace(go.Scatter(x=w_range, y=loss_range,
                                     line=dict(color="#334155", width=2), name="J(w)"), row=1, col=1)
        fig_gd.add_trace(go.Scatter(x=ws_arr, y=losses,
                                     mode="lines+markers",
                                     line=dict(color="#7C3AED", width=2),
                                     marker=dict(size=6, color="#F59E0B"),
                                     name="Trayectoria"), row=1, col=1)
        fig_gd.add_trace(go.Scatter(x=[ws_arr[0]], y=[losses[0]],
                                     mode="markers", marker=dict(size=12, color="#EF4444", symbol="star"),
                                     name="Inicio"), row=1, col=1)
        fig_gd.add_trace(go.Scatter(x=[ws_arr[-1]], y=[losses[-1]],
                                     mode="markers", marker=dict(size=12, color="#10B981", symbol="star"),
                                     name="Final"), row=1, col=1)
        fig_gd.add_trace(go.Scatter(x=list(range(len(losses))), y=losses,
                                     line=dict(color="#EF4444", width=2),
                                     name="Loss"), row=1, col=2)

        fig_gd.update_layout(
            paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
            font=dict(color="#E2E8F0", size=9),
            height=280, margin=dict(l=40, r=20, t=40, b=40),
            showlegend=False,
        )
        for ax in ["xaxis", "xaxis2"]:
            fig_gd.update_layout(**{ax: dict(color="#94A3B8", gridcolor="#1E293B", zeroline=False)})
        fig_gd.update_layout(
            yaxis=dict(color="#94A3B8", gridcolor="#1E293B", title="J(w)"),
            yaxis2=dict(color="#94A3B8", gridcolor="#1E293B", title="Pérdida"),
        )
        for ann in fig_gd.layout.annotations:
            ann.font.color = "#94A3B8"
        st.plotly_chart(fig_gd, use_container_width=True)

        loss_final = losses[-1] if len(losses) > 0 else float("inf")
        if loss_final < 0.001:
            st.success(f"✅ Convergió correctamente. Loss final: {loss_final:.6f}")
        elif loss_final > 5:
            st.error(f"❌ Divergió. Loss final: {loss_final:.2f} — el lr es demasiado alto.")
        else:
            st.warning(f"⚠️ No convergió completamente. Loss final: {loss_final:.4f}")

    st.markdown("---")
    st.markdown("### SGD vs Mini-batch vs Batch")
    st.markdown("""
    En NeuroLab usamos **Batch Gradient Descent** (todos los datos a la vez).
    Existen variantes con diferente balance entre velocidad y estabilidad:
    """)
    gd_data = {
        "Variante": ["Batch GD", "Mini-batch GD", "SGD (Stochastic)"],
        "Ejemplos por paso": ["Todos (m)", "32-256 (típico)", "1"],
        "Gradiente": ["Exacto", "Aproximado", "Muy ruidoso"],
        "Velocidad por epoch": ["Lento", "Rápido", "Muy rápido"],
        "Estabilidad": ["Muy estable", "Estable", "Oscila mucho"],
        "Uso en NeuroLab": ["✅ Sí", "❌ No implementado", "❌ No implementado"],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(gd_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### La condición de convergencia")
    st.markdown("""
    Para una función de pérdida cuadrática simple (L = w²), el gradient descent converge
    si y solo si:
    """)
    st.latex(r"|1 - 2\alpha| < 1 \quad \Longrightarrow \quad 0 < \alpha < 1")
    st.markdown("""
    Para redes neuronales reales la condición es más compleja (depende de los valores
    propios de la matriz Hessiana de la pérdida), pero la intuición es la misma:
    el learning rate debe ser suficientemente pequeño para no "saltar" el mínimo.
    """)

    st.info("""
    💡 **Regla práctica:** si el loss oscila o sube en lugar de bajar, dividí el
    learning rate por 10. Si el loss baja pero muy lento, multiplicalo por 3.
    Ajustá de a órdenes de magnitud, no en valores pequeños.
    """)
