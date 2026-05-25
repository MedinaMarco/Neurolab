"""
acto3.py — La Red
Muestra la arquitectura del MLP y el forward pass explicado visualmente.
"""
import numpy as np
import streamlit as st

from src.viz.network import diagrama_red
from src.viz.curves import grafico_backprop_flujo


def render():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem 2rem; border-radius: 12px;
                border-left: 4px solid #10B981; margin-bottom: 1.5rem;'>
        <h1 style='color:#E2E8F0; margin:0;'>🕸️ Acto 3 — La Red</h1>
        <p style='color:#94A3B8; margin:0.5rem 0 0 0; font-size:1.1rem;'>
            Conectar muchas neuronas juntas crea algo mucho más poderoso.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Analogía ────────────────────────────────────────────────────────
    st.markdown("""
    ## 🏢 La empresa de decisiones

    Imaginá una empresa con tres departamentos:

    1. **Recepción (capa de entrada):** recibe la información del mundo exterior
    2. **Análisis (capas ocultas):** procesa y transforma la información
    3. **Dirección (capa de salida):** toma la decisión final

    Cada empleado (neurona) habla solo con los del departamento siguiente.
    El mensaje va de recepción hacia dirección — eso es el **forward pass**.

    Cuando algo sale mal, la crítica viaja al revés para mejorar — eso es el **backpropagation**.
    """)

    # ── Diseñá tu red ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔧 Diseñá tu propia red")

    col1, col2 = st.columns([1, 2])
    with col1:
        n_entradas   = st.selectbox("Neuronas de entrada", [2, 3, 4], key="n_in")
        n_capas_ocultas = st.slider("Capas ocultas", 1, 3, 1, key="n_hidden")
        n_neuronas_oc = st.slider("Neuronas por capa oculta", 2, 8, 4, key="n_noc")
        n_salidas    = 1  # clasificación binaria siempre

        layer_sizes = [n_entradas] + [n_neuronas_oc] * n_capas_ocultas + [n_salidas]
        total_params = sum(
            layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]
            for i in range(len(layer_sizes) - 1)
        )

        st.markdown(f"""
        <div style='background:#1E293B; padding:1rem; border-radius:8px; margin-top:1rem;'>
            <div style='color:#94A3B8; font-size:0.9rem;'>Arquitectura:</div>
            <div style='color:#E2E8F0; font-size:1.1rem; font-weight:bold;'>
                {" → ".join(str(n) for n in layer_sizes)}
            </div>
            <div style='color:#7C3AED; font-size:1rem; margin-top:0.5rem;'>
                🔢 {total_params} parámetros entrenables
            </div>
            <div style='color:#64748B; font-size:0.8rem; margin-top:0.3rem;'>
                (pesos + biases que el modelo va a aprender)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fig_net = diagrama_red(layer_sizes)
        st.pyplot(fig_net, use_container_width=True)

    # ── Forward pass explicado ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("## ➡️ El Forward Pass: cómo viaja un dato por la red")

    st.markdown("""
    Cuando le damos un dato a la red, este viaja **de izquierda a derecha** por todas las capas.
    En cada capa, cada neurona:
    1. Recibe las salidas de todas las neuronas de la capa anterior
    2. Calcula su suma ponderada `z = W · a_anterior + b`
    3. Aplica la activación `a = f(z)`
    4. Pasa su resultado a la siguiente capa

    El dato final que sale es la **predicción** del modelo.
    """)

    # Ejemplo numérico paso a paso
    with st.expander("📊 Ver ejemplo numérico del forward pass (red 2→3→1)"):
        np.random.seed(42)
        W1 = np.array([[0.5, -0.3, 0.8], [0.2, 0.9, -0.4]])
        b1 = np.array([0.1, -0.1, 0.05])
        W2 = np.array([[0.7], [-0.6], [0.4]])
        b2 = np.array([0.0])

        x = np.array([1.0, -0.5])

        z1 = x @ W1 + b1
        a1 = np.maximum(0, z1)  # ReLU
        z2_val = (a1 @ W2 + b2).item()
        a2_val = float(1 / (1 + np.exp(-z2_val)))
        clase  = "1" if a2_val >= 0.5 else "0"

        z1_r = np.round(z1, 3)
        a1_r = np.round(a1, 3)

        st.markdown(
            f"""
**Entrada:** x = [{x[0]}, {x[1]}]

**Capa oculta 1 (3 neuronas con ReLU):**


**Capa de salida (1 neurona con Sigmoid):**


**Prediccion:** {a2_val:.4f} - Clase **{clase}**
"""
        )

    # ── Forward vs Backward ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## ↔️ Forward y Backward: ida y vuelta")

    st.markdown("""
    La red aprende en dos pasos que se repiten muchas veces:
    - **→ Forward pass:** el dato entra y se produce una predicción
    - **← Backward pass:** se calcula el error y los gradientes viajan de vuelta para mejorar los pesos
    """)

    fig_flow = grafico_backprop_flujo(n_capas=len(layer_sizes))
    st.plotly_chart(fig_flow, use_container_width=True)

    # ── Por qué las capas ocultas son la clave ──────────────────────────
    st.markdown("---")
    st.markdown("## 🔑 ¿Por qué las capas ocultas son tan importantes?")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style='background:#1E293B; padding:1.2rem; border-radius:10px;'>
        <h4 style='color:#EF4444;'>❌ Sin capas ocultas</h4>
        <p style='color:#CBD5E1; font-size:0.9rem;'>
        La red solo puede aprender fronteras <b>lineales</b> (líneas rectas).
        El XOR es imposible.
        </p>
        <code style='color:#94A3B8;'>Arquitectura: 2 → 1</code>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style='background:#1E293B; padding:1.2rem; border-radius:10px;'>
        <h4 style='color:#10B981;'>✅ Con capas ocultas</h4>
        <p style='color:#CBD5E1; font-size:0.9rem;'>
        Cada capa oculta <b>transforma el espacio</b>, creando representaciones
        más complejas. El XOR se resuelve fácilmente.
        </p>
        <code style='color:#94A3B8;'>Arquitectura: 2 → 4 → 1</code>
        </div>
        """, unsafe_allow_html=True)

    # Cierre del acto
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem; border-radius: 12px; border: 1px solid #10B981;'>
    <h3 style='color:#E2E8F0;'>🎯 ¿Qué aprendiste en este acto?</h3>
    <ul style='color:#CBD5E1;'>
        <li>Una red neuronal conecta capas de neuronas en secuencia</li>
        <li>El forward pass lleva los datos de entrada hasta la predicción</li>
        <li>Las capas ocultas son las que permiten resolver problemas no lineales</li>
        <li>Más neuronas y capas = mayor capacidad de aprendizaje (pero con límites)</li>
    </ul>
    <p style='color:#94A3B8; margin-top:1rem;'>
        → En el siguiente acto vemos cómo la red <b>aprende de sus errores</b>.
    </p>
    </div>
    """, unsafe_allow_html=True)
