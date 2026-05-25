"""
acto2.py — La Neurona
Explica de forma interactiva cómo funciona una sola neurona artificial.
"""
import numpy as np
import streamlit as st

from src.viz.neuron import diagrama_neurona, grafico_activacion
from src.model.activations import sigmoid, relu, tanh_fn, ACTIVATION_DESCRIPTIONS


def render():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem 2rem; border-radius: 12px;
                border-left: 4px solid #3B82F6; margin-bottom: 1.5rem;'>
        <h1 style='color:#E2E8F0; margin:0;'>🧠 Acto 2 — La Neurona</h1>
        <p style='color:#94A3B8; margin:0.5rem 0 0 0; font-size:1.1rem;'>
            El bloque fundamental de toda red neuronal.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Analogía ────────────────────────────────────────────────────────
    st.markdown("""
    ## 🎙️ Un jurado con micrófonos

    Imaginá un panel de tres jueces evaluando un cantante.
    Cada juez da su opinión (entrada), pero no todos pesan lo mismo:
    el juez experto tiene el micrófono más potente (peso alto),
    mientras que el novato apenas se escucha (peso bajo).

    La neurona **suma todas las opiniones amplificadas** y decide:
    ¿el cantante pasa o no?

    Eso es exactamente lo que hace una neurona artificial.
    """)

    st.markdown("---")
    st.markdown("## ⚙️ Probalo vos mismo")
    st.markdown("Mové los sliders y observá cómo cambia la salida de la neurona en tiempo real.")

    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### Entradas (xᵢ)")
        x1 = st.slider("Entrada x₁", -2.0, 2.0, 1.0, 0.1, key="x1")
        x2 = st.slider("Entrada x₂", -2.0, 2.0, 0.5, 0.1, key="x2")
        x3 = st.slider("Entrada x₃", -2.0, 2.0, -1.0, 0.1, key="x3")

        st.markdown("#### Pesos (wᵢ)")
        w1 = st.slider("Peso w₁", -3.0, 3.0, 0.8, 0.1, key="w1")
        w2 = st.slider("Peso w₂", -3.0, 3.0, 1.5, 0.1, key="w2")
        w3 = st.slider("Peso w₃", -3.0, 3.0, -0.5, 0.1, key="w3")

        st.markdown("#### Sesgo y activación")
        bias = st.slider("Bias (b)", -3.0, 3.0, 0.0, 0.1, key="bias_n")
        activation = st.selectbox("Función de activación",
                                   ["sigmoid", "relu", "tanh"],
                                   key="act_n")

    with col_viz:
        fig_neuron = diagrama_neurona(
            pesos=[w1, w2, w3],
            bias=bias,
            entradas=[x1, x2, x3],
            activation=activation,
        )
        st.plotly_chart(fig_neuron, use_container_width=True)

        # Resultado interpretado
        act_fns = {"sigmoid": sigmoid, "relu": relu, "tanh": tanh_fn}
        z = w1*x1 + w2*x2 + w3*x3 + bias
        a = float(act_fns[activation](np.array([z]))[0])

        if activation == "sigmoid":
            decision = "✅ CLASE 1 (Sí pasa)" if a >= 0.5 else "❌ CLASE 0 (No pasa)"
            color_d = "#10B981" if a >= 0.5 else "#EF4444"
        else:
            decision = f"Salida = {a:.3f}"
            color_d = "#7C3AED"

        st.markdown(f"""
        <div style='background:#1E293B; padding:1rem; border-radius:8px;
                    text-align:center; border: 1px solid #334155;'>
            <div style='color:#94A3B8; font-size:0.9rem;'>Cálculo:</div>
            <div style='color:#DDD6FE; font-size:1rem; margin:0.3rem 0;'>
                z = {w1:.1f}·{x1:.1f} + {w2:.1f}·{x2:.1f} + {w3:.1f}·{x3:.1f} + {bias:.1f} = {z:.3f}
            </div>
            <div style='color:{color_d}; font-size:1.3rem; font-weight:bold;'>
                {decision}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Función de activación ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📈 La función de activación: el filtro final")
    st.markdown("""
    Después de sumar todo, la neurona aplica una **función de activación**
    que transforma el resultado en algo más útil (por ejemplo, un número entre 0 y 1).

    Sin activación, la red sería solo álgebra lineal — no podría aprender nada complejo.
    """)

    col_act1, col_act2, col_act3 = st.columns(3)
    for col, act_name in zip([col_act1, col_act2, col_act3],
                              ["sigmoid", "relu", "tanh"]):
        with col:
            fig_act = grafico_activacion(act_name)
            st.plotly_chart(fig_act, use_container_width=True)
            st.markdown(f"""
            <div style='background:#1E293B; padding:0.8rem; border-radius:8px;
                        font-size:0.85rem; color:#CBD5E1;'>
                {ACTIVATION_DESCRIPTIONS[act_name]}
            </div>
            """, unsafe_allow_html=True)

    # ── Fórmula matemática ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📐 La fórmula completa")

    st.markdown(r"""
    <div style='background:#1E293B; padding:1.5rem; border-radius:10px;
                border-left:3px solid #7C3AED; font-size:1rem; color:#E2E8F0;'>
    <b>Paso 1 — Combinación lineal (suma ponderada):</b><br>
    <span style='color:#DDD6FE; font-size:1.2rem;'>
        &nbsp;&nbsp;z = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b
    </span>
    <br><br>
    <b>Paso 2 — Activación:</b><br>
    <span style='color:#A7F3D0; font-size:1.2rem;'>
        &nbsp;&nbsp;a = f(z)&nbsp;&nbsp; donde f es sigmoid, ReLU o tanh
    </span>
    <br><br>
    <span style='color:#94A3B8; font-size:0.9rem;'>
    Los <b>pesos (w)</b> determinan qué tan importante es cada entrada.<br>
    El <b>bias (b)</b> desplaza el punto de activación — es como el umbral de decisión.
    </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Rol del bias ─────────────────────────────────────────────────────
    st.markdown("### 🎚️ ¿Qué pasa si cambiás el bias?")
    st.markdown("""
    El bias es como el **nivel de exigencia** del jurado:
    - Bias **negativo** → la neurona necesita señales muy fuertes para activarse (exigente)
    - Bias **cero** → equilibrado
    - Bias **positivo** → la neurona se activa fácilmente (permisiva)

    Sin bias, la frontera de decisión siempre pasa por el origen — una limitación enorme.
    """)

    # Cierre del acto
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem; border-radius: 12px; border: 1px solid #3B82F6;'>
    <h3 style='color:#E2E8F0;'>🎯 ¿Qué aprendiste en este acto?</h3>
    <ul style='color:#CBD5E1;'>
        <li>Una neurona suma sus entradas × pesos y le suma el bias</li>
        <li>Luego aplica una función de activación para "decidir"</li>
        <li>Los pesos controlan la importancia de cada entrada</li>
        <li>El bias ajusta el umbral de activación</li>
    </ul>
    <p style='color:#94A3B8; margin-top:1rem;'>
        → En el siguiente acto vas a ver cómo se conectan muchas neuronas en capas.
    </p>
    </div>
    """, unsafe_allow_html=True)
