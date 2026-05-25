"""
acto4.py — El Error y el Backpropagation
Explica cómo la red mide su error y cómo mejora sus pesos.
Incluye una explicación paso a paso de backpropagation con números reales.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.model.mlp import MLP
from src.data.datasets import generar_xor
from src.viz.curves import grafico_curvas


def render():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem 2rem; border-radius: 12px;
                border-left: 4px solid #EF4444; margin-bottom: 1.5rem;'>
        <h1 style='color:#E2E8F0; margin:0;'>📉 Acto 4 — El Error y el Aprendizaje</h1>
        <p style='color:#94A3B8; margin:0.5rem 0 0 0; font-size:1.1rem;'>
            Equivocarse es el primer paso para mejorar.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Analogía ────────────────────────────────────────────────────────
    st.markdown("""
    ## 🌡️ El error es como un pronóstico del clima

    El meteorólogo predijo 22°C, pero hoy hizo 28°C.
    El error fue de 6°C. ¿Qué hace el pronóstico de mañana?
    **Ajustar su modelo** para no cometer ese error de vuelta.

    Una red neuronal hace exactamente lo mismo, pero en lugar de ajustar a mano,
    usa matemáticas para saber **exactamente cuánto cambiar cada peso**.
    """)

    # ── Función de pérdida ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📏 ¿Cómo medimos el error?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#1E293B; padding:1.2rem; border-radius:10px;'>
        <h4 style='color:#EF4444;'>Binary Cross-Entropy (BCE)</h4>
        <p style='color:#CBD5E1; font-size:0.9rem;'>
        Es la función de pérdida que usamos. Castiga fuertemente
        cuando la red está <b>muy segura pero equivocada</b>.
        </p>
        <div style='color:#FCA5A5; font-size:1.1rem; margin:0.8rem 0; text-align:center;'>
            L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
        </div>
        <ul style='color:#94A3B8; font-size:0.85rem;'>
            <li><b>ŷ</b> = predicción de la red (entre 0 y 1)</li>
            <li><b>y</b> = valor real (0 o 1)</li>
            <li>Cuanto más se equivoca, <b>más grande es L</b></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Gráfico de BCE vs predicción
        fig_bce = _grafico_bce()
        st.plotly_chart(fig_bce, use_container_width=True)

    # ── Backpropagation explicado ───────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔄 Backpropagation: cómo la red aprende de sus errores")

    st.markdown("""
    Backpropagation es el algoritmo más importante del aprendizaje profundo.
    La idea central es simple: **si cometés un error, identificá quién tuvo la culpa**
    y penalizalo proporcionalmente.

    Matemáticamente, usa la **regla de la cadena** del cálculo para calcular
    cómo contribuye cada peso al error final.
    """)

    with st.expander("🔬 Ver backpropagation paso a paso con números reales"):
        _backprop_step_by_step()

    # ── Gradient descent ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## ⛰️ Gradient Descent: bajar la montaña del error")

    st.markdown("""
    Una vez que sabemos el gradiente (pendiente del error respecto a cada peso),
    **bajamos un pequeño paso en la dirección contraria**:

    > `w ← w − lr × ∂L/∂w`

    Es como estar en una montaña con los ojos vendados:
    sentís hacia dónde cuesta abajo y das un pequeño paso.
    La **tasa de aprendizaje (lr)** es el tamaño del paso.
    """)

    col_gd1, col_gd2, col_gd3 = st.columns(3)
    for col, titulo, desc, color in [
        (col_gd1, "lr muy alto 🚀", "Los pasos son gigantes. Saltás de un lado al otro sin converger.", "#EF4444"),
        (col_gd2, "lr ideal ✅",    "Bajás suavemente hacia el mínimo. Converge en pocos epochs.",     "#10B981"),
        (col_gd3, "lr muy bajo 🐢", "Los pasos son tan chicos que tardás miles de epochs en llegar.",  "#F59E0B"),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:#1E293B; padding:1rem; border-radius:8px;
                        border-top:3px solid {color};'>
                <b style='color:{color};'>{titulo}</b>
                <p style='color:#CBD5E1; font-size:0.85rem; margin-top:0.5rem;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Entrenamiento interactivo ───────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🎮 Entrenemos una red y observemos el error en tiempo real")

    col_p, col_g = st.columns([1, 2])
    with col_p:
        lr_demo = st.select_slider(
            "Tasa de aprendizaje",
            options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
            value=0.1, key="lr_demo"
        )
        epochs_demo = st.slider("Epochs", 50, 500, 200, 50, key="ep_demo")
        activation_demo = st.selectbox("Activación oculta", ["relu", "sigmoid", "tanh"], key="act_demo")
        n_ocultas_demo = st.slider("Neuronas ocultas", 2, 16, 4, key="noc_demo")

        btn_entrenar = st.button("▶ Entrenar red", type="primary", use_container_width=True, key="btn_demo")

    with col_g:
        if btn_entrenar or "history_demo" in st.session_state:
            if btn_entrenar:
                X, y = generar_xor(n=200, seed=0)
                model = MLP(
                    layer_sizes=[2, n_ocultas_demo, 1],
                    activation=activation_demo,
                    lr=lr_demo
                )
                history = model.train(X, y, epochs=epochs_demo)
                st.session_state["history_demo"] = history

            history = st.session_state["history_demo"]
            fig_curves = grafico_curvas(history)
            st.plotly_chart(fig_curves, use_container_width=True)

            final_loss = history["loss"][-1]
            final_acc  = history["accuracy"][-1] * 100
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Loss final", f"{final_loss:.4f}", delta=f"{history['loss'][0]-final_loss:.4f} vs inicio")
            with col_m2:
                st.metric("Accuracy final", f"{final_acc:.1f}%")
        else:
            st.info("👆 Configurá los parámetros y hacé clic en **Entrenar red**")

    # ── Cierre ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem; border-radius: 12px; border: 1px solid #EF4444;'>
    <h3 style='color:#E2E8F0;'>🎯 ¿Qué aprendiste en este acto?</h3>
    <ul style='color:#CBD5E1;'>
        <li>La función de pérdida mide qué tan equivocada está la red</li>
        <li>Backpropagation calcula el gradiente: qué tanto aportó cada peso al error</li>
        <li>Gradient descent actualiza los pesos bajando por el gradiente</li>
        <li>La tasa de aprendizaje controla el tamaño de cada paso</li>
    </ul>
    <p style='color:#94A3B8; margin-top:1rem;'>
        → En el siguiente acto vamos a <b>ver visualmente</b> cómo cambia la frontera de decisión.
    </p>
    </div>
    """, unsafe_allow_html=True)


def _grafico_bce():
    """Grafico de la BCE para y=1 y y=0."""
    pred = np.linspace(0.01, 0.99, 300)
    eps = 1e-12
    bce_y1 = -np.log(pred + eps)
    bce_y0 = -np.log(1 - pred + eps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred, y=bce_y1, name="y_real = 1",
                             line=dict(color="#10B981", width=2.5)))
    fig.add_trace(go.Scatter(x=pred, y=bce_y0, name="y_real = 0",
                             line=dict(color="#EF4444", width=2.5)))
    fig.update_layout(
        title=dict(text="Binary Cross-Entropy", font=dict(color="#E2E8F0")),
        xaxis=dict(title="Predicción (ŷ)", color="#94A3B8", gridcolor="#1E293B"),
        yaxis=dict(title="Pérdida (L)", color="#94A3B8", gridcolor="#1E293B", range=[0, 5]),
        paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
        font=dict(color="#E2E8F0"),
        legend=dict(bgcolor="#1E293B"),
        height=260, margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def _backprop_step_by_step():
    """Ejemplo numérico de backpropagation en una red 2→2→1."""
    st.markdown("""
    **Red de ejemplo:** 2 entradas → 2 neuronas ocultas (sigmoid) → 1 salida (sigmoid)

    **Dato:** x = [0.5, 0.8] | y_real = 1
    """)

    # Pesos fijos para el ejemplo — x e y como vectores 1D para evitar problemas de shape
    W1 = np.array([[0.4, 0.7], [-0.2, 0.5]])
    b1 = np.array([0.1, -0.1])
    W2 = np.array([[0.6], [-0.3]])
    b2 = np.array([0.2])
    x  = np.array([[0.5, 0.8]])   # shape (1,2)
    y  = np.array([[1.0]])         # shape (1,1)
    lr = 0.5

    def sig(z): return 1 / (1 + np.exp(-z))

    # Forward
    z1   = x @ W1 + b1          # (1,2)
    a1   = sig(z1)               # (1,2)
    z2   = a1 @ W2 + b2          # (1,1)
    a2   = sig(z2)               # (1,1)

    # Extraer escalares de forma segura (.item() funciona en todas las versiones de NumPy)
    z2_s = z2.item()
    a2_s = a2.item()
    eps  = 1e-12
    loss = float(-np.log(a2_s + eps))   # y=1, simplificado

    col1, col2 = st.columns(2)
    with col1:
        z1_r = np.round(z1, 4)
        a1_r = np.round(a1, 4)
        txt_forward = (
            "**Multiplicacion z1 = x W1 + b1:**\n"
            f"```\n{z1_r}\n```\n"
            "**Activacion a1 = sigmoid(z1):**\n"
            f"```\n{a1_r}\n```\n"
            f"**Suma z2 = a1 W2 + b2:** `{z2_s:.4f}`\n\n"
            f"**Salida a2 = sigmoid(z2):** `{a2_s:.4f}`\n\n"
            f"**Loss = -log({a2_s:.4f}) = {loss:.4f}**"
        )
        st.markdown(txt_forward)

    # Backward
    delta2  = a2 - y                              # (1,1)  simplificado BCE+sigmoid
    dL_dW2  = a1.T @ delta2                       # (2,1)
    delta1  = (delta2 @ W2.T) * (a1 * (1 - a1))  # (1,2)
    dL_dW1  = x.T @ delta1                        # (2,2)

    W1_new  = W1 - lr * dL_dW1
    W2_new  = W2 - lr * dL_dW2

    d2_s    = delta2.item()
    dW2_r   = np.round(dL_dW2.flatten(), 4)
    d1_r    = np.round(delta1, 4)
    dW1_r   = np.round(dL_dW1, 4)
    W2new_r = np.round(W2_new.flatten(), 4)

    with col2:
        txt_back = (
            f"**Delta salida = a2 - y = {a2_s:.4f} - 1 = {d2_s:.4f}**\n\n"
            f"**Gradiente dL/dW2 = a1.T @ delta:**\n```\n{dW2_r}\n```\n"
            f"**Delta oculta = delta @ W2.T * sigm\'(a1):**\n```\n{d1_r}\n```\n"
            f"**Gradiente dL/dW1:**\n```\n{dW1_r}\n```\n\n"
            f"**Actualizacion W2 = W2 - {lr} * dL/dW2:**\n```\n{W2new_r}\n```"
        )
        st.markdown(txt_back)

    st.success("**Despues del ajuste, la red daria una prediccion mas cercana a 1.** El error bajo.")
