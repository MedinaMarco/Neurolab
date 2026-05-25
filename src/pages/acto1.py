"""
acto1.py — El Problema
Introduce el problema de clasificación de forma intuitiva.
El usuario "intenta" separar puntos con una línea y ve por qué falla en XOR.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.data.datasets import generar_xor, generar_and, DATASET_DESCRIPTIONS


def render():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem 2rem; border-radius: 12px;
                border-left: 4px solid #7C3AED; margin-bottom: 1.5rem;'>
        <h1 style='color:#E2E8F0; margin:0;'>⚡ Acto 1 — El Problema</h1>
        <p style='color:#94A3B8; margin:0.5rem 0 0 0; font-size:1.1rem;'>
            ¿Puede una máquina aprender a distinguir cosas? ¿Y si no es tan fácil?
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Introducción narrativa ──────────────────────────────────────────
    st.markdown("""
    ## 🍎 Imaginemos esto...

    Sos empleado de una frutería. Tenés que clasificar frutas: **manzanas** a la izquierda,
    **naranjas** a la derecha. Fácil, ¿no? Ponés una cinta en el piso y listo.

    Ahora imaginá que las frutas están **mezcladas de forma extraña** —
    manzanas en dos esquinas opuestas, naranjas en las otras dos.
    **¿Podés separarlas con UNA sola cinta?**

    Eso es exactamente el problema que vamos a resolver hoy.
    """)

    # ── Separación lineal: dataset AND ──────────────────────────────────
    st.markdown("---")
    st.markdown("## 🟢 El caso fácil: separación lineal")
    st.markdown("""
    Este problema se llama **AND**. Los puntos azules son "NO" y los verdes son "SÍ".
    Una sola línea recta puede separarlos perfectamente.
    """)

    X_and, y_and = generar_and(n=150, seed=1)
    col1, col2 = st.columns([2, 1])

    with col1:
        # Slider para que el usuario "dibuje" la línea
        angulo = st.slider("Ángulo de la línea (grados)", -90, 90, 45, key="ang_and")
        offset = st.slider("Posición de la línea", -2.0, 2.0, -0.5, step=0.1, key="off_and")

        fig = _grafico_con_linea(X_and, y_and, angulo, offset, titulo="Dataset AND")
        st.plotly_chart(fig, use_container_width=True)

        # Calcular cuántos clasifica bien con esa línea
        theta = np.deg2rad(angulo)
        a_line = np.tan(theta) if angulo != 90 else 1e6
        separacion = X_and[:, 1] - (a_line * X_and[:, 0] + offset)
        pred_linea = (separacion > 0).astype(int)
        acc_linea = np.mean(pred_linea == y_and.astype(int)) * 100

        color = "#10B981" if acc_linea > 80 else "#F59E0B" if acc_linea > 60 else "#EF4444"
        st.markdown(f"""
        <div style='background:#1E293B; padding:1rem; border-radius:8px;
                    border-left:3px solid {color}; text-align:center;'>
            <span style='font-size:1.5rem; color:{color}; font-weight:bold;'>
                Clasificás bien el {acc_linea:.0f}% de los puntos
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#1E293B; padding:1.2rem; border-radius:10px; margin-top:2rem;'>
        <h4 style='color:#7C3AED;'>💡 ¿Por qué funciona?</h4>
        <p style='color:#CBD5E1; font-size:0.9rem;'>
        Los dos grupos de puntos están <b>separados en el espacio</b>.
        Una línea recta puede dividirlos perfectamente.
        <br><br>
        En matemáticas, decimos que este problema es
        <b style='color:#10B981;'>linealmente separable</b>.
        </p>
        </div>
        """, unsafe_allow_html=True)

    # ── El caso difícil: XOR ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔴 El caso imposible: XOR")
    st.markdown("""
    Ahora intentá lo mismo con este dataset. Los puntos verdes están en las esquinas
    **superior-izquierda** y **inferior-derecha**, los azules en las otras dos esquinas.

    > **Misión:** intentá separar los grupos con la línea. ¿Podés?
    """)

    X_xor, y_xor = generar_xor(n=120, ruido=0.15, seed=2)
    col3, col4 = st.columns([2, 1])

    with col3:
        angulo_xor = st.slider("Ángulo de la línea", -90, 90, 30, key="ang_xor")
        offset_xor = st.slider("Posición de la línea", -2.0, 2.0, 0.0, step=0.1, key="off_xor")

        fig_xor = _grafico_con_linea(X_xor, y_xor, angulo_xor, offset_xor, titulo="Dataset XOR")
        st.plotly_chart(fig_xor, use_container_width=True)

        theta_xor = np.deg2rad(angulo_xor)
        a_xor = np.tan(theta_xor) if angulo_xor != 90 else 1e6
        sep_xor = X_xor[:, 1] - (a_xor * X_xor[:, 0] + offset_xor)
        pred_xor = (sep_xor > 0).astype(int)
        acc_xor = np.mean(pred_xor == y_xor.astype(int)) * 100

        color_xor = "#10B981" if acc_xor > 80 else "#F59E0B" if acc_xor > 60 else "#EF4444"
        st.markdown(f"""
        <div style='background:#1E293B; padding:1rem; border-radius:8px;
                    border-left:3px solid {color_xor}; text-align:center;'>
            <span style='font-size:1.5rem; color:{color_xor}; font-weight:bold;'>
                Clasificás bien el {acc_xor:.0f}% de los puntos
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style='background:#2D1B1B; padding:1.2rem; border-radius:10px; margin-top:2rem;
                    border:1px solid #7F1D1D;'>
        <h4 style='color:#EF4444;'>🚫 ¿Por qué falla?</h4>
        <p style='color:#CBD5E1; font-size:0.9rem;'>
        No importa cómo inclines o muevas la línea:
        <b>siempre vas a dejar puntos del lado incorrecto</b>.
        <br><br>
        Esto se llama problema <b style='color:#EF4444;'>no linealmente separable</b>.
        Una sola línea recta nunca lo va a resolver.
        <br><br>
        ¿La solución? <b style='color:#7C3AED;'>Una red neuronal</b> que aprende
        a "doblar" el espacio.
        </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Conclusión del acto ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem; border-radius: 12px; border: 1px solid #7C3AED;'>
    <h3 style='color:#E2E8F0;'>🎯 ¿Qué aprendiste en este acto?</h3>
    <ul style='color:#CBD5E1;'>
        <li>Algunos problemas son <b>fáciles</b> para una línea recta (linealmente separables)</li>
        <li>Otros son <b>imposibles</b> con una sola línea, como el XOR</li>
        <li>Necesitamos algo más poderoso: una <b>red neuronal multicapa</b></li>
    </ul>
    <p style='color:#94A3B8; margin-top:1rem;'>
        → En el siguiente acto vas a conocer el bloque fundamental: <b>la neurona</b>.
    </p>
    </div>
    """, unsafe_allow_html=True)


def _grafico_con_linea(X, y, angulo, offset, titulo):
    """Scatter plot con la línea ajustable superpuesta."""
    fig = go.Figure()

    mask0 = y == 0
    mask1 = y == 1
    fig.add_trace(go.Scatter(
        x=X[mask0, 0], y=X[mask0, 1], mode="markers", name="Clase 0",
        marker=dict(color="#3B82F6", size=9, symbol="circle",
                    line=dict(color="white", width=1))))
    fig.add_trace(go.Scatter(
        x=X[mask1, 0], y=X[mask1, 1], mode="markers", name="Clase 1",
        marker=dict(color="#10B981", size=9, symbol="diamond",
                    line=dict(color="white", width=1))))

    # Dibujar la línea
    x_range = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    theta = np.deg2rad(angulo)
    if abs(angulo) < 89:
        a = np.tan(theta)
        y_line = a * x_range + offset
    else:
        x_range = np.array([offset, offset])
        y_line = np.array([X[:, 1].min() - 0.5, X[:, 1].max() + 0.5])

    fig.add_trace(go.Scatter(
        x=x_range, y=y_line, mode="lines", name="Tu línea",
        line=dict(color="#F59E0B", width=3, dash="dash"),
    ))

    fig.update_layout(
        title=dict(text=titulo, font=dict(color="#E2E8F0")),
        xaxis=dict(title="x₁", color="#94A3B8", gridcolor="#1E293B", zeroline=False),
        yaxis=dict(title="x₂", color="#94A3B8", gridcolor="#1E293B", zeroline=False),
        paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
        font=dict(color="#E2E8F0"), height=350,
        legend=dict(bgcolor="#1E293B", bordercolor="#334155"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
