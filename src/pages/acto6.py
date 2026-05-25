"""
acto6.py — Tu Turno
Panel de control completo para que el usuario experimente libremente.
Muestra todos los gráficos juntos y la tabla de predicciones con Pandas.
"""
import numpy as np
import pandas as pd
import streamlit as st

from src.model.mlp import MLP
from src.data.datasets import (DATASETS, DATASET_DESCRIPTIONS, a_dataframe)
from src.viz.boundary import grafico_frontera
from src.viz.curves import grafico_curvas
from src.viz.network import diagrama_red


def render():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem 2rem; border-radius: 12px;
                border-left: 4px solid #A78BFA; margin-bottom: 1.5rem;'>
        <h1 style='color:#E2E8F0; margin:0;'>🎮 Acto 6 — Tu Turno</h1>
        <p style='color:#94A3B8; margin:0.5rem 0 0 0; font-size:1.1rem;'>
            Experimentá libremente. Ya sabés lo suficiente para entender lo que pasa.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Llegaste al panel de control completo. Todo lo que aprendiste en los actos anteriores
    se junta acá. Jugá con los parámetros y observá el efecto.

    **Desafíos sugeridos:**
    - ¿Podés entrenar una red que clasifique la espiral con >90% de accuracy?
    - ¿Qué pasa con lr=1.0? ¿Y con lr=0.001?
    - ¿Cuántas neuronas mínimas necesitás para resolver XOR?
    """)

    # ── Panel de control ─────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("⚙️ Panel de control completo", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Dataset**")
            dataset_name = st.selectbox("Problema", list(DATASETS.keys()), key="ds_libre")
            n_puntos     = st.slider("Cantidad de puntos", 100, 500, 200, 50, key="n_pts")
            ruido        = st.slider("Nivel de ruido", 0.0, 0.5, 0.1, 0.05, key="ruido")
            st.info(DATASET_DESCRIPTIONS[dataset_name])

        with col2:
            st.markdown("**Arquitectura**")
            n_capas    = st.slider("Capas ocultas", 1, 3, 2, key="n_c_libre")
            n_neuronas = st.slider("Neuronas por capa", 2, 16, 6, key="n_n_libre")
            activation = st.selectbox("Activación oculta", ["relu", "sigmoid", "tanh"], key="act_libre")

        with col3:
            st.markdown("**Entrenamiento**")
            lr = st.select_slider(
                "Tasa de aprendizaje (lr)",
                options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
                value=0.1, key="lr_libre"
            )
            epochs = st.slider("Epochs", 50, 2000, 500, 50, key="ep_libre")

            layer_sizes = [2] + [n_neuronas] * n_capas + [1]
            total_params = sum(
                layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]
                for i in range(len(layer_sizes) - 1)
            )
            st.markdown(f"""
            <div style='background:#1E293B; padding:0.8rem; border-radius:8px;
                        font-size:0.9rem; color:#94A3B8; margin-top:0.5rem;'>
            Arquitectura: <b style='color:#E2E8F0;'>{" → ".join(map(str, layer_sizes))}</b><br>
            Parámetros: <b style='color:#7C3AED;'>{total_params}</b>
            </div>
            """, unsafe_allow_html=True)

        btn_libre = st.button("🚀 Entrenar modelo", type="primary",
                              use_container_width=True, key="btn_libre")

    # ── Entrenamiento ────────────────────────────────────────────────────
    if btn_libre:
        gen_fn = DATASETS[dataset_name]
        X, y = gen_fn(n=n_puntos, ruido=ruido, seed=42)

        model = MLP(layer_sizes=layer_sizes, activation=activation, lr=lr)
        with st.spinner(f"Entrenando {epochs} epochs..."):
            history = model.train(X, y, epochs=epochs, snapshot_every=max(1, epochs // 30))

        st.session_state.update({
            "model_libre":   model,
            "X_libre":       X,
            "y_libre":       y,
            "history_libre": history,
            "ds_name_libre": dataset_name,
        })

    # ── Resultados ───────────────────────────────────────────────────────
    if "model_libre" in st.session_state:
        model   = st.session_state["model_libre"]
        X       = st.session_state["X_libre"]
        y       = st.session_state["y_libre"]
        history = st.session_state["history_libre"]
        ds_name = st.session_state["ds_name_libre"]

        preds    = model.predict_classes(X)
        probs    = model.predict(X).flatten()
        final_acc  = np.mean(preds == y.astype(int)) * 100
        final_loss = history["loss"][-1]

        # ── Métricas principales ──────────────────────────────────────────
        st.markdown("### 📊 Resultados del entrenamiento")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{final_acc:.1f}%",
                  delta=f"+{(final_acc - 50):.1f}% vs azar")
        m2.metric("Loss final", f"{final_loss:.4f}",
                  delta=f"{history['loss'][0]-final_loss:.4f} de mejora")
        m3.metric("Epochs", epochs)
        m4.metric("Parámetros", total_params if btn_libre else "—")

        # Calificación automática
        if final_acc >= 95:
            st.success("🏆 ¡Excelente! La red aprendió muy bien el problema.")
        elif final_acc >= 80:
            st.success("✅ Buen resultado. Podés intentar ajustar los parámetros para mejorar.")
        elif final_acc >= 60:
            st.warning("⚠️ Resultado mediocre. Probá más neuronas, más epochs o un lr diferente.")
        else:
            st.error("❌ La red no aprendió. Revisá la tasa de aprendizaje o la arquitectura.")

        # ── Gráficos principales ──────────────────────────────────────────
        tab_vis, tab_curvas, tab_red, tab_tabla = st.tabs([
            "🗺️ Frontera", "📈 Curvas", "🕸️ Arquitectura", "📋 Predicciones"
        ])

        with tab_vis:
            fig_b = grafico_frontera(model, X, y,
                                     titulo=f"{ds_name} — Accuracy: {final_acc:.1f}%",
                                     accuracy=final_acc / 100)
            st.plotly_chart(fig_b, use_container_width=True)

        with tab_curvas:
            fig_c = grafico_curvas(history)
            st.plotly_chart(fig_c, use_container_width=True)

        with tab_red:
            fig_net = diagrama_red(model.layer_sizes, model.weights)
            st.pyplot(fig_net, use_container_width=True)
            st.caption("Los colores de las conexiones representan el valor de cada peso: verde = positivo, rojo = negativo.")

        with tab_tabla:
            _mostrar_tabla_predicciones(X, y, preds, probs)

        # ── Análisis de resultados ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 ¿Qué aprendió la red?")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            n_correcto = int(np.sum(preds == y.astype(int)))
            n_total    = len(y)
            st.markdown(f"""
            <div style='background:#1E293B; padding:1.2rem; border-radius:10px;'>
            <h4 style='color:#E2E8F0;'>Resumen de clasificación</h4>
            <ul style='color:#CBD5E1;'>
                <li>✅ Correctas: <b>{n_correcto}</b> de {n_total}</li>
                <li>❌ Incorrectas: <b>{n_total - n_correcto}</b> de {n_total}</li>
                <li>🎯 Accuracy: <b>{final_acc:.1f}%</b></li>
                <li>📉 Loss final: <b>{final_loss:.5f}</b></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            avg_conf_correct = float(probs[preds == y.astype(int)].mean()) if n_correcto > 0 else 0
            avg_conf_wrong   = float(probs[preds != y.astype(int)].mean()) if (n_total - n_correcto) > 0 else 0
            st.markdown(f"""
            <div style='background:#1E293B; padding:1.2rem; border-radius:10px;'>
            <h4 style='color:#E2E8F0;'>Confianza del modelo</h4>
            <ul style='color:#CBD5E1;'>
                <li>Confianza en aciertos: <b>{avg_conf_correct:.2%}</b></li>
                <li>Confianza en errores: <b>{avg_conf_wrong:.2%}</b></li>
            </ul>
            <p style='color:#94A3B8; font-size:0.85rem; margin-top:0.5rem;'>
            Un modelo sobreajustado puede tener alta confianza pero generalizar mal.
            </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("👆 Configurá los parámetros en el panel de control y hacé clic en **Entrenar modelo**")

    # ── Reflexión final ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #2D1B4E, #1A1A2E);
                padding: 2rem; border-radius: 12px; border: 1px solid #7C3AED;'>
    <h2 style='color:#E2E8F0; text-align:center;'>🎓 ¿Qué aprendiste en todo el recorrido?</h2>
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1rem;'>
    </div>
    <ul style='color:#CBD5E1; font-size:1rem; line-height:1.8;'>
        <li>Una neurona combina entradas con pesos, suma un bias y aplica una activación</li>
        <li>Una red neuronal conecta capas de neuronas para resolver problemas complejos</li>
        <li>El forward pass produce una predicción; la función de pérdida mide el error</li>
        <li>Backpropagation calcula los gradientes propagando el error hacia atrás</li>
        <li>Gradient descent actualiza los pesos para reducir el error en cada epoch</li>
        <li>La frontera de decisión evoluciona con el entrenamiento hasta separar las clases</li>
    </ul>
    <p style='color:#A78BFA; text-align:center; margin-top:1.5rem; font-size:1.1rem;'>
        <b>Próximos pasos:</b> redes convolucionales (CNN), redes recurrentes (RNN),
        regularización, dropout, batch normalization, transformers.
    </p>
    </div>
    """, unsafe_allow_html=True)


def _mostrar_tabla_predicciones(X, y, preds, probs):
    """Tabla interactiva de predicciones vs valores reales."""
    df = a_dataframe(X, y, preds)
    df["probabilidad"] = np.round(probs, 3)
    df["confianza (%)"] = np.round(np.where(preds == 1, probs, 1 - probs) * 100, 1)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filtro = st.selectbox("Filtrar por", ["Todos", "Correctos", "Incorrectos"], key="filtro_tabla")
    with col_f2:
        max_rows = st.slider("Filas a mostrar", 10, min(200, len(df)), 50, key="n_rows_tabla")

    if filtro == "Correctos":
        df_show = df[df["correcto"] == True].head(max_rows)
    elif filtro == "Incorrectos":
        df_show = df[df["correcto"] == False].head(max_rows)
    else:
        df_show = df.head(max_rows)

    # Colorear filas incorrectas
    def highlight_wrong(row):
        color = "background-color: #2D1B1B" if not row["correcto"] else ""
        return [color] * len(row)

    st.dataframe(
        df_show.style.apply(highlight_wrong, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    n_wrong = int((~df["correcto"]).sum())
    st.caption(f"{n_wrong} predicciones incorrectas de {len(df)} totales "
               f"({n_wrong/len(df)*100:.1f}% de error)")
