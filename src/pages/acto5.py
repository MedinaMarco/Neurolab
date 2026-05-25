"""
acto5.py — La Frontera de Decisión
Muestra cómo evoluciona la frontera de decisión durante el entrenamiento.
Permite comparar perceptrón (falla en XOR) vs MLP (lo resuelve).
"""
import numpy as np
import streamlit as st

from src.model.mlp import MLP
from src.data.datasets import generar_xor, generar_and, generar_circulos
from src.viz.boundary import grafico_frontera, grafico_frontera_snapshot


def render():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem 2rem; border-radius: 12px;
                border-left: 4px solid #F59E0B; margin-bottom: 1.5rem;'>
        <h1 style='color:#E2E8F0; margin:0;'>🗺️ Acto 5 — La Frontera de Decisión</h1>
        <p style='color:#94A3B8; margin:0.5rem 0 0 0; font-size:1.1rem;'>
            Mirá cómo la red aprende a dividir el espacio.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🖼️ Ver para creer

    Hasta ahora vimos números: pesos, gradientes, pérdida.
    Pero nada comunica mejor el aprendizaje que **verlo visualmente**.

    La **frontera de decisión** es la línea (o curva) que separa las dos clases.
    - En la zona **azul**, la red predice Clase 0
    - En la zona **verde**, la red predice Clase 1
    - La **línea amarilla** es la frontera exacta (donde ŷ = 0.5)

    Al principio es un caos. A medida que entrena, la frontera se acomoda.
    """)

    # ── Tab 1: Comparación Perceptrón vs MLP en XOR ─────────────────────
    st.markdown("---")
    tab1, tab2 = st.tabs(["⚔️ Perceptrón vs MLP en XOR", "🎬 Evolución epoch a epoch"])

    with tab1:
        _comparacion_perceptron_vs_mlp()

    with tab2:
        _evolucion_epoch_a_epoch()


def _comparacion_perceptron_vs_mlp():
    st.markdown("""
    ### ¿Puede un perceptrón simple resolver XOR?

    Un **perceptrón simple** (sin capas ocultas) solo puede trazar una línea recta.
    El XOR es imposible para él.

    Un **MLP con una capa oculta** transforma el espacio y sí puede resolverlo.
    Entrenemos ambos y comparemos.
    """)

    X_xor, y_xor = generar_xor(n=200, seed=0)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔴 Entrenar Perceptrón en XOR", use_container_width=True, key="btn_pct"):
            model_pct = MLP(layer_sizes=[2, 1], activation="sigmoid", lr=0.5)
            model_pct.train(X_xor, y_xor, epochs=500)
            st.session_state["model_pct"] = model_pct
            st.session_state["X_xor_cmp"] = X_xor
            st.session_state["y_xor_cmp"] = y_xor

    with col_btn2:
        if st.button("🟢 Entrenar MLP en XOR", use_container_width=True, key="btn_mlp"):
            model_mlp = MLP(layer_sizes=[2, 8, 8, 1], activation="relu", lr=0.3)
            model_mlp.train(X_xor, y_xor, epochs=800)
            st.session_state["model_mlp_cmp"] = model_mlp
            st.session_state["X_xor_cmp"] = X_xor
            st.session_state["y_xor_cmp"] = y_xor

    X_plot = st.session_state.get("X_xor_cmp", X_xor)
    y_plot = st.session_state.get("y_xor_cmp", y_xor)

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("#### Perceptrón (2→1)")
        if "model_pct" in st.session_state:
            m = st.session_state["model_pct"]
            preds = m.predict_classes(X_plot)
            acc = np.mean(preds == y_plot.astype(int)) * 100
            fig = grafico_frontera(m, X_plot, y_plot,
                                   titulo=f"Perceptrón — Accuracy: {acc:.1f}%")
            st.plotly_chart(fig, use_container_width=True)
            if acc < 60:
                st.error(f"❌ Accuracy: {acc:.1f}% — No puede resolver XOR con una línea.")
            else:
                st.warning(f"⚠️ Accuracy: {acc:.1f}% — Resultado mediocre.")
        else:
            st.info("Hacé clic en 'Entrenar Perceptrón' para ver la frontera.")

    with col_v2:
        st.markdown("#### MLP (2→6→6→1)")
        if "model_mlp_cmp" in st.session_state:
            m = st.session_state["model_mlp_cmp"]
            preds = m.predict_classes(X_plot)
            acc = np.mean(preds == y_plot.astype(int)) * 100
            fig = grafico_frontera(m, X_plot, y_plot,
                                   titulo=f"MLP — Accuracy: {acc:.1f}%")
            st.plotly_chart(fig, use_container_width=True)
            if acc > 90:
                st.success(f"✅ Accuracy: {acc:.1f}% — El MLP resuelve XOR perfectamente.")
            else:
                st.warning(f"⚠️ Accuracy: {acc:.1f}% — Entrenando más epochs mejoraría.")
        else:
            st.info("Hacé clic en 'Entrenar MLP' para ver la frontera.")

    # Explicación geométrica
    with st.expander("📐 ¿Por qué el perceptrón no puede con XOR? Explicación geométrica"):
        st.markdown("""
        Los cuatro puntos del XOR son: (+,+)=0, (-,-)=0, (+,-)=1, (-,+)=1.

        Si intentás trazar una línea recta que separe los ceros de los unos,
        vas a notar que **siempre queda al menos un punto del lado equivocado**.

        Es matemáticamente imposible: los puntos de las dos clases están
        en **cuadrantes opuestos** y ninguna línea puede aislarlos.

        El MLP resuelve esto **transformando el espacio** en las capas ocultas.
        Imagina que "dobla y estira" el plano hasta que los grupos queden
        de lados distintos, y entonces la capa final puede trazar la separación.
        """)


def _evolucion_epoch_a_epoch():
    st.markdown("""
    ### Mirá cómo evoluciona la frontera mientras la red aprende

    Entrenamos el MLP y guardamos cómo estaba la frontera en distintos momentos.
    Usá el slider para navegar por el tiempo del entrenamiento.
    """)

    col_p, col_g = st.columns([1, 2])
    with col_p:
        dataset_ev = st.selectbox("Dataset", ["XOR", "Círculos"], key="ds_ev")
        n_ocultas_ev = st.slider("Neuronas ocultas", 2, 12, 6, key="noc_ev")
        lr_ev = st.select_slider("Learning rate",
                                  options=[0.01, 0.05, 0.1, 0.3, 0.5],
                                  value=0.1, key="lr_ev")
        epochs_ev = st.slider("Epochs totales", 100, 1000, 400, 100, key="ep_ev")
        btn_ev = st.button("▶ Entrenar y ver evolución", type="primary",
                           use_container_width=True, key="btn_ev")

    with col_g:
        if btn_ev:
            if dataset_ev == "XOR":
                X_ev, y_ev = generar_xor(n=200, seed=1)
            else:
                X_ev, y_ev = generar_circulos(n=200, seed=1)

            model_ev = MLP(
                layer_sizes=[2, n_ocultas_ev, n_ocultas_ev, 1],
                activation="relu", lr=lr_ev
            )
            with st.spinner("Entrenando..."):
                model_ev.train(X_ev, y_ev, epochs=epochs_ev, snapshot_every=max(1, epochs_ev // 20))

            st.session_state["model_ev"] = model_ev
            st.session_state["X_ev"] = X_ev
            st.session_state["y_ev"] = y_ev

        if "model_ev" in st.session_state:
            model_ev = st.session_state["model_ev"]
            X_ev = st.session_state["X_ev"]
            y_ev = st.session_state["y_ev"]
            snapshots = model_ev.weight_snapshots

            if snapshots:
                idx = st.slider(
                    "Momento del entrenamiento",
                    0, len(snapshots) - 1,
                    len(snapshots) - 1,
                    key="snap_slider",
                    format=f"Snapshot %d"
                )
                snap = snapshots[idx]

                # Crear modelo temporal con los pesos del snapshot
                temp = MLP(
                    layer_sizes=model_ev.layer_sizes,
                    activation=model_ev.activation_name,
                    lr=model_ev.lr,
                )
                temp.weights = [w.copy() for w in snap["weights"]]
                temp.biases  = [b.copy() for b in snap["biases"]]

                fig_ev = grafico_frontera(
                    temp, X_ev, y_ev,
                    titulo="Evolución de la frontera",
                    epoch=snap["epoch"],
                    accuracy=snap.get("accuracy"),
                )
                st.plotly_chart(fig_ev, use_container_width=True)

                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Epoch", snap["epoch"])
                with col_s2:
                    st.metric("Loss", f"{snap['loss']:.4f}")
                with col_s3:
                    st.metric("Accuracy", f"{snap.get('accuracy', 0)*100:.1f}%")
        else:
            st.info("👆 Configurá y hacé clic en 'Entrenar y ver evolución'")

    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem; border-radius: 12px; border: 1px solid #F59E0B;'>
    <h3 style='color:#E2E8F0;'>🎯 ¿Qué aprendiste en este acto?</h3>
    <ul style='color:#CBD5E1;'>
        <li>La frontera de decisión cambia con cada epoch de entrenamiento</li>
        <li>El perceptrón solo traza líneas rectas: no puede resolver XOR</li>
        <li>El MLP "dobla el espacio" para separar clases no lineales</li>
        <li>Más epochs = frontera más precisa (hasta cierto punto)</li>
    </ul>
    <p style='color:#94A3B8; margin-top:1rem;'>
        → En el último acto podés experimentar libremente con todos los parámetros.
    </p>
    </div>
    """, unsafe_allow_html=True)
