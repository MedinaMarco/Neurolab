"""
glosario.py
Panel de glosario interactivo de conceptos clave.
Diseñado para ser llamado desde la barra lateral o como página separada.
"""
import streamlit as st


CONCEPTOS = {
    "Peso (w)": {
        "simbolo": "w",
        "definicion": "Un número que indica cuánta importancia le da la neurona a esa entrada. Pesos grandes = esa entrada importa mucho.",
        "formula": "z = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b",
        "bajo": "La entrada casi no influye en la decisión de la neurona.",
        "medio": "Influencia moderada. La entrada cuenta pero no domina.",
        "alto": "Esa entrada domina la decisión de la neurona.",
        "analogia": "El volumen del micrófono de cada juez en un panel de evaluación.",
        "tip": "Pesos muy grandes pueden hacer que el modelo 'sobrereaccione'. La regularización ayuda a controlarlos.",
        "color": "#7C3AED",
    },
    "Bias (b)": {
        "simbolo": "b",
        "definicion": "Un valor extra que suma la neurona independientemente de las entradas. Ajusta el punto de activación.",
        "formula": "z = Σ wᵢ·xᵢ + b",
        "bajo": "La neurona necesita señales muy fuertes para activarse (exigente).",
        "medio": "Comportamiento neutro.",
        "alto": "La neurona se activa fácilmente, incluso con entradas débiles (permisiva).",
        "analogia": "El nivel de exigencia del jurado: un jurado exigente (bias negativo) aprueba muy poco.",
        "tip": "Sin bias, la frontera de decisión siempre pasa por el origen, lo cual es una limitación enorme.",
        "color": "#F59E0B",
    },
    "Función de Activación": {
        "simbolo": "f(z)",
        "definicion": "Una función que transforma la suma ponderada antes de pasar al siguiente nivel. Introduce la no-linealidad que permite resolver problemas complejos.",
        "formula": "a = f(z)  (ej: sigmoid, ReLU, tanh)",
        "bajo": "Sin activación (lineal), la red solo puede aprender fronteras rectas.",
        "medio": "Con ReLU o sigmoid, la red puede aprender curvas y patrones complejos.",
        "alto": "Activaciones profundas apiladas crean representaciones muy ricas.",
        "analogia": "Un filtro de Instagram que transforma la imagen antes de pasarla al siguiente paso.",
        "tip": "ReLU es la más usada en capas ocultas. Sigmoid solo en la capa de salida para clasificación binaria.",
        "color": "#3B82F6",
    },
    "Epoch": {
        "simbolo": "T",
        "definicion": "Una pasada completa por todo el dataset de entrenamiento. En cada epoch, la red ve todos los datos una vez y actualiza sus pesos.",
        "formula": "for t in range(epochs): forward → loss → backward → update",
        "bajo": "La red todavía no aprendió suficiente. Underfitting.",
        "medio": "La red está aprendiendo bien.",
        "alto": "Riesgo de sobreajuste (overfitting): memorizar los datos en lugar de generalizarlos.",
        "analogia": "Un estudiante que repite el mismo ejercicio muchas veces para aprenderlo.",
        "tip": "Usá validación cruzada para saber cuándo parar. Más no siempre es mejor.",
        "color": "#10B981",
    },
    "Learning Rate (lr)": {
        "simbolo": "α",
        "definicion": "El tamaño del paso que da el modelo al actualizar los pesos. Controla qué tan rápido aprende.",
        "formula": "w ← w − α · ∂L/∂w",
        "bajo": "Aprende muy lento. Puede tardar miles de epochs en converger.",
        "medio": "Converge suavemente hacia el mínimo. El valor óptimo para la mayoría de los casos.",
        "alto": "Los pasos son demasiado grandes. El modelo puede diverger (loss que sube en lugar de bajar).",
        "analogia": "El tamaño del paso al bajar una montaña con los ojos vendados. Muy grande = caés; muy pequeño = tardás horas.",
        "tip": "Valores comunes: 0.001 a 0.1. Empezá con 0.01 y ajustá según el comportamiento del loss.",
        "color": "#EF4444",
    },
    "Loss (Pérdida)": {
        "simbolo": "L",
        "definicion": "Una medida de qué tan equivocada está la predicción. Es el número que el modelo intenta reducir con cada epoch.",
        "formula": "L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]  (BCE)",
        "bajo": "La red predice bien. Modelo bien entrenado.",
        "medio": "Hay margen de mejora.",
        "alto": "La red se equivoca mucho. Necesita más entrenamiento o mejor arquitectura.",
        "analogia": "La diferencia entre el pronóstico del tiempo y la temperatura real: cuanto más grande, peor el modelo.",
        "tip": "Observá la curva de loss. Si baja suavemente, todo va bien. Si oscila, bajá el learning rate.",
        "color": "#EF4444",
    },
    "Backpropagation": {
        "simbolo": "∂L/∂w",
        "definicion": "Algoritmo que calcula el gradiente de la pérdida respecto a cada peso, usando la regla de la cadena del cálculo.",
        "formula": "δ_L = ŷ − y;  δ_l = (δ_{l+1}·Wᵀ) × f'(aₗ)",
        "bajo": "N/A — es un algoritmo, no un parámetro.",
        "medio": "N/A",
        "alto": "N/A",
        "analogia": "Después de un mal resultado, identificar quién tuvo la culpa y cuánto: el jefe de área, el empleado, el proveedor.",
        "tip": "Backprop puede tener el problema del 'vanishing gradient': en redes muy profundas, el gradiente se vuelve casi cero y los pesos no se actualizan. ReLU ayuda a mitigarlo.",
        "color": "#A78BFA",
    },
    "Gradient Descent": {
        "simbolo": "∇L",
        "definicion": "Método de optimización que actualiza los pesos en la dirección que más reduce el error.",
        "formula": "w ← w − lr · ∂L/∂w",
        "bajo": "N/A",
        "medio": "N/A",
        "alto": "N/A",
        "analogia": "Bajar una montaña a ciegas: sentís la pendiente (gradiente) y das un paso hacia abajo.",
        "tip": "Existen variantes más sofisticadas (Adam, RMSProp) que adaptan el learning rate automáticamente.",
        "color": "#06B6D4",
    },
    "Overfitting": {
        "simbolo": "—",
        "definicion": "Cuando el modelo memoriza los datos de entrenamiento en lugar de aprender patrones generales. Funciona bien en entrenamiento pero mal en datos nuevos.",
        "formula": "Train accuracy >> Test accuracy",
        "bajo": "Underfitting: el modelo no aprendió suficiente.",
        "medio": "Buena generalización: aprendió los patrones sin memorizar.",
        "alto": "Overfitting: memorizó los datos. Necesita regularización o menos capacidad.",
        "analogia": "Estudiar el examen anterior de memoria en lugar de entender la materia. Aprobás si te toman lo mismo, reprobás si cambian las preguntas.",
        "tip": "Prevención: más datos, dropout, regularización L2, early stopping, o reducir la arquitectura.",
        "color": "#F97316",
    },
}


def render_glosario_sidebar():
    """Versión compacta para la barra lateral."""
    st.sidebar.markdown("""
    <div style='background:#1E293B; padding:0.8rem; border-radius:8px; margin-bottom:0.5rem;'>
        <h3 style='color:#7C3AED; margin:0;'>📖 Glosario</h3>
    </div>
    """, unsafe_allow_html=True)

    concepto_sel = st.sidebar.selectbox(
        "Consultá un concepto",
        list(CONCEPTOS.keys()),
        key="glosario_sel",
    )

    if concepto_sel:
        c = CONCEPTOS[concepto_sel]
        st.sidebar.markdown(f"""
        <div style='background:#0F172A; padding:0.8rem; border-radius:8px;
                    border-left:3px solid {c["color"]}; margin-top:0.5rem;'>
            <div style='color:{c["color"]}; font-weight:bold; font-size:0.9rem;'>
                {concepto_sel} ({c["simbolo"]})
            </div>
            <p style='color:#CBD5E1; font-size:0.8rem; margin:0.5rem 0;'>
                {c["definicion"]}
            </p>
            <div style='background:#1E293B; padding:0.4rem; border-radius:4px;
                        font-family:monospace; font-size:0.75rem; color:#DDD6FE; margin:0.3rem 0;'>
                {c["formula"]}
            </div>
            <p style='color:#F59E0B; font-size:0.78rem; margin:0.3rem 0;'>
                💡 {c["analogia"]}
            </p>
            <p style='color:#64748B; font-size:0.75rem; margin:0.3rem 0;'>
                🛠️ {c["tip"]}
            </p>
        </div>
        """, unsafe_allow_html=True)


def render():
    """Versión completa como página."""
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 1.5rem 2rem; border-radius: 12px;
                border-left: 4px solid #7C3AED; margin-bottom: 1.5rem;'>
        <h1 style='color:#E2E8F0; margin:0;'>📖 Glosario de Conceptos</h1>
        <p style='color:#94A3B8; margin:0.5rem 0 0 0;'>
            Todos los términos clave explicados sin jerga innecesaria.
        </p>
    </div>
    """, unsafe_allow_html=True)

    for nombre, c in CONCEPTOS.items():
        with st.expander(f"**{nombre}** — {c['simbolo']}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"""
                <div style='background:#0F172A; padding:1rem; border-radius:8px;
                            border-left:3px solid {c["color"]};'>
                <p style='color:#E2E8F0; font-size:1rem;'>{c["definicion"]}</p>
                <div style='background:#1E293B; padding:0.6rem; border-radius:6px;
                            font-family:monospace; color:#DDD6FE; font-size:0.9rem; margin:0.5rem 0;'>
                    {c["formula"]}
                </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style='background:#1E293B; padding:0.8rem; border-radius:8px;'>
                <p style='color:#F59E0B; font-size:0.85rem;'>
                    <b>🎯 Analogía:</b><br>{c["analogia"]}
                </p>
                <p style='color:#10B981; font-size:0.85rem; margin-top:0.5rem;'>
                    <b>🔼 Valor bajo:</b> {c["bajo"]}
                </p>
                <p style='color:#F59E0B; font-size:0.85rem;'>
                    <b>➡️ Valor medio:</b> {c["medio"]}
                </p>
                <p style='color:#EF4444; font-size:0.85rem;'>
                    <b>🔽 Valor alto:</b> {c["alto"]}
                </p>
                <p style='color:#7C3AED; font-size:0.85rem; margin-top:0.5rem;'>
                    <b>💡 Tip:</b> {c["tip"]}
                </p>
                </div>
                """, unsafe_allow_html=True)
