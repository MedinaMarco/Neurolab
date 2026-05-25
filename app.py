"""
app.py — NeuroLab: Aprendé cómo aprende una máquina
Punto de entrada principal. Ejecutar con: streamlit run app.py
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="NeuroLab — Tutorial Interactivo de Redes Neuronales",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "NeuroLab — TP Integrador. MLP desde cero con NumPy."},
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0F0F1A; }
    ::-webkit-scrollbar-thumb { background: #4C1D95; border-radius: 3px; }
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7C3AED, #4C1D95);
        border: none; font-weight: bold;
    }
    [data-testid="metric-container"] {
        background: #1E293B; padding: 0.8rem;
        border-radius: 8px; border: 1px solid #334155;
    }
    [data-testid="stSidebar"] {
        background-color: #0A0A14;
        border-right: 1px solid #1E293B;
    }
    hr { border-color: #1E293B; }
</style>
""", unsafe_allow_html=True)

from src.pages.acto1 import render as render_acto1
from src.pages.acto2 import render as render_acto2
from src.pages.acto3 import render as render_acto3
from src.pages.acto4 import render as render_acto4
from src.pages.acto5 import render as render_acto5
from src.pages.acto6 import render as render_acto6
from src.pages.glosario import render as render_glosario, render_glosario_sidebar

DESCRIPTIONS = {
    "acto1":    "Intentás separar datos con una línea. Descubrís por qué no siempre alcanza.",
    "acto2":    "Conocés el bloque fundamental: la neurona y cómo decide.",
    "acto3":    "Entendés cómo se conectan las capas y qué es el forward pass.",
    "acto4":    "Mirás cómo la red mide y corrige sus errores con backpropagation.",
    "acto5":    "Visualizás la frontera de decisión evolucionando epoch a epoch.",
    "acto6":    "Controlás todos los parámetros y entrenás tu propia red.",
    "glosario": "Todos los conceptos clave explicados con analogías y fórmulas.",
}

PAGES = {
    "🏠  Inicio":                         "inicio",
    "⚡  Acto 1 — El Problema":            "acto1",
    "🧠  Acto 2 — La Neurona":             "acto2",
    "🕸️  Acto 3 — La Red":                "acto3",
    "📉  Acto 4 — El Error y Backprop":    "acto4",
    "🗺️  Acto 5 — La Frontera":           "acto5",
    "🎮  Acto 6 — Tu Turno":              "acto6",
    "📖  Glosario Completo":              "glosario",
}

RENDER_FNS = {
    "acto1": render_acto1, "acto2": render_acto2, "acto3": render_acto3,
    "acto4": render_acto4, "acto5": render_acto5, "acto6": render_acto6,
    "glosario": render_glosario,
}


def render_inicio():
    st.markdown("""
    <div style='text-align:center; padding:3rem 1rem 2rem 1rem;'>
        <div style='font-size:5rem; margin-bottom:1rem;'>🧠</div>
        <h1 style='color:#E2E8F0; font-size:3rem; margin:0;'>NeuroLab</h1>
        <p style='color:#7C3AED; font-size:1.3rem; margin:0.5rem 0 1.5rem 0;
                  font-weight:500; letter-spacing:1px;'>
            APRENDÉ CÓMO APRENDE UNA MÁQUINA
        </p>
        <p style='color:#94A3B8; font-size:1.1rem; max-width:600px;
                  margin:0 auto 2rem auto; line-height:1.6;'>
            Un recorrido interactivo desde el problema más simple hasta entender
            cómo una red neuronal aprende con backpropagation.
            <b style='color:#E2E8F0;'>Sin conocimientos previos necesarios.</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🗺️ El recorrido")
    cols = st.columns(3)
    actos_info = [
        ("acto1", "⚡", "El Problema",        "#EF4444"),
        ("acto2", "🧠", "La Neurona",          "#3B82F6"),
        ("acto3", "🕸️", "La Red",              "#10B981"),
        ("acto4", "📉", "El Error y Backprop", "#F59E0B"),
        ("acto5", "🗺️", "La Frontera",         "#7C3AED"),
        ("acto6", "🎮", "Tu Turno",            "#A78BFA"),
    ]
    for i, (acto_id, emoji, titulo, color) in enumerate(actos_info):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background:#1E293B; padding:1.2rem; border-radius:12px;
                        border-top:3px solid {color}; margin-bottom:1rem; min-height:130px;'>
                <div style='font-size:1.8rem;'>{emoji}</div>
                <h4 style='color:{color}; margin:0.5rem 0 0.3rem 0;'>{titulo}</h4>
                <p style='color:#94A3B8; font-size:0.85rem; margin:0;'>
                    {DESCRIPTIONS[acto_id]}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### 🛠️ Stack tecnológico
        <div style='background:#1E293B; padding:1rem; border-radius:8px;'>
        <ul style='color:#CBD5E1; margin:0;'>
            <li>🔢 <b>NumPy</b> — MLP implementado desde cero</li>
            <li>📊 <b>Plotly</b> — visualizaciones interactivas</li>
            <li>📈 <b>Matplotlib</b> — diagramas de arquitectura</li>
            <li>🐼 <b>Pandas</b> — tablas de predicciones</li>
            <li>🌐 <b>Streamlit</b> — interfaz web</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        ### 📚 Lo que vas a aprender
        <div style='background:#1E293B; padding:1rem; border-radius:8px;'>
        <ul style='color:#CBD5E1; margin:0;'>
            <li>Qué es una neurona artificial y cómo funciona</li>
            <li>Cómo se conectan las capas en un MLP</li>
            <li>Qué es el forward pass y el backward pass</li>
            <li>Cómo funciona el backpropagation matemáticamente</li>
            <li>Qué es el gradient descent y la tasa de aprendizaje</li>
            <li>Por qué el MLP puede resolver lo que el perceptrón no puede</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:0.5rem;'>
        <p style='color:#64748B; font-size:0.9rem;'>
            👈 Usá el menú de la izquierda para navegar, o empezá por el
            <b style='color:#7C3AED;'>⚡ Acto 1 — El Problema</b>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem 0;'>
        <div style='font-size:2.5rem;'>🧠</div>
        <h2 style='color:#E2E8F0; margin:0; font-size:1.3rem;'>NeuroLab</h2>
        <p style='color:#64748B; font-size:0.8rem; margin:0.3rem 0;'>
            Tutorial interactivo de redes neuronales
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page_label = st.radio("Navegación", list(PAGES.keys()),
                           key="nav", label_visibility="collapsed")
    st.markdown("---")
    render_glosario_sidebar()
    st.markdown("---")
    st.markdown("""
    <div style='color:#374151; font-size:0.75rem; text-align:center; padding:0.5rem;'>
    TP Integrador — Redes Neuronales<br>
    Python · NumPy · Streamlit · Plotly · Pandas
    </div>
    """, unsafe_allow_html=True)

# ── Enrutamiento principal ───────────────────────────────────────────────
page_id = PAGES[page_label]
if page_id == "inicio":
    render_inicio()
else:
    RENDER_FNS[page_id]()
