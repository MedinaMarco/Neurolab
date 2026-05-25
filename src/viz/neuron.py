"""
neuron.py
Diagrama interactivo de una neurona con Plotly.
"""
import numpy as np
import plotly.graph_objects as go
from src.model.activations import sigmoid, relu, tanh_fn


def diagrama_neurona(pesos: list, bias: float, entradas: list, activation: str = "sigmoid") -> go.Figure:
    """
    Dibuja una neurona interactiva.
    Muestra: entradas → pesos → suma → activación → salida.
    """
    act_fns = {"sigmoid": sigmoid, "relu": relu, "tanh": tanh_fn}
    fn = act_fns[activation]

    n = len(pesos)
    z = sum(w * x for w, x in zip(pesos, entradas)) + bias
    a = float(fn(np.array([z]))[0])

    fig = go.Figure()

    # ── Nodos de entrada ──
    y_positions = np.linspace(0.8, 0.2, n) if n > 1 else [0.5]
    input_x, input_y = 0.1, y_positions

    colores_peso = ["#7C3AED" if w >= 0 else "#EF4444" for w in pesos]

    for i, (y_pos, w, x_val) in enumerate(zip(input_y, pesos, entradas)):
        # Nodo de entrada
        fig.add_shape(type="circle", x0=input_x - 0.04, y0=y_pos - 0.04,
                      x1=input_x + 0.04, y1=y_pos + 0.04,
                      fillcolor="#1E293B", line=dict(color="#94A3B8", width=2))

        # Etiqueta valor de entrada
        fig.add_annotation(x=input_x, y=y_pos, text=f"x{i+1} = {x_val:.2f}",
                           showarrow=False, font=dict(size=11, color="#E2E8F0"),
                           xanchor="center")

        # Línea de conexión (grosor proporcional al peso)
        abs_w = abs(w)
        width_line = 1 + abs_w * 4
        fig.add_shape(type="line", x0=input_x + 0.04, y0=y_pos, x1=0.46, y1=0.5,
                      line=dict(color=colores_peso[i], width=width_line))

        # Etiqueta del peso
        mid_x = (input_x + 0.04 + 0.46) / 2 + 0.02
        mid_y = (y_pos + 0.5) / 2
        fig.add_annotation(x=mid_x, y=mid_y,
                           text=f"w{i+1}={w:.2f}",
                           showarrow=False,
                           font=dict(size=10, color=colores_peso[i]),
                           bgcolor="rgba(15,15,26,0.8)",
                           bordercolor=colores_peso[i])

    # ── Neurona (círculo central) ──
    fig.add_shape(type="circle", x0=0.42, y0=0.35, x1=0.58, y1=0.65,
                  fillcolor="#4C1D95", line=dict(color="#7C3AED", width=3))

    # Fórmula dentro de la neurona
    formula_text = f"z={z:.2f}"
    fig.add_annotation(x=0.5, y=0.52, text=formula_text,
                       showarrow=False, font=dict(size=11, color="#DDD6FE"), xanchor="center")
    fig.add_annotation(x=0.5, y=0.44, text=f"{activation}",
                       showarrow=False, font=dict(size=10, color="#A78BFA"), xanchor="center")

    # ── Línea hacia la salida ──
    fig.add_shape(type="line", x0=0.58, y0=0.5, x1=0.86, y1=0.5,
                  line=dict(color="#7C3AED", width=3))

    # Bias
    fig.add_annotation(x=0.5, y=0.22,
                       text=f"bias b = {bias:.2f}",
                       showarrow=True, ax=0.5, ay=0.33,
                       arrowhead=2, arrowcolor="#F59E0B",
                       font=dict(size=11, color="#FCD34D"),
                       bgcolor="rgba(15,15,26,0.9)")

    # ── Nodo de salida ──
    color_salida = "#10B981" if a >= 0.5 else "#EF4444"
    fig.add_shape(type="circle", x0=0.86, y0=0.42, x1=0.96, y1=0.58,
                  fillcolor="#0F172A", line=dict(color=color_salida, width=3))
    fig.add_annotation(x=0.91, y=0.5, text=f"a={a:.3f}",
                       showarrow=False, font=dict(size=11, color=color_salida), xanchor="center")

    # ── Fórmula completa debajo ──
    suma_str = " + ".join([f"{w:.2f}·{x:.2f}" for w, x in zip(pesos, entradas)])
    fig.add_annotation(x=0.5, y=0.08,
                       text=f"z = {suma_str} + {bias:.2f} = {z:.3f}   →   a = {activation}({z:.3f}) = {a:.3f}",
                       showarrow=False, font=dict(size=11, color="#94A3B8"),
                       xanchor="center")

    fig.update_layout(
        width=700, height=350,
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=False,
    )
    return fig


def grafico_activacion(activation: str) -> go.Figure:
    """Gráfico de la función de activación elegida."""
    z_range = np.linspace(-5, 5, 300)
    act_fns = {"sigmoid": sigmoid, "relu": relu, "tanh": tanh_fn}
    a_vals = act_fns[activation](z_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=z_range, y=a_vals,
        mode="lines",
        line=dict(color="#7C3AED", width=3),
        name=activation
    ))
    fig.add_hline(y=0, line=dict(color="#475569", width=1, dash="dash"))
    fig.add_vline(x=0, line=dict(color="#475569", width=1, dash="dash"))

    fig.update_layout(
        title=dict(text=f"Función de activación: {activation}", font=dict(color="#E2E8F0")),
        xaxis=dict(title="z (entrada)", color="#94A3B8", gridcolor="#1E293B"),
        yaxis=dict(title="a (salida)", color="#94A3B8", gridcolor="#1E293B"),
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#E2E8F0"),
        height=280,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig
