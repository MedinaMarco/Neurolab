"""
curves.py
Curvas de pérdida y accuracy con Plotly.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def grafico_curvas(history: dict) -> go.Figure:
    """
    Gráfico combinado: curva de pérdida (loss) + curva de accuracy.
    """
    epochs = history["epochs"]
    loss   = history["loss"]
    acc    = history["accuracy"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Pérdida (Loss)", "Exactitud (Accuracy)")
    )

    # ── Curva de pérdida ──
    fig.add_trace(go.Scatter(
        x=epochs, y=loss,
        mode="lines",
        name="Loss",
        line=dict(color="#EF4444", width=2.5),
        hovertemplate="Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # ── Curva de accuracy ──
    fig.add_trace(go.Scatter(
        x=epochs, y=[a * 100 for a in acc],
        mode="lines",
        name="Accuracy",
        line=dict(color="#10B981", width=2.5),
        hovertemplate="Epoch %{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
    ), row=1, col=2)

    # Referencia 100% en accuracy
    fig.add_hline(y=100, line=dict(color="#334155", dash="dot", width=1),
                  row=1, col=2)

    # Valor final de loss mínimo
    min_loss = min(loss)
    min_epoch = loss.index(min_loss)
    fig.add_annotation(
        x=min_epoch, y=min_loss,
        text=f"Mínimo: {min_loss:.4f}",
        showarrow=True, arrowhead=2,
        arrowcolor="#F59E0B",
        font=dict(color="#FCD34D", size=10),
        ax=40, ay=-30,
        row=1, col=1
    )

    fig.update_layout(
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#E2E8F0"),
        showlegend=False,
        height=320,
        margin=dict(l=50, r=30, t=50, b=50),
    )
    for axis in ["xaxis", "xaxis2"]:
        fig.update_layout(**{axis: dict(
            title="Epoch", color="#94A3B8", gridcolor="#1E293B", zeroline=False
        )})
    fig.update_layout(
        yaxis=dict(title="Pérdida", color="#94A3B8", gridcolor="#1E293B"),
        yaxis2=dict(title="Accuracy (%)", color="#94A3B8", gridcolor="#1E293B"),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#94A3B8"

    return fig


def grafico_backprop_flujo(n_capas: int = 3) -> go.Figure:
    """
    Diagrama simplificado del flujo forward → backward en la red.
    Flecha verde = forward, flecha roja = backward (gradiente).
    """
    fig = go.Figure()
    xs = [i / (n_capas - 1) for i in range(n_capas)]
    labels = (["Entrada"] + [f"Oculta {i+1}" for i in range(n_capas - 2)] + ["Salida"])

    for i, (x, label) in enumerate(zip(xs, labels)):
        color = "#3B82F6" if i == 0 else ("#10B981" if i == n_capas - 1 else "#7C3AED")
        fig.add_shape(type="circle", x0=x - 0.04, y0=0.46, x1=x + 0.04, y1=0.54,
                      fillcolor="#0F172A", line=dict(color=color, width=3))
        fig.add_annotation(x=x, y=0.3, text=label,
                           showarrow=False, font=dict(size=10, color="#94A3B8"))

    # Flechas forward (verde)
    for i in range(n_capas - 1):
        fig.add_annotation(
            x=xs[i + 1] - 0.05, y=0.62,
            ax=xs[i] + 0.05, ay=0.62,
            arrowhead=2, arrowcolor="#10B981", arrowwidth=2,
            text="", showarrow=True
        )

    # Flechas backward (rojo)
    for i in range(n_capas - 2, 0, -1):
        fig.add_annotation(
            x=xs[i - 1] + 0.05, y=0.38,
            ax=xs[i] - 0.05, ay=0.38,
            arrowhead=2, arrowcolor="#EF4444", arrowwidth=2,
            text="", showarrow=True
        )

    fig.add_annotation(x=0.5, y=0.72, text="→ Forward pass (predicción)",
                       showarrow=False, font=dict(size=11, color="#10B981"))
    fig.add_annotation(x=0.5, y=0.28, text="← Backward pass (gradientes)",
                       showarrow=False, font=dict(size=11, color="#EF4444"))

    # Pérdida a la derecha
    fig.add_annotation(x=1.15, y=0.5, text="📉 Loss",
                       showarrow=False, font=dict(size=13, color="#F59E0B"))

    fig.update_layout(
        paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
        xaxis=dict(visible=False, range=[-0.15, 1.3]),
        yaxis=dict(visible=False, range=[0.15, 0.85]),
        height=200, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig
