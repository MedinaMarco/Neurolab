"""
boundary.py
Visualización de la frontera de decisión del MLP con Plotly.
Permite navegar entre epochs usando snapshots del modelo.
"""
import numpy as np
import plotly.graph_objects as go


def _meshgrid_prediction(predict_fn, X: np.ndarray, resolution: int = 120):
    """Genera la grilla de predicciones para colorear el fondo."""
    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    Z = predict_fn(grid).reshape(xx.shape)
    return xx, yy, Z


def grafico_frontera(model, X: np.ndarray, y: np.ndarray,
                     titulo: str = "Frontera de decisión",
                     epoch: int = None,
                     accuracy: float = None) -> go.Figure:
    """
    Dibuja la frontera de decisión del modelo sobre el dataset X, y.
    El fondo muestra la probabilidad predicha (azul = clase 0, verde = clase 1).
    """
    xx, yy, Z = _meshgrid_prediction(model.predict, X)

    fig = go.Figure()

    # ── Fondo de colores (heatmap de probabilidad) ──
    fig.add_trace(go.Contour(
        x=np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 120),
        y=np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 120),
        z=Z,
        colorscale=[[0, "#1E3A5F"], [0.5, "#1E293B"], [1, "#1A3A1A"]],
        showscale=False,
        opacity=0.9,
        contours=dict(coloring="fill"),
        line=dict(width=0),
        hoverinfo="skip",
    ))

    # ── Línea de frontera (contorno en 0.5) ──
    fig.add_trace(go.Contour(
        x=np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 120),
        y=np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 120),
        z=Z,
        showscale=False,
        contours=dict(
            start=0.5, end=0.5, size=0,
            coloring="lines",
        ),
        line=dict(color="#F59E0B", width=3),
        hoverinfo="skip",
        name="Frontera (p=0.5)",
    ))

    # ── Puntos del dataset ──
    mask0 = y == 0
    mask1 = y == 1
    for mask, color, name, symbol in [
        (mask0, "#3B82F6", "Clase 0", "circle"),
        (mask1, "#10B981", "Clase 1", "diamond"),
    ]:
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1],
            mode="markers",
            name=name,
            marker=dict(color=color, size=8, symbol=symbol,
                        line=dict(color="white", width=1)),
            hovertemplate=f"({name})<br>x₁=%{{x:.2f}}, x₂=%{{y:.2f}}<extra></extra>",
        ))

    # ── Título dinámico ──
    titulo_completo = titulo
    if epoch is not None:
        titulo_completo += f"  —  Epoch {epoch}"
    if accuracy is not None:
        titulo_completo += f"  —  Accuracy: {accuracy*100:.1f}%"

    fig.update_layout(
        title=dict(text=titulo_completo, font=dict(color="#E2E8F0", size=14)),
        xaxis=dict(title="x₁", color="#94A3B8", gridcolor="#1E293B", zeroline=False),
        yaxis=dict(title="x₂", color="#94A3B8", gridcolor="#1E293B", zeroline=False),
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        font=dict(color="#E2E8F0"),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155", borderwidth=1),
        height=420,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


def grafico_frontera_snapshot(mlp_class, snapshot: dict, X: np.ndarray, y: np.ndarray) -> go.Figure:
    """
    Dibuja la frontera para un snapshot específico (para la animación por epochs).
    Crea una instancia temporal del modelo con los pesos guardados.
    """
    temp_model = mlp_class.__class__(
        layer_sizes=mlp_class.layer_sizes,
        activation=mlp_class.activation_name,
        lr=mlp_class.lr,
    )
    temp_model.weights = [w.copy() for w in snapshot["weights"]]
    temp_model.biases  = [b.copy() for b in snapshot["biases"]]

    return grafico_frontera(
        temp_model, X, y,
        titulo="Evolución de la frontera",
        epoch=snapshot["epoch"],
        accuracy=snapshot.get("accuracy"),
    )
