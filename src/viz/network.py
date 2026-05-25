"""
network.py
Diagrama de arquitectura de la red neuronal con Matplotlib.
Colores de las conexiones proporcionales al peso real.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def diagrama_red(layer_sizes: list, weights: list = None, highlight_forward: bool = False) -> plt.Figure:
    """
    Dibuja el grafo de la red neuronal.

    layer_sizes : [n_input, n_hidden1, ..., n_output]
    weights     : lista de matrices de pesos (opcional, colorea las conexiones)
    """
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 2.5), max(5, max_neurons * 0.9)))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    neuron_positions = []
    layer_labels = (
        ["Entrada"] +
        [f"Capa oculta {i+1}" for i in range(n_layers - 2)] +
        ["Salida"]
    )

    # Posiciones de cada neurona
    for l, n in enumerate(layer_sizes):
        x = l / (n_layers - 1)
        ys = np.linspace(0.1, 0.9, n) if n > 1 else [0.5]
        neuron_positions.append(list(zip([x] * n, ys)))

    # Normalizador de pesos para colorear las conexiones
    if weights is not None:
        all_w = np.concatenate([w.flatten() for w in weights])
        vmax = max(abs(all_w).max(), 0.01)
        norm = Normalize(vmin=-vmax, vmax=vmax)
        cmap = plt.cm.RdYlGn
    else:
        norm, cmap = None, None

    # Dibujar conexiones (de atrás hacia adelante para que neuronas queden encima)
    for l in range(n_layers - 1):
        for i, (x1, y1) in enumerate(neuron_positions[l]):
            for j, (x2, y2) in enumerate(neuron_positions[l + 1]):
                if weights is not None:
                    w_val = weights[l][i, j]
                    color = cmap(norm(w_val))
                    alpha = 0.3 + 0.7 * abs(w_val) / vmax
                    lw = 0.5 + 2.5 * abs(w_val) / vmax
                else:
                    color = "#334155"
                    alpha = 0.5
                    lw = 1.0
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=lw, zorder=1)

    # Dibujar neuronas
    for l, (positions, label) in enumerate(zip(neuron_positions, layer_labels)):
        if l == 0:
            neuron_color = "#1E3A5F"
            edge_color   = "#3B82F6"
        elif l == n_layers - 1:
            neuron_color = "#1A2E1A"
            edge_color   = "#10B981"
        else:
            neuron_color = "#2D1B4E"
            edge_color   = "#7C3AED"

        for idx, (x, y) in enumerate(positions):
            circle = plt.Circle((x, y), 0.035, color=neuron_color,
                                  ec=edge_color, linewidth=2, zorder=2)
            ax.add_patch(circle)

        # Etiqueta de la capa
        mid_y = np.mean([p[1] for p in positions])
        ax.text(positions[0][0], -0.02, label,
                ha="center", va="top", fontsize=9, color="#94A3B8", fontweight="bold")

        # Número de neuronas
        ax.text(positions[0][0], 0.97,
                f"{layer_sizes[l]} {'entrada' if layer_sizes[l]==1 else 'neuronas' if l > 0 and l < n_layers-1 else 'entradas' if l==0 else 'salida'}",
                ha="center", va="bottom", fontsize=8, color="#64748B")

    # Colorbar de pesos
    if weights is not None:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.08,
                            fraction=0.03, shrink=0.4)
        cbar.set_label("Valor del peso", color="#94A3B8", fontsize=9)
        cbar.ax.xaxis.set_tick_params(color="#94A3B8")
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color="#94A3B8", fontsize=8)
        cbar.outline.set_edgecolor("#334155")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.05)
    ax.axis("off")

    # Leyenda de colores de capas
    patches = [
        mpatches.Patch(color="#3B82F6", label="Capa de entrada"),
        mpatches.Patch(color="#7C3AED", label="Capa(s) oculta(s)"),
        mpatches.Patch(color="#10B981", label="Capa de salida"),
    ]
    ax.legend(handles=patches, loc="upper right", framealpha=0.2,
              labelcolor="white", facecolor="#1E293B", edgecolor="#334155", fontsize=8)

    plt.tight_layout()
    return fig
