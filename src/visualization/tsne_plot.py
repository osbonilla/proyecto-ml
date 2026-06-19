"""
Visualización T-SNE del espacio de características TF-IDF.
Reduce embeddings de alta dimensión a 2D para análisis visual de separabilidad de clases.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, List

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder


# Paleta de colores para los 8 intents
INTENT_COLORS = [
    "#E63946", "#2196F3", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
]

INTENT_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


def compute_tsne(
    X_sparse,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    use_svd_first: bool = True,
    svd_components: int = 50,
) -> np.ndarray:
    """
    Aplica T-SNE para reducción de dimensionalidad a 2D.

    Nota sobre el pipeline de reducción:
        TF-IDF → TruncatedSVD (50D) → T-SNE (2D)

        ¿Por qué SVD antes de T-SNE?
        - T-SNE es O(n²) en dimensionalidad → muy lento con 5000 features
        - TruncatedSVD (LSA) reduce a 50D conservando varianza semántica
        - Combinación estándar en NLP para visualización

    Args:
        X_sparse: Matriz TF-IDF sparse (n_samples, n_features)
        n_components: Dimensiones de salida (default=2)
        perplexity: Parámetro T-SNE (balance local/global). Rango recomendado: 5-50
        n_iter: Iteraciones de optimización T-SNE
        random_state: Semilla de reproducibilidad
        use_svd_first: Si True, aplica SVD antes de T-SNE
        svd_components: Componentes para TruncatedSVD

    Returns:
        Array (n_samples, 2) con coordenadas 2D
    """
    print("🔄 Aplicando reducción de dimensionalidad...")

    if use_svd_first:
        n_comp = min(svd_components, X_sparse.shape[1] - 1, X_sparse.shape[0] - 1)
        print(f"   Paso 1: TruncatedSVD ({X_sparse.shape[1]}D → {n_comp}D)...")
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        X_reduced = svd.fit_transform(X_sparse)
        explained = svd.explained_variance_ratio_.sum()
        print(f"   Varianza explicada por SVD: {explained:.2%}")
    else:
        X_reduced = X_sparse.toarray() if hasattr(X_sparse, "toarray") else X_sparse

    print(f"   Paso 2: T-SNE ({X_reduced.shape[1]}D → 2D, perplexity={perplexity})...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1,
        learning_rate="auto",
        init="pca",
    )
    X_2d = tsne.fit_transform(X_reduced)
    print(f"   ✅ T-SNE completado. Shape: {X_2d.shape}")
    return X_2d


def plot_tsne(
    X_2d: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "T-SNE — Espacio de características TF-IDF por Intent",
    output_path: str = "reports/figures/tsne_visualization.png",
    figsize: tuple = (12, 8),
    alpha: float = 0.7,
    point_size: int = 40,
) -> plt.Figure:
    """
    Visualiza el espacio T-SNE con colores por clase (intent).

    Args:
        X_2d: Coordenadas 2D de T-SNE (n_samples, 2)
        labels: Labels numéricos de cada muestra
        class_names: Nombres de las clases para la leyenda
        title: Título del gráfico
        output_path: Ruta donde guardar la figura
        figsize: Tamaño de la figura
        alpha: Transparencia de los puntos
        point_size: Tamaño de los puntos

    Returns:
        Figura de matplotlib
    """
    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)

    fig, ax = plt.subplots(figsize=figsize)

    legend_patches = []
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = INTENT_COLORS[i % len(INTENT_COLORS)]
        marker = INTENT_MARKERS[i % len(INTENT_MARKERS)]
        name = class_names[label] if class_names else f"Class {label}"

        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=color,
            marker=marker,
            s=point_size,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.3,
            label=name,
        )
        legend_patches.append(
            mpatches.Patch(color=color, label=name)
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("T-SNE Dimensión 1", fontsize=11)
    ax.set_ylabel("T-SNE Dimensión 2", fontsize=11)
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(1.18, 1.0),
        fontsize=9,
        framealpha=0.9,
        title="Intents",
    )
    ax.grid(linestyle="--", alpha=0.3)

    # Anotación metodológica
    ax.annotate(
        "Pipeline: TF-IDF → TruncatedSVD(50) → T-SNE(perp=30)",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        fontsize=8,
        color="gray",
        style="italic",
    )

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Visualización T-SNE guardada: {output_path}")

    return fig


def run_tsne_pipeline(
    X_sparse,
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "reports/figures/tsne_visualization.png",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> plt.Figure:
    """
    Pipeline completo: matriz TF-IDF → T-SNE → visualización.

    Args:
        X_sparse: Matriz TF-IDF sparse
        y: Labels
        class_names: Nombres de clases
        output_path: Ruta de salida
        perplexity: Parámetro T-SNE
        n_iter: Iteraciones T-SNE
        random_state: Semilla

    Returns:
        Figura matplotlib
    """
    X_2d = compute_tsne(
        X_sparse,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
    )

    fig = plot_tsne(
        X_2d=X_2d,
        labels=y,
        class_names=class_names,
        output_path=output_path,
    )

    return fig
