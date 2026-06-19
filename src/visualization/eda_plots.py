"""
Módulo de visualizaciones para Exploratory Data Analysis (EDA).
Genera gráficos de distribución de clases, longitud de textos, word clouds, etc.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from collections import Counter
from typing import Optional, List


def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str = "intent",
    title: str = "Distribución de Clases (Intents)",
    output_path: str = "reports/figures/class_distribution.png",
    palette: str = "Set2",
) -> plt.Figure:
    """Grafica la distribución de clases del dataset."""
    counts = df[label_col].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Barplot
    colors = sns.color_palette(palette, n_colors=len(counts))
    bars = axes[0].barh(counts.index, counts.values, color=colors, edgecolor="white")
    axes[0].set_title("Frecuencia por Intent", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Número de muestras", fontsize=10)
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9
        )
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.8,
    )
    axes[1].set_title("Proporción por Intent", fontsize=12, fontweight="bold")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Distribución de clases guardada: {output_path}")
    return fig


def plot_text_length_distribution(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "intent",
    output_path: str = "reports/figures/text_length_distribution.png",
) -> plt.Figure:
    """Grafica la distribución de longitudes de texto por clase."""
    df = df.copy()
    df["n_words"] = df[text_col].str.split().str.len()
    df["n_chars"] = df[text_col].str.len()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=df, x=label_col, y="n_words", palette="Set3", ax=axes[0])
    axes[0].set_title("Distribución de palabras por Intent", fontsize=12)
    axes[0].set_xlabel("Intent", fontsize=10)
    axes[0].set_ylabel("Número de palabras", fontsize=10)
    axes[0].tick_params(axis="x", rotation=45)

    sns.histplot(data=df, x="n_words", hue=label_col, bins=15,
                 kde=True, ax=axes[1], palette="Set2", alpha=0.5)
    axes[1].set_title("Histograma de longitud de texto", fontsize=12)
    axes[1].set_xlabel("Número de palabras", fontsize=10)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Distribución de longitudes guardada: {output_path}")
    return fig


def plot_top_tokens(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    top_n: int = 20,
    output_path: str = "reports/figures/top_tokens.png",
) -> plt.Figure:
    """Grafica los tokens más frecuentes en el corpus."""
    all_tokens = " ".join(df[text_col].dropna()).split()
    token_counts = Counter(all_tokens).most_common(top_n)
    tokens, counts = zip(*token_counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(tokens)))
    bars = ax.barh(tokens[::-1], counts[::-1], color=colors[::-1])
    ax.set_title(f"Top {top_n} tokens más frecuentes", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frecuencia", fontsize=11)
    for bar, val in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Top tokens guardado: {output_path}")
    return fig


def generate_eda_summary(df: pd.DataFrame, label_col: str = "intent") -> pd.DataFrame:
    """
    Genera un resumen estadístico del dataset para el EDA.

    Returns:
        DataFrame con estadísticas del dataset
    """
    df = df.copy()
    df["n_words"] = df["text"].str.split().str.len()
    df["n_chars"] = df["text"].str.len()

    summary_global = pd.DataFrame({
        "Métrica": [
            "Total de muestras",
            "Número de clases",
            "Muestras por clase (promedio)",
            "Balance de clases",
            "Longitud media (palabras)",
            "Longitud mínima (palabras)",
            "Longitud máxima (palabras)",
        ],
        "Valor": [
            len(df),
            df[label_col].nunique(),
            round(len(df) / df[label_col].nunique(), 1),
            f"{df[label_col].value_counts().std():.2f} std",
            round(df["n_words"].mean(), 1),
            df["n_words"].min(),
            df["n_words"].max(),
        ]
    })

    print("\n📊 Resumen del Dataset:")
    print(summary_global.to_string(index=False))
    print(f"\nDistribución por intent:\n{df[label_col].value_counts().to_string()}")

    return summary_global
