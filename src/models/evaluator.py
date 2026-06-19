"""
Evaluador de modelos: métricas, matriz de confusión y test estadístico de Wilcoxon.
Incluye comparación estadística entre dos técnicas de ML para la rúbrica.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from scipy.stats import wilcoxon, ttest_rel


class ModelEvaluator:
    """
    Evaluador completo de modelos de clasificación.

    Métricas:
        - Accuracy, F1-macro, F1-weighted
        - Precision y Recall por clase
        - Confusion Matrix
        - Test de Wilcoxon para comparación estadística

    Compatible con la rúbrica:
        ✅ Comparación estadística de 2 técnicas ML (Wilcoxon)
        ✅ Repeated k-fold cross-validation
        ✅ Confusion matrix
        ✅ Per-class performance
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        output_dir: str = "reports/figures",
    ):
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        class_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Calcula métricas completas de clasificación.

        Args:
            y_true: Labels verdaderos
            y_pred: Labels predichos
            model_name: Nombre del modelo para reportes
            class_names: Nombres de las clases

        Returns:
            Dict con todas las métricas
        """
        names = class_names or self.class_names

        metrics = {
            "model": model_name,
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
            "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
            "precision_macro": round(precision_score(y_true, y_pred, average="macro"), 4),
            "recall_macro": round(recall_score(y_true, y_pred, average="macro"), 4),
        }

        # Reporte por clase
        report = classification_report(
            y_true, y_pred,
            target_names=names,
            output_dict=True,
            zero_division=0,
        )
        metrics["per_class"] = {
            k: v for k, v in report.items()
            if k not in ["accuracy", "macro avg", "weighted avg"]
        }

        print(f"\n📊 Métricas para {model_name}:")
        print(f"   Accuracy:   {metrics['accuracy']:.4f}")
        print(f"   F1-macro:   {metrics['f1_macro']:.4f}")
        print(f"   F1-weighted:{metrics['f1_weighted']:.4f}")
        print(f"\n{classification_report(y_true, y_pred, target_names=names, zero_division=0)}")

        return metrics

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
        save: bool = True,
    ) -> plt.Figure:
        """
        Genera y guarda la matriz de confusión.

        Args:
            y_true: Labels verdaderos
            y_pred: Labels predichos
            model_name: Nombre del modelo
            class_names: Nombres de clases
            normalize: Si True, normaliza por fila (recall por clase)
            save: Si True, guarda la figura

        Returns:
            Figura de matplotlib
        """
        names = class_names or self.class_names or [str(i) for i in sorted(set(y_true))]
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
            title = f"Confusion Matrix (normalizada) — {model_name}"
        else:
            cm_plot = cm
            fmt = "d"
            title = f"Confusion Matrix — {model_name}"

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=names,
            yticklabels=names,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_xlabel("Predicho", fontsize=12)
        ax.set_ylabel("Real", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        if save:
            path = self.output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"✅ Confusion matrix guardada: {path}")

        return fig

    def wilcoxon_test(
        self,
        scores_model1: List[float],
        scores_model2: List[float],
        model1_name: str = "LR",
        model2_name: str = "SVM",
        alpha: float = 0.05,
    ) -> Dict:
        """
        Test estadístico de Wilcoxon para comparar dos modelos.

        ¿Por qué Wilcoxon y no t-test?
            - Wilcoxon es no paramétrico → no asume distribución normal
            - Más robusto para muestras pequeñas
            - Apropiado para comparar scores de CV donde la distribución
              puede no ser gaussiana
            - Recomendado por Demšar (2006) para comparar clasificadores

        Args:
            scores_model1: Scores de CV del modelo 1 (LR)
            scores_model2: Scores de CV del modelo 2 (SVM)
            model1_name: Nombre del modelo 1
            model2_name: Nombre del modelo 2
            alpha: Nivel de significancia (default 0.05)

        Returns:
            Dict con estadístico, p-value e interpretación
        """
        scores1 = np.array(scores_model1)
        scores2 = np.array(scores_model2)

        # Test de Wilcoxon signed-rank (pareado)
        stat_w, pvalue_w = wilcoxon(scores1, scores2, zero_method="zsplit")

        # Test t pareado (complementario)
        stat_t, pvalue_t = ttest_rel(scores1, scores2)

        significant = pvalue_w < alpha
        winner = (
            model1_name if scores1.mean() > scores2.mean() else model2_name
        ) if significant else "Sin diferencia significativa"

        result = {
            "model1": model1_name,
            "model2": model2_name,
            "mean_model1": round(float(scores1.mean()), 4),
            "std_model1": round(float(scores1.std()), 4),
            "mean_model2": round(float(scores2.mean()), 4),
            "std_model2": round(float(scores2.std()), 4),
            "wilcoxon_statistic": round(float(stat_w), 4),
            "wilcoxon_pvalue": round(float(pvalue_w), 6),
            "ttest_statistic": round(float(stat_t), 4),
            "ttest_pvalue": round(float(pvalue_t), 6),
            "alpha": alpha,
            "significant": bool(significant),
            "winner": winner,
            "interpretation": (
                f"{'✅ Diferencia estadísticamente significativa' if significant else '❌ Sin diferencia significativa'} "
                f"(p={pvalue_w:.4f}, α={alpha}). "
                f"Modelo superior: {winner}."
            ),
        }

        print(f"\n🧪 Test de Wilcoxon: {model1_name} vs {model2_name}")
        print(f"   {model1_name}: {result['mean_model1']:.4f} ± {result['std_model1']:.4f}")
        print(f"   {model2_name}: {result['mean_model2']:.4f} ± {result['std_model2']:.4f}")
        print(f"   Wilcoxon p-value: {pvalue_w:.6f}")
        print(f"   {result['interpretation']}")

        return result

    def plot_model_comparison(
        self,
        results: Dict,
        save: bool = True,
    ) -> plt.Figure:
        """
        Gráfico de comparación de scores entre modelos (boxplot).

        Args:
            results: Dict con resultados de trainer.train_all()
            save: Si True, guarda la figura

        Returns:
            Figura de matplotlib
        """
        data = []
        for model_type, res in results.items():
            scores = res["cv_result"]["scores"]
            for s in scores:
                data.append({"Model": model_type.upper(), "F1-Macro": s})

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="Model", y="F1-Macro", palette="Set2", ax=ax)
        sns.stripplot(data=df, x="Model", y="F1-Macro", color="black",
                      alpha=0.4, size=4, ax=ax)

        ax.set_title("Comparación de Modelos — F1-Macro (RepeatedKFold)", fontsize=13)
        ax.set_ylabel("F1-Macro Score", fontsize=11)
        ax.set_xlabel("Modelo", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        if save:
            path = self.output_dir / "model_comparison_boxplot.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"✅ Boxplot guardado: {path}")

        return fig

    def plot_learning_curve(
        self,
        lc_data: Dict,
        save: bool = True,
    ) -> plt.Figure:
        """
        Grafica la curva de aprendizaje de un modelo.

        Args:
            lc_data: Dict con datos de learning_curve (del trainer)
            save: Si True, guarda la figura

        Returns:
            Figura de matplotlib
        """
        model_name = lc_data["model"].upper()
        train_sizes = lc_data["train_sizes"]
        train_mean = np.array(lc_data["train_mean"])
        train_std = np.array(lc_data["train_std"])
        val_mean = np.array(lc_data["val_mean"])
        val_std = np.array(lc_data["val_std"])

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(train_sizes, train_mean, "o-", color="royalblue",
                label="Train score", linewidth=2)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color="royalblue")

        ax.plot(train_sizes, val_mean, "o-", color="darkorange",
                label="Validation score", linewidth=2)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color="darkorange")

        ax.set_title(f"Curva de Aprendizaje — {model_name}", fontsize=13)
        ax.set_xlabel("Tamaño del conjunto de entrenamiento", fontsize=11)
        ax.set_ylabel("F1-Macro Score", fontsize=11)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(linestyle="--", alpha=0.6)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        if save:
            path = self.output_dir / f"learning_curve_{model_name.lower()}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"✅ Learning curve guardada: {path}")

        return fig
