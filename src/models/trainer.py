"""
Entrenador de modelos con validación cruzada repetida y búsqueda de hiperparámetros.
Implementa RepeatedKFold + GridSearchCV para comparación estadística robusta.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    GridSearchCV,
    cross_val_score,
    learning_curve,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from src.models.pipeline_builder import get_pipeline, save_pipeline


# =============================================================================
# GRIDS DE HIPERPARÁMETROS
# =============================================================================

PARAM_GRID_LR = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [3000, 5000],
    "feature_selection__k": [1000, 2000, 3000],
    "classifier__C": [0.1, 1.0, 10.0],
}

PARAM_GRID_SVM = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [3000, 5000],
    "feature_selection__k": [1000, 2000, 3000],
    "classifier__C": [0.1, 1.0, 10.0],
}


class ModelTrainer:
    """
    Entrenador de modelos con RepeatedStratifiedKFold.

    Flujo:
        1. Encode labels
        2. GridSearchCV para encontrar mejores hiperparámetros
        3. RepeatedStratifiedKFold para evaluación robusta
        4. Calcular learning curves
        5. Guardar mejor modelo
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 3,
        scoring: str = "f1_macro",
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
    ):
        """
        Args:
            n_splits: Número de folds en CV
            n_repeats: Número de repeticiones de CV
            scoring: Métrica de evaluación
            n_jobs: Procesadores a usar (-1 = todos)
            random_state: Semilla de reproducibilidad
            verbose: Nivel de detalle en logs
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.label_encoder = LabelEncoder()
        self.cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        self.results_: Dict = {}
        self.best_pipelines_: Dict[str, Pipeline] = {}

    def encode_labels(self, y: pd.Series) -> np.ndarray:
        """Codifica labels de texto a enteros."""
        return self.label_encoder.fit_transform(y)

    def optimize_hyperparams(
        self,
        model_type: str,
        X_train: list,
        y_train: np.ndarray,
        param_grid: Optional[dict] = None,
        cv_folds: int = 5,
    ) -> Tuple[Pipeline, dict]:
        """
        Búsqueda de hiperparámetros con GridSearchCV + StratifiedKFold.

        Args:
            model_type: 'lr' o 'svm'
            X_train: Lista de textos de entrenamiento
            y_train: Labels codificados
            param_grid: Grid de parámetros (None = usar default)
            cv_folds: Folds para GridSearchCV interno

        Returns:
            Tuple (mejor pipeline, mejores parámetros)
        """
        from sklearn.model_selection import StratifiedKFold

        if param_grid is None:
            param_grid = PARAM_GRID_LR if model_type == "lr" else PARAM_GRID_SVM

        pipeline = get_pipeline(model_type)
        cv_inner = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                   random_state=self.random_state)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_inner,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            refit=True,
            return_train_score=True,
        )

        print(f"\n🔍 Optimizando hiperparámetros para: {model_type.upper()}")
        grid_search.fit(X_train, y_train)

        print(f"   ✅ Mejor score ({self.scoring}): {grid_search.best_score_:.4f}")
        print(f"   ✅ Mejores parámetros: {grid_search.best_params_}")

        return grid_search.best_estimator_, grid_search.best_params_

    def evaluate_with_cv(
        self,
        model_type: str,
        pipeline: Pipeline,
        X: list,
        y: np.ndarray,
    ) -> Dict:
        """
        Evaluación robusta con RepeatedStratifiedKFold.

        Args:
            model_type: Nombre del modelo
            pipeline: Pipeline entrenado con mejores hiperparámetros
            X: Lista completa de textos
            y: Labels completos codificados

        Returns:
            Dict con scores por fold y estadísticas
        """
        print(f"\n📊 Evaluando {model_type.upper()} con RepeatedStratifiedKFold "
              f"({self.n_repeats} repeticiones × {self.n_splits} folds)...")

        scores = cross_val_score(
            pipeline, X, y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )

        result = {
            "model": model_type,
            "scores": scores.tolist(),
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
            "n_evaluations": len(scores),
        }

        print(f"   F1-macro: {result['mean']:.4f} ± {result['std']:.4f}")
        print(f"   Min: {result['min']:.4f} | Max: {result['max']:.4f}")

        return result

    def compute_learning_curve(
        self,
        model_type: str,
        pipeline: Pipeline,
        X: list,
        y: np.ndarray,
        train_sizes: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Calcula la curva de aprendizaje para detectar overfitting/underfitting.

        Args:
            model_type: Nombre del modelo
            pipeline: Pipeline
            X: Textos
            y: Labels
            train_sizes: Proporciones de training a evaluar

        Returns:
            Dict con datos de la curva de aprendizaje
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 8)

        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        print(f"\n📈 Calculando curva de aprendizaje para {model_type.upper()}...")

        train_sizes_abs, train_scores, val_scores = learning_curve(
            pipeline, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=0,
        )

        return {
            "model": model_type,
            "train_sizes": train_sizes_abs.tolist(),
            "train_mean": train_scores.mean(axis=1).tolist(),
            "train_std": train_scores.std(axis=1).tolist(),
            "val_mean": val_scores.mean(axis=1).tolist(),
            "val_std": val_scores.std(axis=1).tolist(),
        }

    def train_all(
        self,
        X_train: list,
        y_train: np.ndarray,
        X_all: list,
        y_all: np.ndarray,
        optimize_hyperparams: bool = True,
    ) -> Dict:
        """
        Entrena y evalúa todos los modelos.

        Args:
            X_train: Textos de entrenamiento (para optimización)
            y_train: Labels de entrenamiento
            X_all: Todos los textos (para CV final)
            y_all: Todos los labels

        Returns:
            Dict con todos los resultados
        """
        all_results = {}

        for model_type in ["lr", "svm"]:
            print(f"\n{'='*60}")
            print(f"  MODELO: {model_type.upper()}")
            print(f"{'='*60}")

            # Optimización de hiperparámetros
            if optimize_hyperparams:
                best_pipeline, best_params = self.optimize_hyperparams(
                    model_type, X_train, y_train
                )
            else:
                best_pipeline = get_pipeline(model_type)
                best_params = {}

            self.best_pipelines_[model_type] = best_pipeline

            # Evaluación con CV completa
            cv_result = self.evaluate_with_cv(model_type, best_pipeline, X_all, y_all)

            # Curva de aprendizaje
            lc_result = self.compute_learning_curve(model_type, best_pipeline, X_all, y_all)

            all_results[model_type] = {
                "cv_result": cv_result,
                "learning_curve": lc_result,
                "best_params": best_params,
            }

        self.results_ = all_results
        return all_results

    def fit_final_model(
        self,
        model_type: str,
        X: list,
        y: np.ndarray,
    ) -> Pipeline:
        """Entrena el modelo final sobre todos los datos."""
        print(f"\n🏋️ Entrenando modelo final {model_type.upper()} sobre todos los datos...")
        pipeline = self.best_pipelines_.get(model_type, get_pipeline(model_type))
        pipeline.fit(X, y)
        print("   ✅ Entrenamiento completo")
        return pipeline

    def save_results(self, path: str = "reports/metrics/model_comparison.csv") -> None:
        """Guarda resultados de comparación en CSV."""
        rows = []
        for model_type, result in self.results_.items():
            cv = result["cv_result"]
            rows.append({
                "model": model_type.upper(),
                "f1_macro_mean": cv["mean"],
                "f1_macro_std": cv["std"],
                "f1_macro_min": cv["min"],
                "f1_macro_max": cv["max"],
                "n_evaluations": cv["n_evaluations"],
            })

        df = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"\n✅ Resultados guardados en: {path}")
        print(df.to_string(index=False))
