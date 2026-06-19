"""
Feature Selection para clasificación de texto.
Implementa métodos de tipo Filter: Chi², Mutual Information y Varianza.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    mutual_info_classif,
    VarianceThreshold,
)
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal, Optional


class TextFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selector de features de tipo Filter para texto vectorizado (TF-IDF).

    Métodos soportados:
        - 'chi2': Chi-cuadrado. Mide dependencia entre feature y clase.
                  Requiere valores no negativos (compatible con TF-IDF).
        - 'mutual_info': Información Mutua. Mide información compartida.
                         Más robusto pero más lento.
        - 'variance': Elimina features con varianza baja (casi constantes).

    Por qué Chi² para TF-IDF:
        - Rápido y eficiente
        - Compatible con matrices sparse
        - Buen desempeño en clasificación de texto multiclase
        - Interpretable: p-values disponibles
    """

    def __init__(
        self,
        method: Literal["chi2", "mutual_info", "variance"] = "chi2",
        k: int = 3000,
        variance_threshold: float = 0.0,
    ):
        """
        Args:
            method: Método de selección de features
            k: Número de features a seleccionar (para chi2 y mutual_info)
            variance_threshold: Umbral de varianza mínima (para 'variance')
        """
        self.method = method
        self.k = k
        self.variance_threshold = variance_threshold
        self.selector = None
        self._feature_names_in = None

    def fit(self, X, y=None) -> "TextFeatureSelector":
        """Ajusta el selector al dataset."""
        if self.method == "chi2":
            self.selector = SelectKBest(score_func=chi2, k=min(self.k, X.shape[1]))
        elif self.method == "mutual_info":
            self.selector = SelectKBest(
                score_func=mutual_info_classif, k=min(self.k, X.shape[1])
            )
        elif self.method == "variance":
            self.selector = VarianceThreshold(threshold=self.variance_threshold)
        else:
            raise ValueError(f"Método desconocido: {self.method}")

        self.selector.fit(X, y)
        return self

    def transform(self, X):
        """Aplica la selección de features."""
        if self.selector is None:
            raise RuntimeError("El selector no ha sido ajustado. Llama fit() primero.")
        return self.selector.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_support_mask(self) -> np.ndarray:
        """Retorna máscara booleana de features seleccionados."""
        return self.selector.get_support()

    def get_scores(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Retorna scores de importancia de features.

        Args:
            feature_names: Nombres de los features originales

        Returns:
            DataFrame ordenado por score descendente
        """
        if self.method == "variance":
            scores = self.selector.variances_
            pvalues = [None] * len(scores)
        else:
            scores = self.selector.scores_
            pvalues = self.selector.pvalues_

        n = len(scores)
        names = feature_names if feature_names else [f"feature_{i}" for i in range(n)]
        selected = self.selector.get_support()

        df = pd.DataFrame({
            "feature": names,
            "score": scores,
            "pvalue": pvalues,
            "selected": selected,
        }).sort_values("score", ascending=False)

        return df

    def get_selected_feature_names(self, feature_names: list) -> list:
        """Retorna los nombres de los features seleccionados."""
        mask = self.get_support_mask()
        return [f for f, m in zip(feature_names, mask) if m]

    def n_features_selected(self) -> int:
        """Retorna el número de features seleccionados."""
        return int(self.get_support_mask().sum())


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import scipy.sparse as sp

    # Test con datos de ejemplo
    X, y = make_classification(n_samples=200, n_features=500, n_classes=4,
                                n_informative=20, random_state=42)
    X_sparse = sp.csr_matrix(np.abs(X))  # TF-IDF siempre >= 0

    selector = TextFeatureSelector(method="chi2", k=50)
    X_selected = selector.fit_transform(X_sparse, y)

    print(f"Features originales: {X_sparse.shape[1]}")
    print(f"Features seleccionados: {X_selected.shape[1]}")
    print(f"Reducción: {(1 - X_selected.shape[1]/X_sparse.shape[1])*100:.1f}%")
