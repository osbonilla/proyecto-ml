"""
Extractor de features TF-IDF con n-gramas para clasificación de intención.
Incluye configuración optimizada para texto en español de dominio geoespacial.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional


class TFIDFExtractor:
    """
    Extractor de features basado en TF-IDF con n-gramas.

    Configuración recomendada para dominio geoespacial:
        - Unigramas + bigramas: captura frases como "calcular distancia", "uso suelo"
        - max_features: controla dimensionalidad
        - sublinear_tf: normaliza frecuencias altas (mejor para texto corto)
        - min_df / max_df: filtra términos muy raros o muy comunes
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 5000,
        sublinear_tf: bool = True,
        min_df: int = 2,
        max_df: float = 0.95,
        analyzer: str = "word",
    ):
        """
        Args:
            ngram_range: (min_n, max_n) para n-gramas. (1,2) = unigramas+bigramas
            max_features: máximo de features en el vocabulario
            sublinear_tf: aplica log(1+tf) en lugar de tf (recomendado)
            min_df: mínimo de documentos que deben contener el término
            max_df: máximo de documentos que pueden contener el término
            analyzer: 'word' para n-gramas de palabras, 'char' para caracteres
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df = max_df
        self.analyzer = analyzer

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            max_df=self.max_df,
            analyzer=self.analyzer,
            token_pattern=r"(?u)\b\w+\b",  # incluye palabras con tildes
        )
        self._is_fitted = False

    def fit(self, texts: list) -> "TFIDFExtractor":
        """Ajusta el vectorizador al corpus."""
        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: list) -> np.ndarray:
        """Transforma textos a matriz TF-IDF."""
        if not self._is_fitted:
            raise RuntimeError("El vectorizador no ha sido ajustado. Llama fit() primero.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list) -> np.ndarray:
        """Ajusta y transforma en un solo paso."""
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> list:
        """Retorna lista de features (n-gramas) aprendidos."""
        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_features_per_class(
        self, X, y, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Retorna los top features más discriminativos por clase.
        Útil para interpretar qué palabras definen cada intent.

        Args:
            X: Matriz TF-IDF (sparse)
            y: Labels de las clases
            top_n: Número de top features por clase

        Returns:
            DataFrame con top features por clase
        """
        feature_names = self.get_feature_names()
        classes = sorted(set(y))
        records = []

        for cls in classes:
            mask = np.array(y) == cls
            X_cls = X[mask]
            mean_tfidf = np.asarray(X_cls.mean(axis=0)).flatten()
            top_indices = mean_tfidf.argsort()[::-1][:top_n]
            for rank, idx in enumerate(top_indices):
                records.append({
                    "intent": cls,
                    "rank": rank + 1,
                    "feature": feature_names[idx],
                    "mean_tfidf": round(mean_tfidf[idx], 4),
                })

        return pd.DataFrame(records)

    def save(self, path: str = "models/saved/tfidf_vectorizer.pkl") -> None:
        """Serializa el vectorizador entrenado."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"✅ Vectorizador guardado en: {path}")

    @classmethod
    def load(cls, path: str = "models/saved/tfidf_vectorizer.pkl") -> "TFIDFExtractor":
        """Carga un vectorizador serializado."""
        extractor = cls()
        extractor.vectorizer = joblib.load(path)
        extractor._is_fitted = True
        print(f"✅ Vectorizador cargado desde: {path}")
        return extractor

    def get_vocabulary_stats(self) -> dict:
        """Retorna estadísticas del vocabulario aprendido."""
        if not self._is_fitted:
            return {}
        vocab = self.vectorizer.vocabulary_
        return {
            "vocab_size": len(vocab),
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
            "unigrams": sum(1 for k in vocab if len(k.split()) == 1),
            "bigrams": sum(1 for k in vocab if len(k.split()) == 2),
        }


if __name__ == "__main__":
    # Test rápido
    sample_texts = [
        "calcular distancia quito guayaquil",
        "analizar cobertura forestal zona norte",
        "visualizar mapa densidad poblacional",
        "buscar ubicación hospital metropolitano",
    ]

    extractor = TFIDFExtractor(ngram_range=(1, 2), max_features=100)
    X = extractor.fit_transform(sample_texts)

    print(f"Shape de la matriz TF-IDF: {X.shape}")
    print(f"Stats vocabulario: {extractor.get_vocabulary_stats()}")
