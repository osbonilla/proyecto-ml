"""
Constructor de pipelines de ML para clasificación de intención geoespacial.
Implementa pipelines completos: Preprocesamiento → TF-IDF → Feature Selection → Modelo.

Modelos implementados:
    - Logistic Regression (baseline robusto)
    - SVM con kernel lineal (mejor para texto de alta dimensión)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Literal, Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder


def build_lr_pipeline(
    ngram_range: tuple = (1, 2),
    max_features: int = 5000,
    k_features: int = 3000,
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str = "balanced",
) -> Pipeline:
    """
    Pipeline: TF-IDF → Chi² Feature Selection → Logistic Regression

    ¿Por qué Logistic Regression?
        - Excelente baseline para clasificación de texto
        - Rápido, interpretable, probabilístico
        - Buena regularización con parámetro C
        - Funciona bien con TF-IDF en alta dimensión

    Args:
        ngram_range: Rango de n-gramas para TF-IDF
        max_features: Máximo de features TF-IDF
        k_features: Features a seleccionar con Chi²
        C: Parámetro de regularización (menor C = más regularización)
        max_iter: Máximo de iteraciones del solver
        class_weight: 'balanced' compensa desbalance de clases

    Returns:
        Pipeline de sklearn listo para fit/predict
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            token_pattern=r"(?u)\b\w+\b",
        )),
        ("feature_selection", SelectKBest(
            score_func=chi2,
            k=min(k_features, max_features),
        )),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=42,
            n_jobs=-1,
        )),
    ], memory=None)

    return pipeline


def build_svm_pipeline(
    ngram_range: tuple = (1, 2),
    max_features: int = 5000,
    k_features: int = 3000,
    C: float = 1.0,
    class_weight: str = "balanced",
    max_iter: int = 2000,
) -> Pipeline:
    """
    Pipeline: TF-IDF → Chi² Feature Selection → LinearSVC

    ¿Por qué LinearSVC para texto?
        - Optimizado para alta dimensionalidad (TF-IDF)
        - Generalmente supera a Logistic Regression en texto
        - Muy eficiente en entrenamiento (dual=False para n>>p)
        - No da probabilidades directamente (usar CalibratedClassifierCV si se necesitan)

    Args:
        ngram_range: Rango de n-gramas
        max_features: Máximo de features TF-IDF
        k_features: Features a seleccionar con Chi²
        C: Parámetro de margen SVM
        class_weight: 'balanced' compensa desbalance
        max_iter: Máximo iteraciones

    Returns:
        Pipeline de sklearn
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            token_pattern=r"(?u)\b\w+\b",
        )),
        ("feature_selection", SelectKBest(
            score_func=chi2,
            k=min(k_features, max_features),
        )),
        ("classifier", LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=42,
        )),
    ])

    return pipeline


def get_pipeline(
    model_type: Literal["lr", "svm"] = "lr",
    **kwargs,
) -> Pipeline:
    """
    Factory function para obtener un pipeline por nombre.

    Args:
        model_type: 'lr' para Logistic Regression, 'svm' para LinearSVC
        **kwargs: Parámetros del pipeline

    Returns:
        Pipeline configurado
    """
    builders = {
        "lr": build_lr_pipeline,
        "svm": build_svm_pipeline,
    }

    if model_type not in builders:
        raise ValueError(f"Modelo '{model_type}' no soportado. Opciones: {list(builders.keys())}")

    return builders[model_type](**kwargs)


def save_pipeline(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    path_pipeline: str = "models/saved/best_model.pkl",
    path_encoder: str = "models/saved/label_encoder.pkl",
) -> None:
    """Serializa pipeline y label encoder."""
    Path(path_pipeline).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path_pipeline)
    joblib.dump(label_encoder, path_encoder)
    print(f"✅ Pipeline guardado en: {path_pipeline}")
    print(f"✅ Label encoder guardado en: {path_encoder}")


def load_pipeline(
    path_pipeline: str = "models/saved/best_model.pkl",
    path_encoder: str = "models/saved/label_encoder.pkl",
):
    """Carga pipeline y label encoder serializados."""
    pipeline = joblib.load(path_pipeline)
    label_encoder = joblib.load(path_encoder)
    print(f"✅ Pipeline cargado desde: {path_pipeline}")
    return pipeline, label_encoder


def predict_intent(
    text: str,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
) -> Dict[str, Any]:
    """
    Predice el intent de una consulta geoespacial.

    Args:
        text: Consulta del usuario
        pipeline: Pipeline entrenado
        label_encoder: Encoder de clases

    Returns:
        Dict con intent predicho y metadata
    """
    prediction = pipeline.predict([text])[0]
    intent_label = label_encoder.inverse_transform([prediction])[0]

    result = {
        "text": text,
        "intent_id": int(prediction),
        "intent": intent_label,
    }

    # Probabilidades solo si el modelo las soporta (LR)
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        intent_proba = {
            label_encoder.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)
        }
        result["confidence"] = round(float(max(proba)), 4)
        result["probabilities"] = dict(
            sorted(intent_proba.items(), key=lambda x: x[1], reverse=True)
        )
    else:
        result["confidence"] = None
        result["probabilities"] = {}

    return result


if __name__ == "__main__":
    # Test de construcción de pipelines
    lr_pipe = get_pipeline("lr")
    svm_pipe = get_pipeline("svm")
    print("✅ Pipelines construidos correctamente")
    print(f"   LR steps: {list(lr_pipe.named_steps.keys())}")
    print(f"   SVM steps: {list(svm_pipe.named_steps.keys())}")
