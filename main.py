"""
main.py — Entry point del proyecto GeoIntent Classifier.

Uso:
    python main.py --train          # Genera dataset + entrena + guarda modelo
    python main.py --train --eval   # Entrenamiento + evaluación completa
    python main.py --serve          # Levanta la API FastAPI
    python main.py --predict "texto" # Predicción de una consulta

Prompt de referencia:
"Genera un pipeline completo de ML para clasificación de intención geoespacial en Python"
"""

import argparse
import sys
import os
from pathlib import Path

# Asegurar que el módulo src sea encontrado
sys.path.insert(0, str(Path(__file__).parent))


def train(evaluate: bool = True, optimize: bool = True):
    """Pipeline completo de entrenamiento."""
    print("\n" + "="*70)
    print("  GeoIntent Classifier — Pipeline de Entrenamiento")
    print("="*70)

    # 1. Imports
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    from src.data.dataset_builder import generate_dataset, save_dataset
    from src.data.preprocessor import preprocess_dataset
    from src.models.trainer import ModelTrainer
    from src.models.evaluator import ModelEvaluator
    from src.models.pipeline_builder import save_pipeline, get_pipeline
    from src.visualization.eda_plots import (
        plot_class_distribution,
        plot_text_length_distribution,
        plot_top_tokens,
        generate_eda_summary,
    )

    # ----------------------------------------------------------------
    # PASO 1: Generar dataset
    # ----------------------------------------------------------------
    print("\n[1/7] Generando dataset sintético...")
    raw_path = "data/raw/dataset_raw.csv"
    if not Path(raw_path).exists():
        df_raw = generate_dataset(samples_per_intent=100)
        save_dataset(df_raw, raw_path)
    else:
        print(f"   Dataset existente encontrado: {raw_path}")
        df_raw = pd.read_csv(raw_path)

    # ----------------------------------------------------------------
    # PASO 2: Preprocesamiento
    # ----------------------------------------------------------------
    print("\n[2/7] Preprocesando texto...")
    df = preprocess_dataset(
        input_path=raw_path,
        output_path="data/processed/dataset_clean.csv",
        remove_stopwords=True,
        remove_accents=False,
    )

    # ----------------------------------------------------------------
    # PASO 3: EDA básico
    # ----------------------------------------------------------------
    print("\n[3/7] Generando visualizaciones EDA...")
    generate_eda_summary(df)
    plot_class_distribution(df, label_col="intent")
    plot_text_length_distribution(df, text_col="text", label_col="intent")
    plot_top_tokens(df, text_col="text_clean")

    # ----------------------------------------------------------------
    # PASO 4: Preparar features y labels
    # ----------------------------------------------------------------
    print("\n[4/7] Preparando features y labels...")
    X = df["text_clean"].tolist()
    y_raw = df["intent"].tolist()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()

    print(f"   Clases: {class_names}")
    print(f"   Total muestras: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # ----------------------------------------------------------------
    # PASO 5: Entrenamiento y optimización
    # ----------------------------------------------------------------
    print("\n[5/7] Entrenando modelos con validación cruzada...")
    trainer = ModelTrainer(
        n_splits=5,
        n_repeats=3,    # 3x5=15 evaluaciones por modelo
        scoring="f1_macro",
        verbose=0,
    )
    trainer.label_encoder = le

    results = trainer.train_all(
        X_train=X_train,
        y_train=y_train,
        X_all=X,
        y_all=y,
        optimize_hyperparams=optimize,
    )

    trainer.save_results()

    # ----------------------------------------------------------------
    # PASO 6: Evaluación en test set
    # ----------------------------------------------------------------
    if evaluate:
        print("\n[6/7] Evaluando en test set...")
        evaluator = ModelEvaluator(class_names=class_names)

        for model_type in ["lr", "svm"]:
            pipeline = trainer.best_pipelines_.get(model_type)
            if pipeline is None:
                continue

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            evaluator.compute_metrics(y_test, y_pred, model_name=model_type.upper(),
                                      class_names=class_names)
            evaluator.plot_confusion_matrix(y_test, y_pred, model_name=model_type.upper(),
                                            class_names=class_names)

        # Curvas de aprendizaje
        for model_type, res in results.items():
            evaluator.plot_learning_curve(res["learning_curve"])

        # Boxplot comparativo
        evaluator.plot_model_comparison(results)

        # Test de Wilcoxon
        scores_lr = results["lr"]["cv_result"]["scores"]
        scores_svm = results["svm"]["cv_result"]["scores"]
        wilcoxon_result = evaluator.wilcoxon_test(scores_lr, scores_svm)

        import json
        Path("reports/metrics").mkdir(parents=True, exist_ok=True)
        with open("reports/metrics/wilcoxon_test.json", "w") as f:
            json.dump(wilcoxon_result, f, indent=2)
        print("✅ Resultado Wilcoxon guardado")

    # ----------------------------------------------------------------
    # PASO 7: Guardar mejor modelo
    # ----------------------------------------------------------------
    print("\n[7/7] Guardando mejor modelo final...")

    # Determinar mejor modelo según F1-macro
    best_model_type = max(
        results.keys(),
        key=lambda k: results[k]["cv_result"]["mean"]
    )
    print(f"   🏆 Mejor modelo: {best_model_type.upper()}")

    best_pipeline = trainer.best_pipelines_[best_model_type]
    best_pipeline.fit(X, y)  # Entrenar en todos los datos

    save_pipeline(
        pipeline=best_pipeline,
        label_encoder=le,
        path_pipeline="models/saved/best_model.pkl",
        path_encoder="models/saved/label_encoder.pkl",
    )

    print("\n" + "="*70)
    print("  ✅ ENTRENAMIENTO COMPLETADO")
    print(f"  Mejor modelo: {best_model_type.upper()}")
    print(f"  F1-macro: {results[best_model_type]['cv_result']['mean']:.4f}")
    print("  Para iniciar la API: python main.py --serve")
    print("="*70)


def serve():
    """Levanta la API FastAPI con uvicorn."""
    import uvicorn
    print("\n🚀 Iniciando API GeoIntent Classifier...")
    print("   Documentación: http://localhost:8000/docs")
    print("   Health:        http://localhost:8000/health")
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )


def predict_single(text: str):
    """Predice el intent de una consulta en modo CLI."""
    import joblib
    from src.models.pipeline_builder import predict_intent
    import json

    model_path = "models/saved/best_model.pkl"
    encoder_path = "models/saved/label_encoder.pkl"

    if not Path(model_path).exists():
        print("❌ Modelo no encontrado. Ejecuta: python main.py --train")
        sys.exit(1)

    pipeline = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    result = predict_intent(text, pipeline, label_encoder)
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GeoIntent Classifier — Pipeline de ML para clasificación de intención geoespacial"
    )
    parser.add_argument("--train", action="store_true", help="Ejecutar pipeline de entrenamiento")
    parser.add_argument("--eval", action="store_true", help="Incluir evaluación completa en el entrenamiento")
    parser.add_argument("--no-optimize", action="store_true", help="Saltar GridSearchCV (más rápido)")
    parser.add_argument("--serve", action="store_true", help="Levantar API FastAPI")
    parser.add_argument("--predict", type=str, metavar="TEXT", help="Predecir intent de un texto")

    args = parser.parse_args()

    if args.train:
        train(evaluate=args.eval, optimize=not args.no_optimize)
    elif args.serve:
        serve()
    elif args.predict:
        predict_single(args.predict)
    else:
        parser.print_help()
        print("\nEjemplos:")
        print("  python main.py --train --eval")
        print("  python main.py --serve")
        print('  python main.py --predict "¿Cuántos km hay entre Quito y Guayaquil?"')
