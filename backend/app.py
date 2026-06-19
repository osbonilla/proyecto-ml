"""
API REST con FastAPI para clasificación de intención geoespacial.
Expone endpoint de predicción para integración con el agente multiagente.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import os
from pathlib import Path

from backend.schemas.request_models import PredictRequest, PredictResponse, HealthResponse
from backend.routes.predict import router as predict_router


# =============================================================================
# LIFECYCLE — carga el modelo al iniciar
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y label encoder al arrancar la app."""
    model_path = os.getenv("MODEL_PATH", "models/saved/best_model.pkl")
    encoder_path = os.getenv("ENCODER_PATH", "models/saved/label_encoder.pkl")

    if not Path(model_path).exists():
        print(f"⚠️  Modelo no encontrado en {model_path}")
        print("   Ejecuta: python main.py --train para entrenar primero")
        app.state.pipeline = None
        app.state.label_encoder = None
    else:
        print(f"✅ Cargando modelo desde {model_path}...")
        app.state.pipeline = joblib.load(model_path)
        app.state.label_encoder = joblib.load(encoder_path)
        print("✅ Modelo cargado exitosamente")

    yield

    # Cleanup
    app.state.pipeline = None
    app.state.label_encoder = None


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="GeoIntent Classifier API",
    description="""
    API para clasificación de intención en consultas geoespaciales.
    
    ## Arquitectura
    - **Modelo**: TF-IDF + Feature Selection (Chi²) + Logistic Regression / SVM
    - **Dataset**: 800 consultas sintéticas en español (8 intents geoespaciales)
    - **Uso**: Router inteligente para arquitectura multiagente en ArcGIS
    
    ## Intents soportados
    - `calcular_distancia` → Agente de Geometría
    - `analizar_cobertura` → Agente de Análisis Espacial
    - `visualizar_mapa` → Agente de Visualización
    - `buscar_ubicacion` → Agente de Geocodificación
    - `consultar_capa` → Agente de Gestión de Capas
    - `exportar_datos` → Agente de Exportación
    - `detectar_cambios` → Agente de Análisis Temporal
    - `generar_reporte` → Agente de Reportes
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — permite requests desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas
app.include_router(predict_router, prefix="/api/v1", tags=["Predicción"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "GeoIntent Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    model_loaded = app.state.pipeline is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
    )
