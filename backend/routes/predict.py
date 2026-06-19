"""
Rutas de predicción para la API de clasificación de intención geoespacial.
"""

from fastapi import APIRouter, HTTPException, Request
from backend.schemas.request_models import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
)

router = APIRouter()

# Mapa de intent → acción del agente multiagente
INTENT_TO_AGENT = {
    "calcular_distancia": "Activar Agente de Geometría → calcular ruta/distancia entre puntos",
    "analizar_cobertura": "Activar Agente de Análisis Espacial → calcular cobertura del suelo",
    "visualizar_mapa": "Activar Agente de Visualización → renderizar mapa temático",
    "buscar_ubicacion": "Activar Agente de Geocodificación → buscar coordenadas/dirección",
    "consultar_capa": "Activar Agente de Gestión de Capas → listar/describir capas ArcGIS",
    "exportar_datos": "Activar Agente de Exportación → generar archivo de salida",
    "detectar_cambios": "Activar Agente de Análisis Temporal → comparar periodos de tiempo",
    "generar_reporte": "Activar Agente de Reportes → generar informe automático",
}


def _predict_single(text: str, request: Request) -> dict:
    """Lógica de predicción para una sola consulta."""
    pipeline = request.app.state.pipeline
    label_encoder = request.app.state.label_encoder

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Ejecuta el entrenamiento primero.",
        )

    # Predicción
    prediction = pipeline.predict([text])[0]
    intent_label = label_encoder.inverse_transform([prediction])[0]

    result = {
        "text": text,
        "intent": intent_label,
        "intent_id": int(prediction),
        "confidence": None,
        "probabilities": {},
        "agent_action": INTENT_TO_AGENT.get(intent_label, "Agente desconocido"),
    }

    # Probabilidades si el modelo las soporta (LR)
    classifier = pipeline.named_steps.get("classifier")
    if hasattr(classifier, "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        intent_proba = {
            label_encoder.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)
        }
        result["confidence"] = round(float(max(proba)), 4)
        result["probabilities"] = dict(
            sorted(intent_proba.items(), key=lambda x: x[1], reverse=True)
        )

    return result


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Clasificar intent de una consulta geoespacial",
    description="""
    Recibe una consulta en lenguaje natural y devuelve:
    - **intent**: la intención detectada
    - **confidence**: confianza de la predicción
    - **agent_action**: qué agente del sistema multiagente debe activarse
    """,
)
async def predict(body: PredictRequest, request: Request):
    """Endpoint principal de predicción."""
    result = _predict_single(body.text, request)
    return PredictResponse(**result)


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Clasificar múltiples consultas en batch",
)
async def predict_batch(body: BatchPredictRequest, request: Request):
    """Predicción en lote para múltiples consultas."""
    results = []
    for text in body.texts:
        result = _predict_single(text, request)
        results.append(PredictResponse(**result))

    return BatchPredictResponse(
        results=results,
        total=len(results),
    )


@router.get(
    "/intents",
    summary="Listar intents soportados",
)
async def list_intents():
    """Retorna la lista de intents soportados y su agente correspondiente."""
    return {
        "intents": [
            {"intent": intent, "agent_action": action}
            for intent, action in INTENT_TO_AGENT.items()
        ],
        "total": len(INTENT_TO_AGENT),
    }
