"""
Schemas Pydantic para validación de requests y responses de la API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict


class PredictRequest(BaseModel):
    """Schema para request de predicción de intent."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Consulta geoespacial del usuario",
        example="¿Cuántos kilómetros hay entre Quito y Guayaquil?",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Muéstrame el mapa de densidad poblacional de la zona norte"
            }
        }


class PredictResponse(BaseModel):
    """Schema para response de predicción."""

    text: str = Field(..., description="Texto original de la consulta")
    intent: str = Field(..., description="Intent clasificado")
    intent_id: int = Field(..., description="ID numérico del intent")
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confianza de la predicción (solo para Logistic Regression)",
    )
    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Probabilidades por clase (solo para Logistic Regression)",
    )
    agent_action: str = Field(
        ...,
        description="Agente del sistema multiagente que debe activarse",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "¿Cuántos km hay entre Quito y Guayaquil?",
                "intent": "calcular_distancia",
                "intent_id": 0,
                "confidence": 0.9821,
                "probabilities": {
                    "calcular_distancia": 0.9821,
                    "buscar_ubicacion": 0.0112,
                    "analizar_cobertura": 0.0067,
                },
                "agent_action": "Activar Agente de Geometría → calcular ruta entre puntos",
            }
        }


class BatchPredictRequest(BaseModel):
    """Schema para predicción en batch."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Lista de consultas geoespaciales",
    )


class BatchPredictResponse(BaseModel):
    """Schema para response de predicción en batch."""

    results: list[PredictResponse]
    total: int
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    """Schema para health check."""

    status: str
    model_loaded: bool
    version: str
