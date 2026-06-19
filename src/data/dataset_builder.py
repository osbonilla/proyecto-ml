"""
Dataset Builder - Generador de dataset sintético para clasificación de intención geoespacial.
Genera consultas en español relacionadas con operaciones en ArcGIS / sistemas multiagente.

Prompt usado para generar variantes adicionales:
"Genera 10 variantes naturales en español de la consulta: '{query}' para intent '{intent}'"
"""

import pandas as pd
import numpy as np
import random
import re
from pathlib import Path

# Semilla para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# DEFINICIÓN DE INTENTS Y PLANTILLAS DE CONSULTAS
# =============================================================================

INTENT_TEMPLATES = {
    "calcular_distancia": [
        "¿Cuántos kilómetros hay entre {lugar1} y {lugar2}?",
        "Calcula la distancia entre {lugar1} y {lugar2}",
        "¿Qué tan lejos está {lugar1} de {lugar2}?",
        "Dame la distancia en metros entre {lugar1} y {lugar2}",
        "¿Cuánto mide el trayecto de {lugar1} a {lugar2}?",
        "Necesito saber la distancia entre {lugar1} y {lugar2}",
        "Calcula la ruta más corta entre {lugar1} y {lugar2}",
        "Distancia euclidiana entre {lugar1} y {lugar2}",
        "¿Cuánto dista {lugar1} de {lugar2}?",
        "Mide la separación entre {lugar1} y {lugar2}",
        "¿A qué distancia está {lugar2} desde {lugar1}?",
        "Obtén la distancia geodésica entre {lugar1} y {lugar2}",
        "¿Cuál es el recorrido entre {lugar1} y {lugar2}?",
        "Quiero saber cuántos metros separan {lugar1} de {lugar2}",
        "Calcula el buffer de {lugar1} hasta {lugar2}",
    ],
    "analizar_cobertura": [
        "¿Qué porcentaje del área tiene cobertura forestal?",
        "Analiza la cobertura de uso de suelo en {zona}",
        "¿Cuánta área urbana hay en {zona}?",
        "Dame el porcentaje de área verde en {zona}",
        "Calcula la cobertura vegetal de la región {zona}",
        "¿Qué tanto del territorio de {zona} es agrícola?",
        "Analiza el uso de suelo en {zona}",
        "¿Qué cobertura predomina en {zona}?",
        "Muéstrame el análisis de cobertura del suelo",
        "Porcentaje de cobertura forestal en {zona}",
        "¿Cuánta área tiene cobertura de agua en {zona}?",
        "Calcula la superficie construida en {zona}",
        "Análisis de cobertura y uso del suelo para {zona}",
        "¿Qué fracción del área está deforestada en {zona}?",
        "Cobertura de vegetación en la zona {zona}",
    ],
    "visualizar_mapa": [
        "Muéstrame el mapa de {tipo_mapa} en {zona}",
        "Visualiza la capa de {tipo_mapa}",
        "Despliega el mapa de densidad en {zona}",
        "Quiero ver el mapa de {tipo_mapa}",
        "Genera una visualización de {tipo_mapa}",
        "Renderiza el mapa de calor de {tipo_mapa}",
        "Muestra el mapa con las capas activas",
        "Visualiza los datos de {tipo_mapa} en el mapa",
        "Crea un mapa temático de {tipo_mapa}",
        "Despliega el mapa coroplético de {tipo_mapa}",
        "Ver mapa de {tipo_mapa} para la región {zona}",
        "Muéstrame la distribución espacial de {tipo_mapa}",
        "Genera mapa de {tipo_mapa} en ArcGIS",
        "Visualización espacial de {tipo_mapa}",
        "Activa la vista de mapa para {tipo_mapa}",
    ],
    "buscar_ubicacion": [
        "¿Dónde está {lugar1}?",
        "Encuentra la ubicación de {lugar1}",
        "Geocodifica la dirección {lugar1}",
        "¿Cuáles son las coordenadas de {lugar1}?",
        "Localiza {lugar1} en el mapa",
        "Busca {lugar1} en el sistema",
        "Dame la latitud y longitud de {lugar1}",
        "¿En qué zona queda {lugar1}?",
        "Encuentra las coordenadas GPS de {lugar1}",
        "Localiza el punto {lugar1} en el mapa",
        "¿Dónde se encuentra exactamente {lugar1}?",
        "Busca la posición geográfica de {lugar1}",
        "¿En qué provincia está {lugar1}?",
        "Ubica {lugar1} en el sistema GIS",
        "Obtén las coordenadas de {lugar1}",
    ],
    "consultar_capa": [
        "¿Qué capas tengo disponibles en ArcGIS?",
        "Lista todas las capas del proyecto",
        "¿Qué capas están activas?",
        "Muéstrame las capas disponibles",
        "¿Cuántas capas tiene el mapa?",
        "Lista las capas de {tipo_capa} disponibles",
        "¿Qué información tiene la capa {tipo_capa}?",
        "Describe los atributos de la capa {tipo_capa}",
        "¿La capa {tipo_capa} está cargada?",
        "Consulta los metadatos de la capa {tipo_capa}",
        "¿Cuáles son los campos de la capa {tipo_capa}?",
        "Dame información sobre la capa {tipo_capa}",
        "¿Qué capas vectoriales hay disponibles?",
        "Muéstrame las capas raster del proyecto",
        "¿Existe la capa de {tipo_capa} en el proyecto?",
    ],
    "exportar_datos": [
        "Exporta los datos en formato {formato}",
        "Descarga la capa {tipo_capa} como {formato}",
        "Guarda el análisis en formato {formato}",
        "Exporta el mapa a {formato}",
        "¿Cómo exporto los resultados a {formato}?",
        "Genera un archivo {formato} con los datos",
        "Descarga los resultados en {formato}",
        "Exporta la capa a shapefile",
        "Guarda los datos como GeoJSON",
        "Exporta el análisis a CSV",
        "Convierte la capa a formato {formato}",
        "Descarga el mapa en {formato}",
        "Exportar datos geoespaciales a {formato}",
        "Genera shapefile con los resultados",
        "Guarda el proyecto en formato {formato}",
    ],
    "detectar_cambios": [
        "¿Qué cambios hubo en {zona} entre {año1} y {año2}?",
        "Detecta cambios en la cobertura de {zona} del {año1} al {año2}",
        "Analiza la evolución temporal de {zona}",
        "¿Cómo ha cambiado {zona} en los últimos años?",
        "Compara {zona} en {año1} vs {año2}",
        "Detecta cambios de uso de suelo en {zona}",
        "Análisis multitemporal de {zona} entre {año1} y {año2}",
        "¿Ha habido deforestación en {zona}?",
        "Muéstrame los cambios ocurridos en {zona}",
        "Identifica áreas de cambio en {zona} entre {año1} y {año2}",
        "¿Qué zona cambió más entre {año1} y {año2}?",
        "Análisis de cambio temporal en {zona}",
        "Detecta expansión urbana en {zona} desde {año1}",
        "¿Cuánto creció la ciudad en {zona} desde {año1}?",
        "Compara imágenes satelitales de {zona} en {año1} y {año2}",
    ],
    "generar_reporte": [
        "Genera un reporte del análisis de {tipo_reporte}",
        "Crea un informe de {tipo_reporte}",
        "Dame un reporte completo de {tipo_reporte}",
        "Genera el resumen ejecutivo de {tipo_reporte}",
        "Exporta el reporte de {tipo_reporte} en PDF",
        "Crea documentación del análisis de {tipo_reporte}",
        "Genera un reporte automático de {tipo_reporte}",
        "¿Puedes crear un informe de {tipo_reporte}?",
        "Necesito un reporte técnico de {tipo_reporte}",
        "Genera reporte estadístico de {tipo_reporte}",
        "Crea el informe final de {tipo_reporte}",
        "Genera un reporte de {tipo_reporte} para el cliente",
        "Dame el reporte de resultados de {tipo_reporte}",
        "Crea reporte ejecutivo de {tipo_reporte}",
        "Documenta el análisis de {tipo_reporte}",
    ],
}

# =============================================================================
# VARIABLES DE RELLENO PARA TEMPLATES
# =============================================================================

FILL_VARS = {
    "lugar1": [
        "Quito", "Guayaquil", "Cuenca", "Ambato", "Loja",
        "el Hospital Metropolitano", "la Mitad del Mundo", "el Aeropuerto Mariscal Sucre",
        "el Parque La Carolina", "el estadio Monumental", "la Plaza Grande",
        "Santo Domingo", "Manta", "Esmeraldas", "Ibarra",
    ],
    "lugar2": [
        "Guayaquil", "Cuenca", "el centro histórico", "la terminal terrestre",
        "el aeropuerto", "el puerto marítimo", "la refinería", "Manta",
        "Ambato", "Riobamba", "Latacunga", "Tulcán", "Macas",
    ],
    "zona": [
        "la zona norte", "la región amazónica", "la costa ecuatoriana",
        "la sierra central", "el distrito metropolitano", "la zona sur",
        "el litoral", "la región andina", "el oriente", "Galápagos",
        "Pichincha", "Guayas", "Manabí", "Azuay", "El Oro",
    ],
    "tipo_mapa": [
        "densidad poblacional", "riesgo sísmico", "uso de suelo",
        "cobertura vegetal", "temperatura", "precipitación",
        "elevación", "pendientes", "vialidad", "hidrografía",
        "zonas de inundación", "riesgo volcánico", "áreas protegidas",
    ],
    "tipo_capa": [
        "polígonos", "puntos de interés", "vías", "ríos", "límites administrativos",
        "curvas de nivel", "imágenes satelitales", "edificaciones", "predios",
        "zonas de riesgo", "áreas verdes", "infraestructura",
    ],
    "formato": [
        "shapefile", "GeoJSON", "CSV", "KML", "PDF", "GeoTIFF",
        "Excel", "GeoPackage", "DWG", "JSON",
    ],
    "año1": ["2015", "2016", "2017", "2018", "2019", "2020"],
    "año2": ["2021", "2022", "2023", "2024", "2025"],
    "tipo_reporte": [
        "riesgo sísmico", "cobertura forestal", "uso de suelo",
        "análisis de inundaciones", "expansión urbana", "calidad del aire",
        "vulnerabilidad territorial", "cambio climático", "biodiversidad",
        "infraestructura vial", "análisis demográfico",
    ],
}


def fill_template(template: str) -> str:
    """Rellena un template con variables aleatorias."""
    pattern = re.compile(r"\{(\w+)\}")
    matches = pattern.findall(template)
    result = template
    for var in matches:
        if var in FILL_VARS:
            value = random.choice(FILL_VARS[var])
            result = result.replace(f"{{{var}}}", value, 1)
    return result


def generate_dataset(samples_per_intent: int = 100) -> pd.DataFrame:
    """
    Genera un dataset sintético balanceado de consultas geoespaciales.

    Args:
        samples_per_intent: Número de muestras por clase (default=100)

    Returns:
        DataFrame con columnas ['text', 'intent', 'intent_id']
    """
    records = []
    intent_to_id = {intent: idx for idx, intent in enumerate(INTENT_TEMPLATES.keys())}

    for intent, templates in INTENT_TEMPLATES.items():
        generated = set()
        attempts = 0
        max_attempts = samples_per_intent * 20

        while len(generated) < samples_per_intent and attempts < max_attempts:
            template = random.choice(templates)
            filled = fill_template(template)
            filled_clean = filled.strip()
            if filled_clean not in generated:
                generated.add(filled_clean)
                records.append({
                    "text": filled_clean,
                    "intent": intent,
                    "intent_id": intent_to_id[intent],
                })
            attempts += 1

        # Si no alcanzamos el objetivo, completar con variantes numeradas
        extra = samples_per_intent - len(generated)
        if extra > 0:
            for i in range(extra):
                template = random.choice(templates)
                filled = fill_template(template) + f" (v{i})"
                records.append({
                    "text": filled,
                    "intent": intent,
                    "intent_id": intent_to_id[intent],
                })

    df = pd.DataFrame(records).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, path: str = "data/raw/dataset_raw.csv") -> None:
    """Guarda el dataset en disco."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"✅ Dataset guardado en: {path}")
    print(f"   Total muestras: {len(df)}")
    print(f"   Clases: {df['intent'].nunique()}")
    print(f"   Distribución:\n{df['intent'].value_counts().to_string()}")


if __name__ == "__main__":
    print("🔨 Generando dataset sintético geoespacial...")
    df = generate_dataset(samples_per_intent=100)
    save_dataset(df, "data/raw/dataset_raw.csv")
