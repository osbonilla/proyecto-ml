# 🌍 GeoIntent Classifier

> **Clasificador de Intención para Arquitectura Multiagente de Análisis Geoespacial en ArcGIS**
>
> Pipeline completo de Machine Learning para routing inteligente de consultas en lenguaje natural hacia agentes especializados en análisis geoespacial.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)

---

## 📋 Tabla de Contenidos

- [Introducción](#-introducción)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Intents y Agentes](#-intents-y-agentes)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Pipeline de ML](#-pipeline-de-ml)
- [Instalación y Uso](#-instalación-y-uso)
- [Docker](#-docker)
- [API REST](#-api-rest)
- [Evaluación y Resultados](#-evaluación-y-resultados)
- [Referencias](#-referencias)

---

## 🎯 Introducción

El presente proyecto implementa un **clasificador de intención** basado en Machine Learning clásico, diseñado como componente de routing inteligente para una arquitectura multiagente de análisis geoespacial en ArcGIS.

### Motivación

En sistemas de análisis geoespacial modernos, los usuarios interactúan mediante lenguaje natural con múltiples agentes especializados. El desafío central es determinar, dada una consulta del usuario, **qué agente debe responder y ejecutar la acción correspondiente**.

Sin un clasificador de intención preciso, el sistema multiagente no puede enrutar correctamente las consultas, generando respuestas incorrectas o activación de agentes equivocados. Este problema es equivalente a resolver:

```
f(consulta_usuario) → intent → agente_geoespacial → acción
```

### Relevancia

- **Tesis de referencia**: Planificación jerárquica de una arquitectura multiagente para análisis geoespacial en ArcGIS
- **Dominio**: Procesamiento de Lenguaje Natural (NLP) + Sistemas Geoespaciales
- **Impacto**: Permite que el sistema multiagente procese consultas en español de manera autónoma y precisa

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                    SISTEMA MULTIAGENTE                  │
│                                                         │
│  Usuario                                                │
│     │                                                   │
│     ▼                                                   │
│  ┌──────────────────────────────────────────┐           │
│  │         GeoIntent Classifier             │           │
│  │                                          │           │
│  │  Texto → Preprocesamiento → TF-IDF       │           │
│  │       → Chi² Selection → LR/SVM          │           │
│  │       → Intent                           │           │
│  └──────────────────┬───────────────────────┘           │
│                     │                                   │
│         ┌───────────┼───────────┐                       │
│         ▼           ▼           ▼                       │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│   │ Agente   │ │ Agente   │ │ Agente   │  ...           │
│   │Geometría │ │Cobertura │ │  Mapas   │                │
│   └──────────┘ └──────────┘ └──────────┘                │
└─────────────────────────────────────────────────────────┘
```

### Componentes

| Componente | Tecnología | Descripción |
|-----------|-----------|-------------|
| **Clasificador ML** | scikit-learn | TF-IDF + Chi² + LR/SVM |
| **API REST** | FastAPI | Endpoint de predicción |
| **Frontend** | HTML/CSS/JS | Interfaz de demostración |
| **Orquestación** | Docker Compose | Backend + Frontend |

---

## 🤖 Intents y Agentes

El clasificador detecta **8 intents geoespaciales**, cada uno mapeado a un agente especializado:

| Intent | Agente | Ejemplo de consulta |
|--------|--------|-------------------|
| `calcular_distancia` | Agente de Geometría | "¿Cuántos km hay entre Quito y Guayaquil?" |
| `analizar_cobertura` | Agente de Análisis Espacial | "¿Qué porcentaje del área es forestal?" |
| `visualizar_mapa` | Agente de Visualización | "Muéstrame el mapa de densidad poblacional" |
| `buscar_ubicacion` | Agente de Geocodificación | "¿Dónde está el Hospital Metropolitano?" |
| `consultar_capa` | Agente de Gestión de Capas | "¿Qué capas tengo en ArcGIS?" |
| `exportar_datos` | Agente de Exportación | "Exporta los datos como shapefile" |
| `detectar_cambios` | Agente de Análisis Temporal | "¿Qué cambios hubo entre 2020 y 2024?" |
| `generar_reporte` | Agente de Reportes | "Genera el reporte de riesgo sísmico" |

---

## Estructura del Proyecto

```
geo-intent-classifier/
│
├── 📁 data/
│   ├── raw/                    # Dataset sintético original (800 muestras)
│   ├── processed/              # Dataset limpio tras preprocesamiento
│   └── augmented/              # Dataset con data augmentation (opcional)
│
├── 📁 notebooks/               # Jupyter Notebooks del análisis
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training_comparison.ipynb   # LR vs SVM
│   ├── 04_optimization.ipynb   # GridSearchCV + curvas
│   └── 05_tsne_visualization.ipynb
│
├── 📁 src/
│   ├── data/
│   │   ├── dataset_builder.py  # Generador de dataset sintético
│   │   └── preprocessor.py     # Limpieza y normalización de texto
│   ├── features/
│   │   ├── tfidf_extractor.py  # TF-IDF con n-gramas
│   │   └── feature_selector.py # Chi², Mutual Info, Varianza
│   ├── models/
│   │   ├── pipeline_builder.py # Construcción de pipelines sklearn
│   │   ├── trainer.py          # RepeatedKFold + GridSearchCV
│   │   └── evaluator.py        # Métricas, CM, Wilcoxon test
│   └── visualization/
│       ├── eda_plots.py        # Gráficos de EDA
│       └── tsne_plot.py        # Visualización T-SNE
│
├── 📁 models/saved/            # Modelos serializados (.pkl)
│
├── 📁 reports/
│   ├── figures/                # Gráficos generados
│   └── metrics/                # CSV y JSON con resultados
│
├── 📁 backend/
│   ├── app.py                  # FastAPI application
│   ├── routes/predict.py       # Endpoints de predicción
│   └── schemas/request_models.py # Pydantic schemas
│
├── 📁 frontend/                # Interfaz web
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── 📁 docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
│
├── docker-compose.yml
├── main.py                     # Entry point CLI
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧠 Pipeline de ML

### 1. Dataset

- **Tamaño**: 800 muestras sintéticas en español (100 por clase)
- **Clases**: 8 intents balanceados
- **Generación**: Templates con variables geoespaciales del Ecuador

### 2. Preprocesamiento de Texto

```
Texto original
    ↓
Minúsculas
    ↓
Eliminación de caracteres especiales
    ↓
Eliminación de stopwords (español)
    ↓
Texto limpio
```

### 3. Feature Extraction — TF-IDF

```python
TfidfVectorizer(
    ngram_range=(1, 2),     # Unigramas + Bigramas
    max_features=5000,
    sublinear_tf=True,       # log(1+tf) → mejor para texto corto
    min_df=2,
    max_df=0.95,
)
```

**¿Por qué unigramas + bigramas?**
- Unigramas: "calcular", "distancia", "mapa"
- Bigramas: "calcular_distancia", "uso_suelo" → capturan frases clave del dominio

### 4. Feature Selection — Chi²

```python
SelectKBest(score_func=chi2, k=3000)
```

Reduce de ~5000 → 3000 features eliminando términos no discriminativos.

### 5. Modelos Comparados

| Modelo | Ventajas | Desventajas |
|--------|----------|-------------|
| **Logistic Regression** | Probabilístico, interpretable, rápido | Puede no capturar no-linealidades |
| **LinearSVC** | Mejor margen, eficiente en alta dim. | No da probabilidades directamente |

### 6. Validación — RepeatedStratifiedKFold

```python
RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
# = 15 evaluaciones por modelo
# Métrica: F1-Macro (penaliza clases desbalanceadas)
```

### 7. Comparación Estadística — Wilcoxon

Test de Wilcoxon signed-rank (no paramétrico) para comparar LR vs SVM:
- H₀: No hay diferencia significativa entre los modelos
- H₁: Existe diferencia significativa
- α = 0.05

---

## 🚀 Instalación y Uso

### Prerrequisitos

- Python 3.11+
- Docker + Docker Compose (opcional pero recomendado)

### Opción A — Local (sin Docker)

```bash
# 1. Clonar repositorio
git clone https://github.com/TU_USUARIO/geo-intent-classifier.git
cd geo-intent-classifier

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Copiar variables de entorno
cp .env.example .env

# 5. Entrenar el modelo
python main.py --train --eval

# 6. Levantar la API
python main.py --serve

# 7. Abrir el frontend en el navegador
# Abrir: frontend/index.html
```

### Opción B — Docker Compose (recomendado)

```bash
# 1. Copiar .env
cp .env.example .env

# 2. Entrenar el modelo primero (fuera de Docker)
pip install -r requirements.txt
python main.py --train --eval

# 3. Levantar todos los servicios
docker-compose up --build

# Frontend: http://localhost:80
# API:      http://localhost:8000
# Docs API: http://localhost:8000/docs
```

### Predicción vía CLI

```bash
python main.py --predict "¿Cuántos km hay entre Quito y Guayaquil?"
```

Output:
```json
{
  "text": "¿Cuántos km hay entre Quito y Guayaquil?",
  "intent": "calcular_distancia",
  "intent_id": 0,
  "confidence": 0.9821,
  "agent_action": "Activar Agente de Geometría → calcular ruta entre puntos"
}
```

---

## 🐳 Docker

### Servicios

| Servicio | Puerto | Descripción |
|---------|--------|-------------|
| `backend` | 8000 | FastAPI + Modelo ML |
| `frontend` | 80 | Nginx + HTML/CSS/JS |

### Comandos útiles

```bash
# Levantar todo
docker-compose up --build

# Solo backend
docker-compose up backend

# Ver logs
docker-compose logs -f backend

# Parar
docker-compose down

# Entrar al contenedor
docker exec -it geointent_backend bash
```

### Volúmenes persistentes

Los modelos, datos y reportes se montan como volúmenes para persistir entre reinicios del contenedor:
- `./models` → `/app/models`
- `./reports` → `/app/reports`
- `./data` → `/app/data`

---

## 🌐 API REST

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado de la API y modelo |
| `POST` | `/api/v1/predict` | Clasificar intent |
| `POST` | `/api/v1/predict/batch` | Clasificar múltiples consultas |
| `GET` | `/api/v1/intents` | Listar intents soportados |
| `GET` | `/docs` | Documentación Swagger |

### Ejemplo de uso

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Muéstrame el mapa de cobertura forestal"}'
```

Response:
```json
{
  "text": "Muéstrame el mapa de cobertura forestal",
  "intent": "visualizar_mapa",
  "intent_id": 2,
  "confidence": 0.8934,
  "probabilities": {
    "visualizar_mapa": 0.8934,
    "analizar_cobertura": 0.0721,
    "generar_reporte": 0.0345
  },
  "agent_action": "Activar Agente de Visualización → renderizar mapa temático"
}
```

---

## 📊 Evaluación y Resultados

### Métricas reportadas

- **Accuracy**: proporción de predicciones correctas
- **F1-Macro**: media de F1 por clase (sin ponderación) — métrica principal
- **F1-Weighted**: media de F1 ponderada por frecuencia de clase
- **Confusion Matrix**: visualización de errores por clase
- **Curva de aprendizaje**: detección de overfitting/underfitting

### Comparación estadística

El test de Wilcoxon signed-rank compara los 15 scores de CV entre LR y SVM para determinar si la diferencia en rendimiento es estadísticamente significativa (p < 0.05).

### Visualización T-SNE

La reducción dimensional TF-IDF → SVD(50D) → T-SNE(2D) permite verificar visualmente la separabilidad entre intents en el espacio de características.

---

## 📚 Referencias

```
[1] Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python."
    Journal of Machine Learning Research, 12, 2825-2830.
    URL: https://jmlr.org/papers/v12/pedregosa11a.html

[2] Joachims, T. (1998). "Text categorization with Support Vector Machines:
    Learning with many relevant features." ECML-98. Springer, Berlin.
    DOI: 10.1007/BFb0026683

[3] Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic
    text retrieval." Information Processing & Management, 24(5), 513-523.
    DOI: 10.1016/0306-4573(88)90021-0

[4] Demšar, J. (2006). "Statistical Comparisons of Classifiers over Multiple
    Data Sets." Journal of Machine Learning Research, 7, 1-30.
    URL: https://jmlr.org/papers/v7/demsar06a.html

[5] van der Maaten, L. & Hinton, G. (2008). "Visualizing Data using t-SNE."
    Journal of Machine Learning Research, 9, 2579-2605.
    URL: https://jmlr.org/papers/v9/vandermaaten08a.html

[6] FastAPI Documentation. (2024). Sebastián Ramírez.
    URL: https://fastapi.tiangolo.com

[7] Docker Documentation. (2024). Docker Inc.
    URL: https://docs.docker.com
```

---

## 👤 Autor

Desarrollado como proyecto de materia — alineado con tesis de grado:
*"Planificación Jerárquica de una Arquitectura Multiagente para Análisis Geoespacial en ArcGIS"*
