/**
 * GeoIntent Classifier — Frontend App
 * Conecta con la API FastAPI para clasificación de intención geoespacial.
 */

const API_BASE = (window.ENV_API_URL || "http://localhost:8000") + "/api/v1";

// Datos de los intents para la tabla y ejemplos
const INTENTS_DATA = [
  {
    intent: "calcular_distancia",
    icon: "📏",
    agent: "Agente de Geometría",
    example: "¿Cuántos km hay entre Quito y Guayaquil?",
    color: "#4f8ef7",
  },
  {
    intent: "analizar_cobertura",
    icon: "🌿",
    agent: "Agente de Análisis Espacial",
    example: "¿Qué porcentaje del área tiene cobertura forestal?",
    color: "#22c55e",
  },
  {
    intent: "visualizar_mapa",
    icon: "🗺️",
    agent: "Agente de Visualización",
    example: "Muéstrame el mapa de densidad poblacional",
    color: "#6ee7b7",
  },
  {
    intent: "buscar_ubicacion",
    icon: "📍",
    agent: "Agente de Geocodificación",
    example: "¿Dónde está el Hospital Metropolitano?",
    color: "#f59e0b",
  },
  {
    intent: "consultar_capa",
    icon: "📂",
    agent: "Agente de Gestión de Capas",
    example: "¿Qué capas tengo disponibles en ArcGIS?",
    color: "#a78bfa",
  },
  {
    intent: "exportar_datos",
    icon: "💾",
    agent: "Agente de Exportación",
    example: "Exporta los datos en formato shapefile",
    color: "#f87171",
  },
  {
    intent: "detectar_cambios",
    icon: "🔄",
    agent: "Agente de Análisis Temporal",
    example: "¿Qué cambios hubo en la zona norte entre 2020 y 2024?",
    color: "#38bdf8",
  },
  {
    intent: "generar_reporte",
    icon: "📄",
    agent: "Agente de Reportes",
    example: "Genera un reporte del análisis de riesgo sísmico",
    color: "#fb923c",
  },
];

// ================================================================
// INICIALIZACIÓN
// ================================================================

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  renderExamples();
  renderIntentsTable();

  // Enter para clasificar
  document.getElementById("queryInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      classify();
    }
  });
});

// ================================================================
// HEALTH CHECK
// ================================================================

async function checkHealth() {
  const dot = document.getElementById("statusDot");
  const label = document.getElementById("statusLabel");
  try {
    const res = await fetch(API_BASE.replace("/api/v1", "/health"));
    const data = await res.json();
    if (data.model_loaded) {
      dot.className = "status-dot ok";
      label.textContent = "API conectada · Modelo cargado";
    } else {
      dot.className = "status-dot error";
      label.textContent = "Modelo no cargado";
    }
  } catch {
    dot.className = "status-dot error";
    label.textContent = "API no disponible";
  }
}

// ================================================================
// CLASIFICACIÓN
// ================================================================

async function classify() {
  const input = document.getElementById("queryInput");
  const btn = document.getElementById("classifyBtn");
  const text = input.value.trim();

  if (!text) {
    input.focus();
    return;
  }

  // Estado de carga
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Clasificando...';

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Error del servidor");
    }

    const data = await res.json();
    renderResult(data);

  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "<span>⚡ Clasificar Intent</span>";
  }
}

// ================================================================
// RENDER RESULTADO
// ================================================================

function renderResult(data) {
  const intentInfo = INTENTS_DATA.find((i) => i.intent === data.intent) || {};

  // Mostrar tarjeta
  const card = document.getElementById("resultCard");
  card.style.display = "block";
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });

  // Intent badge
  document.getElementById("intentIcon").textContent = intentInfo.icon || "🤖";
  document.getElementById("intentValue").textContent = data.intent;
  document.getElementById("intentBadge").style.borderColor =
    (intentInfo.color || "#4f8ef7") + "55";

  // Agente
  document.getElementById("agentValue").textContent = data.agent_action;

  // Confianza
  const confBox = document.getElementById("confidenceBox");
  if (data.confidence !== null) {
    confBox.style.display = "block";
    const pct = Math.round(data.confidence * 100);
    document.getElementById("confBar").style.width = `${pct}%`;
    document.getElementById("confPct").textContent = `${pct}%`;
  } else {
    confBox.style.display = "none";
  }

  // Probabilidades
  const probaSection = document.getElementById("probaSection");
  const probaChart = document.getElementById("probaChart");

  if (data.probabilities && Object.keys(data.probabilities).length > 0) {
    probaSection.style.display = "block";
    probaChart.innerHTML = "";

    const sorted = Object.entries(data.probabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6);

    sorted.forEach(([intent, prob]) => {
      const info = INTENTS_DATA.find((i) => i.intent === intent) || {};
      const pct = Math.round(prob * 100);
      const row = document.createElement("div");
      row.className = "proba-row";
      row.innerHTML = `
        <span class="proba-name">${info.icon || ""} ${intent}</span>
        <div class="proba-bar-wrap">
          <div class="proba-bar-fill" style="width:${pct}%;background:${info.color || "#4f8ef7"}"></div>
        </div>
        <span class="proba-pct">${pct}%</span>
      `;
      probaChart.appendChild(row);
    });
  } else {
    probaSection.style.display = "none";
  }
}

function showError(message) {
  const card = document.getElementById("resultCard");
  card.style.display = "block";
  card.innerHTML = `
    <h2 class="card-title">⚠️ Error</h2>
    <p style="color:#ef4444;">${message}</p>
    <p style="color:#94a3b8;font-size:0.85rem;margin-top:8px;">
      Asegúrate de que la API esté corriendo: <code>docker-compose up</code>
    </p>
  `;
}

// ================================================================
// EJEMPLOS RÁPIDOS
// ================================================================

function renderExamples() {
  const grid = document.getElementById("examplesGrid");
  INTENTS_DATA.forEach((item) => {
    const chip = document.createElement("button");
    chip.className = "example-chip";
    chip.textContent = `${item.icon} ${item.example}`;
    chip.onclick = () => {
      document.getElementById("queryInput").value = item.example;
      classify();
    };
    grid.appendChild(chip);
  });
}

// ================================================================
// TABLA DE INTENTS
// ================================================================

function renderIntentsTable() {
  const tbody = document.getElementById("intentsBody");
  INTENTS_DATA.forEach((item) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="td-intent">${item.icon} ${item.intent}</td>
      <td class="td-agent">${item.agent}</td>
      <td class="td-example">"${item.example}"</td>
    `;
    tbody.appendChild(tr);
  });
}
