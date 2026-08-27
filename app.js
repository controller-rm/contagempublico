/* Contador de Pessoas - Auditório
 * Detecção de pessoas 100% no navegador usando TensorFlow.js + modelo coco-ssd.
 * Nenhum vídeo/frame é enviado para servidor algum.
 */

const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const countEl = document.getElementById("count");
const statusEl = document.getElementById("status");
const capacityInput = document.getElementById("capacity");
const occupancyEl = document.getElementById("occupancy");
const switchBtn = document.getElementById("switchCamera");
const installBtn = document.getElementById("installBtn");
const historyCanvas = document.getElementById("history");
const hctx = historyCanvas.getContext("2d");
const modelSelect = document.getElementById("modelSelect");
const sensitivitySlider = document.getElementById("sensitivity");
const sensitivityValueEl = document.getElementById("sensitivityValue");
const eventsListEl = document.getElementById("eventsList");
const exportBtn = document.getElementById("exportBtn");
const clearReportBtn = document.getElementById("clearReportBtn");

const DETECTION_INTERVAL_MS = 250; // pausa entre inferências
const TILE_OVERLAP = 0.18;
const TILE_TTL_MS = 3000;
const NMS_IOU_THRESHOLD = 0.45;
const MAX_HISTORY_POINTS = 40;
const SMOOTHING_WINDOW = 3; // nº de leituras usadas para suavizar o contador exibido
const MAX_LOG_ROWS = 2000; // limite do registro completo (para exportação)

let model = null;
let currentStream = null;
let facingMode = "environment";
let running = false;
let history = []; // pontos exibidos no mini-gráfico (contagem suavizada)
let rawBuffer = []; // últimas leituras brutas, usadas para suavizar
let fullLog = []; // registro completo p/ exportar em CSV: {time, count, pct}
let events = []; // eventos de limiar (70% / 90% / 100%) exibidos na tela
let lastTier = 0; // último limiar de ocupação já disparado (0, 70, 90, 100)
let deferredInstallPrompt = null;
let SCORE_THRESHOLD = parseFloat(sensitivitySlider.value);
let detectionGeneration = 0;
let tileIndex = 0;
let tileCache = new Map();
const tileCanvas = document.createElement("canvas");
const tileCtx = tileCanvas.getContext("2d", { willReadFrequently: true });

function setStatus(text) {
  statusEl.textContent = text;
}

function nowLabel() {
  return new Date().toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

async function loadModel() {
  running = false;
  setStatus("Carregando modelo de IA (primeira vez pode demorar alguns segundos)...");
  model = await cocoSsd.load({ base: modelSelect.value });
  setStatus("Modelo carregado.");
}

async function startCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
  }
  try {
    currentStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    video.srcObject = currentStream;
    await video.play();

    // aguarda dimensões reais do vídeo antes de dimensionar o canvas
    await new Promise((resolve) => {
      if (video.videoWidth) return resolve();
      video.onloadedmetadata = () => resolve();
    });

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  } catch (err) {
    setStatus(
      "Não foi possível acessar a câmera (" + err.message + "). Verifique as permissões do navegador."
    );
    throw err;
  }
}

function drawDetections(predictions) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  let count = 0;

  predictions.forEach((p) => {
    if (p.class === "person" && p.score >= SCORE_THRESHOLD) {
      count++;
      const [x, y, w, h] = p.bbox;

      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = Math.max(2, canvas.width / 400);
      ctx.strokeRect(x, y, w, h);

      ctx.fillStyle = "rgba(34, 211, 238, 0.12)";
      ctx.fillRect(x, y, w, h);

      const label = Math.round(p.score * 100) + "%";
      ctx.font = "14px sans-serif";
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = "#22d3ee";
      ctx.fillRect(x, Math.max(0, y - 18), textWidth + 8, 18);
      ctx.fillStyle = "#0f172a";
      ctx.fillText(label, x + 4, Math.max(13, y - 5));
    }
  });

  return count;
}

function boxOverlap(a, b) {
  const [ax, ay, aw, ah] = a.bbox;
  const [bx, by, bw, bh] = b.bbox;
  const left = Math.max(ax, bx);
  const top = Math.max(ay, by);
  const right = Math.min(ax + aw, bx + bw);
  const bottom = Math.min(ay + ah, by + bh);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = aw * ah + bw * bh - intersection;
  const smallerArea = Math.min(aw * ah, bw * bh);
  return {
    iou: union > 0 ? intersection / union : 0,
    containment: smallerArea > 0 ? intersection / smallerArea : 0,
  };
}

function removeDuplicatePeople(predictions) {
  const people = predictions
    .filter((p) => p.class === "person" && p.score >= SCORE_THRESHOLD)
    .sort((a, b) => b.score - a.score);
  const kept = [];
  people.forEach((candidate) => {
    const isDuplicate = kept.some((existing) => {
      const overlap = boxOverlap(candidate, existing);
      return overlap.iou > NMS_IOU_THRESHOLD || overlap.containment > 0.7;
    });
    if (!isDuplicate) {
      kept.push(candidate);
    }
  });
  return kept;
}

function getScanRegions() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  const regionW = w * (0.5 + TILE_OVERLAP / 2);
  const regionH = h * (0.5 + TILE_OVERLAP / 2);
  return [
    { x: 0, y: 0, w: regionW, h: regionH },
    { x: w - regionW, y: 0, w: regionW, h: regionH },
    { x: 0, y: h - regionH, w: regionW, h: regionH },
    { x: w - regionW, y: h - regionH, w: regionW, h: regionH },
  ];
}

async function detectPeopleMultiScale() {
  const fullFrame = await model.detect(video, 50, SCORE_THRESHOLD);
  const regions = getScanRegions();
  const currentTile = tileIndex % regions.length;
  const region = regions[currentTile];
  tileIndex++;

  tileCanvas.width = Math.round(region.w);
  tileCanvas.height = Math.round(region.h);
  tileCtx.drawImage(video, region.x, region.y, region.w, region.h, 0, 0, tileCanvas.width, tileCanvas.height);

  const tilePredictions = await model.detect(tileCanvas, 50, SCORE_THRESHOLD);
  const mapped = tilePredictions.map((p) => ({
    ...p,
    bbox: [p.bbox[0] + region.x, p.bbox[1] + region.y, p.bbox[2], p.bbox[3]],
  }));
  tileCache.set(currentTile, { timestamp: performance.now(), predictions: mapped });

  const now = performance.now();
  const recentTiles = [];
  tileCache.forEach((entry, key) => {
    if (now - entry.timestamp <= TILE_TTL_MS) recentTiles.push(...entry.predictions);
    else tileCache.delete(key);
  });
  return removeDuplicatePeople([...fullFrame, ...recentTiles]);
}

// suaviza o número exibido usando a média das últimas leituras — evita que o
// contador "pisque" entre valores muito diferentes de um frame para o outro
function smoothCount(rawCount) {
  rawBuffer.push(rawCount);
  if (rawBuffer.length > SMOOTHING_WINDOW) rawBuffer.shift();
  const avg = rawBuffer.reduce((a, b) => a + b, 0) / rawBuffer.length;
  return Math.round(avg);
}

function logEvent(text, level) {
  events.unshift({ time: nowLabel(), text, level });
  events = events.slice(0, 8);
  renderEvents();
}

function renderEvents() {
  if (events.length === 0) {
    eventsListEl.innerHTML = '<li class="event-empty">Nenhum evento ainda.</li>';
    return;
  }
  eventsListEl.innerHTML = events
    .map(
      (e) =>
        `<li class="event event-${e.level}"><span class="event-time">${e.time}</span>${e.text}</li>`
    )
    .join("");
}

function checkOccupancyEvents(count, capacity, pct) {
  if (capacity <= 0) return;

  if (pct >= 100 && lastTier < 100) {
    logEvent(`Capacidade máxima atingida (${count}/${capacity})`, "danger");
    lastTier = 100;
  } else if (pct >= 90 && lastTier < 90) {
    logEvent(`Ocupação crítica: ${pct}% (${count}/${capacity})`, "warn");
    lastTier = 90;
  } else if (pct >= 70 && lastTier < 70) {
    logEvent(`Ocupação alta: ${pct}% (${count}/${capacity})`, "warn");
    lastTier = 70;
  } else if (pct < 70 && lastTier !== 0) {
    // ocupação caiu — rearma os limiares para poderem disparar de novo depois
    lastTier = 0;
  }
}

function updateCountUI(rawCount) {
  const count = smoothCount(rawCount);
  countEl.textContent = count;

  const capacity = parseInt(capacityInput.value, 10) || 0;
  let pct = 0;
  if (capacity > 0) {
    pct = Math.min(100, Math.round((count / capacity) * 100));
    let color = "#4ade80";
    if (pct >= 90) color = "#f87171";
    else if (pct >= 70) color = "#facc15";
    occupancyEl.textContent = pct + "% da capacidade (" + count + " / " + capacity + ")";
    occupancyEl.style.color = color;
    checkOccupancyEvents(count, capacity, pct);
  } else {
    occupancyEl.textContent = "";
  }

  history.push(count);
  if (history.length > MAX_HISTORY_POINTS) history.shift();
  drawHistory();

  fullLog.push({ time: new Date().toISOString(), count, pct });
  if (fullLog.length > MAX_LOG_ROWS) fullLog.shift();
}

function drawHistory() {
  const w = (historyCanvas.width = historyCanvas.clientWidth);
  const h = (historyCanvas.height = historyCanvas.clientHeight);
  hctx.clearRect(0, 0, w, h);

  if (history.length < 2) return;

  const max = Math.max(...history, 1);
  hctx.strokeStyle = "#22d3ee";
  hctx.lineWidth = 2;
  hctx.beginPath();

  history.forEach((count, i) => {
    const x = (i / (MAX_HISTORY_POINTS - 1)) * w;
    const y = h - (count / max) * (h - 12) - 6;
    if (i === 0) hctx.moveTo(x, y);
    else hctx.lineTo(x, y);
  });

  hctx.stroke();

  hctx.fillStyle = "#22d3ee";
  history.forEach((count, i) => {
    const x = (i / (MAX_HISTORY_POINTS - 1)) * w;
    const y = h - (count / max) * (h - 12) - 6;
    hctx.beginPath();
    hctx.arc(x, y, 2, 0, Math.PI * 2);
    hctx.fill();
  });
}

async function detectLoop(generation) {
  if (!running || generation !== detectionGeneration) return;

  try {
    if (video.readyState >= video.HAVE_ENOUGH_DATA) {
      const predictions = await detectPeopleMultiScale();
      if (!running || generation !== detectionGeneration) return;
      const count = drawDetections(predictions);
      updateCountUI(count);
    }
  } catch (err) {
    console.error("Erro na detecção:", err);
  }

  setTimeout(() => detectLoop(generation), DETECTION_INTERVAL_MS);
}

function startDetectionLoop() {
  tileCache.clear();
  tileIndex = 0;
  rawBuffer = [];
  running = true;
  detectionGeneration++;
  detectLoop(detectionGeneration);
}

async function init() {
  try {
    await loadModel();
    await startCamera();
    startDetectionLoop();
    setStatus("Detectando pessoas em tempo real...");
  } catch (err) {
    console.error(err);
  }
}

switchBtn.addEventListener("click", async () => {
  running = false;
  detectionGeneration++;
  facingMode = facingMode === "environment" ? "user" : "environment";
  setStatus("Trocando câmera...");
  try {
    await startCamera();
    startDetectionLoop();
    setStatus("Detectando pessoas em tempo real...");
  } catch (err) {
    // mensagem de erro já definida em startCamera
  }
});

modelSelect.addEventListener("change", async () => {
  setStatus("Trocando modelo de IA...");
  rawBuffer = [];
  try {
    await loadModel();
    startDetectionLoop();
    setStatus("Detectando pessoas em tempo real...");
  } catch (err) {
    console.error(err);
  }
});

sensitivitySlider.addEventListener("input", () => {
  SCORE_THRESHOLD = parseFloat(sensitivitySlider.value);
  sensitivityValueEl.textContent = SCORE_THRESHOLD.toFixed(2);
});

exportBtn.addEventListener("click", () => {
  if (fullLog.length === 0) {
    setStatus("Ainda não há leituras para exportar.");
    return;
  }
  const header = "data_hora,pessoas,ocupacao_pct\n";
  const rows = fullLog
    .map((r) => `${r.time},${r.count},${r.pct}`)
    .join("\n");
  const blob = new Blob([header + rows], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  a.href = url;
  a.download = `relatorio-auditorio-${stamp}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

clearReportBtn.addEventListener("click", () => {
  fullLog = [];
  events = [];
  lastTier = 0;
  renderEvents();
  setStatus("Registro de relatório limpo.");
});

// --- Instalação do PWA ---
window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  deferredInstallPrompt = event;
  installBtn.style.display = "inline-block";
});

installBtn.addEventListener("click", async () => {
  if (!deferredInstallPrompt) return;
  deferredInstallPrompt.prompt();
  await deferredInstallPrompt.userChoice;
  deferredInstallPrompt = null;
  installBtn.style.display = "none";
});

window.addEventListener("appinstalled", () => {
  installBtn.style.display = "none";
});

// --- Service worker + inicialização ---
window.addEventListener("load", () => {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch((err) => {
      console.warn("Falha ao registrar service worker:", err);
    });
  }
  renderEvents();
  init();
});
