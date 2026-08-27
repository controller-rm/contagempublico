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

const DETECTION_INTERVAL_MS = 300; // throttle p/ não sobrecarregar dispositivos móveis
const SCORE_THRESHOLD = 0.5;
const MAX_HISTORY_POINTS = 40;

let model = null;
let currentStream = null;
let facingMode = "environment";
let running = false;
let history = [];
let deferredInstallPrompt = null;

function setStatus(text) {
  statusEl.textContent = text;
}

async function loadModel() {
  setStatus("Carregando modelo de IA (primeira vez pode demorar alguns segundos)...");
  model = await cocoSsd.load({ base: "lite_mobilenet_v2" });
  setStatus("Modelo carregado. Solicitando câmera...");
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

function updateCountUI(count) {
  countEl.textContent = count;

  const capacity = parseInt(capacityInput.value, 10) || 0;
  if (capacity > 0) {
    const pct = Math.min(100, Math.round((count / capacity) * 100));
    let color = "#4ade80";
    if (pct >= 90) color = "#f87171";
    else if (pct >= 70) color = "#facc15";
    occupancyEl.textContent = pct + "% da capacidade (" + count + " / " + capacity + ")";
    occupancyEl.style.color = color;
  } else {
    occupancyEl.textContent = "";
  }

  history.push(count);
  if (history.length > MAX_HISTORY_POINTS) history.shift();
  drawHistory();
}

function drawHistory() {
  const w = historyCanvas.width = historyCanvas.clientWidth;
  const h = historyCanvas.height = historyCanvas.clientHeight;
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

  // pontos
  hctx.fillStyle = "#22d3ee";
  history.forEach((count, i) => {
    const x = (i / (MAX_HISTORY_POINTS - 1)) * w;
    const y = h - (count / max) * (h - 12) - 6;
    hctx.beginPath();
    hctx.arc(x, y, 2, 0, Math.PI * 2);
    hctx.fill();
  });
}

async function detectLoop() {
  if (!running) return;

  try {
    if (video.readyState >= video.HAVE_ENOUGH_DATA) {
      const predictions = await model.detect(video);
      const count = drawDetections(predictions);
      updateCountUI(count);
    }
  } catch (err) {
    console.error("Erro na detecção:", err);
  }

  setTimeout(detectLoop, DETECTION_INTERVAL_MS);
}

async function init() {
  try {
    await loadModel();
    await startCamera();
    running = true;
    setStatus("Detectando pessoas em tempo real...");
    detectLoop();
  } catch (err) {
    console.error(err);
  }
}

switchBtn.addEventListener("click", async () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  setStatus("Trocando câmera...");
  try {
    await startCamera();
    setStatus("Detectando pessoas em tempo real...");
  } catch (err) {
    // mensagem de erro já definida em startCamera
  }
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
  init();
});
