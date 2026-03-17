const sessionId = `ui-${Math.random().toString(36).slice(2, 8)}`;
const sessionEl = document.getElementById("session-id");
const chatForm = document.getElementById("chat-form");
const chatLog = document.getElementById("chat-log");
const chatEmpty = document.getElementById("chat-empty");
const chatInput = document.getElementById("chat-input");
const chatStatus = document.getElementById("chat-status");
const sendButton = document.getElementById("send-button");
const mapDetail = document.getElementById("map-detail");
const refreshMapBtn = document.getElementById("refresh-map");

sessionEl.textContent = sessionId;

const map = L.map("map").setView([8.9833, -79.5167], 12);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "? OpenStreetMap",
}).addTo(map);

let markers = [];
let isSending = false;
let typingMessage = null;

function clearMarkers() {
  markers.forEach((marker) => map.removeLayer(marker));
  markers = [];
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatPriorityLabel(priority) {
  if (priority === "high") return "Alta";
  if (priority === "medium") return "Media";
  if (priority === "low") return "Baja";
  return "Media";
}

function createClusterIcon(cluster) {
  const count = Number(cluster.count || 0);
  const isGroup = count > 1;
  const className = [
    "cluster-pin",
    `cluster-pin--${cluster.priority || "medium"}`,
    isGroup ? "cluster-pin--group" : "cluster-pin--single",
  ].join(" ");
  const label = isGroup ? `<span class="cluster-pin__label">${count}</span>` : '<span class="cluster-pin__dot"></span>';
  return L.divIcon({
    className: "cluster-marker",
    html: `<div class="${className}">${label}</div>`,
    iconSize: isGroup ? [44, 44] : [28, 28],
    iconAnchor: isGroup ? [22, 22] : [14, 14],
    popupAnchor: [0, -18],
  });
}

function buildClusterPopup(cluster) {
  const ids = (cluster.report_ids || []).map((id) => `#${id}`).join(", ");
  const area = cluster.location_text || "Sin referencia";
  const countLabel = Number(cluster.count || 0) === 1 ? "1 reporte" : `${cluster.count} reportes`;
  return `
    <div class="cluster-popup">
      <strong>${escapeHtml(cluster.report_type)}</strong><br />
      ${escapeHtml(countLabel)}<br />
      Prioridad: ${escapeHtml(formatPriorityLabel(cluster.priority))}<br />
      Area: ${escapeHtml(area)}<br />
      IDs: ${escapeHtml(ids)}
    </div>
  `;
}

function renderMapDetail(cluster) {
  const ids = (cluster.report_ids || []).map((id) => `#${id}`).join(", ");
  const area = cluster.location_text || "Sin referencia";
  const reportLabel = Number(cluster.count || 0) === 1 ? "reporte" : "reportes";
  const selectedLabel = Number(cluster.count || 0) === 1 ? "Reporte seleccionado" : "Cluster seleccionado";
  mapDetail.innerHTML = `
    <div class="map-detail__eyebrow">${selectedLabel}</div>
    <div class="map-detail__title">${escapeHtml(cluster.report_type)}</div>
    <div class="map-detail__grid">
      <div class="map-detail__item">
        <span class="map-detail__label">Prioridad</span>
        <span class="map-badge map-badge--${escapeHtml(cluster.priority || "medium")}">${escapeHtml(formatPriorityLabel(cluster.priority))}</span>
      </div>
      <div class="map-detail__item">
        <span class="map-detail__label">Cantidad</span>
        <span class="map-detail__value">${escapeHtml(`${cluster.count} ${reportLabel}`)}</span>
      </div>
      <div class="map-detail__item map-detail__item--wide">
        <span class="map-detail__label">Area</span>
        <span class="map-detail__value">${escapeHtml(area)}</span>
      </div>
      <div class="map-detail__item map-detail__item--wide">
        <span class="map-detail__label">IDs</span>
        <span class="map-detail__value">${escapeHtml(ids)}</span>
      </div>
    </div>
  `;
}

async function loadMap() {
  const res = await fetch("/reports/clusters?scope=all");
  if (!res.ok) {
    mapDetail.textContent = "No se pudo cargar el mapa.";
    return;
  }

  const data = await res.json();
  clearMarkers();

  if (!data.length) {
    mapDetail.textContent = "No hay clusters para mostrar.";
    return;
  }

  const bounds = [];
  data.forEach((cluster) => {
    const marker = L.marker([cluster.lat, cluster.lon], { icon: createClusterIcon(cluster) }).addTo(map);
    marker.bindPopup(buildClusterPopup(cluster));
    marker.on("click", () => renderMapDetail(cluster));
    markers.push(marker);
    bounds.push([cluster.lat, cluster.lon]);
  });

  if (bounds.length === 1) {
    map.setView(bounds[0], 14);
  } else {
    map.fitBounds(bounds, { padding: [28, 28], maxZoom: 14 });
  }

  renderMapDetail(data[0]);
}

function formatTimestamp() {
  return new Date().toLocaleTimeString("es-PA", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function removeEmptyState() {
  if (chatEmpty && chatEmpty.parentElement) {
    chatEmpty.remove();
  }
}

function scrollChatToBottom() {
  chatLog.scrollTo({
    top: chatLog.scrollHeight,
    behavior: "smooth",
  });
}

function buildMessageContent(text) {
  const content = document.createElement("div");
  content.className = "chat-bubble__content";

  const blocks = String(text || "Sin respuesta")
    .trim()
    .split(/\n{2,}/)
    .filter(Boolean);

  if (!blocks.length) {
    const paragraph = document.createElement("p");
    paragraph.textContent = "Sin respuesta";
    content.appendChild(paragraph);
    return content;
  }

  blocks.forEach((block) => {
    const lines = block
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length && lines.every((line) => /^\d+\./.test(line))) {
      const list = document.createElement("ol");
      list.className = "chat-list";
      lines.forEach((line) => {
        const item = document.createElement("li");
        item.textContent = line.replace(/^\d+\.\s*/, "");
        list.appendChild(item);
      });
      content.appendChild(list);
      return;
    }

    const paragraph = document.createElement("p");
    paragraph.textContent = lines.join("\n");
    content.appendChild(paragraph);
  });

  return content;
}

function createTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "typing-indicator";
  for (let index = 0; index < 3; index += 1) {
    const dot = document.createElement("span");
    dot.style.animationDelay = `${index * 0.14}s`;
    wrapper.appendChild(dot);
  }
  return wrapper;
}

function appendBubble(text, type, options = {}) {
  removeEmptyState();

  const message = document.createElement("article");
  message.className = `chat-message chat-message--${type}`;
  if (options.pending) {
    message.classList.add("chat-message--pending");
  }

  const avatar = document.createElement("div");
  avatar.className = `chat-avatar chat-avatar--${type}`;
  avatar.textContent = type === "user" ? "TU" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";

  const meta = document.createElement("div");
  meta.className = "chat-meta";

  const sender = document.createElement("span");
  sender.textContent = type === "user" ? "Tu" : "Agente";
  meta.appendChild(sender);

  if (!options.pending) {
    const time = document.createElement("span");
    time.textContent = formatTimestamp();
    meta.appendChild(time);
  }

  bubble.appendChild(meta);
  bubble.appendChild(options.pending ? createTypingIndicator() : buildMessageContent(text));
  message.append(avatar, bubble);

  chatLog.appendChild(message);
  scrollChatToBottom();
  return message;
}

function removeTypingBubble() {
  if (typingMessage) {
    typingMessage.remove();
    typingMessage = null;
  }
}

function setComposerState(nextState) {
  isSending = nextState;
  chatInput.disabled = nextState;
  sendButton.disabled = nextState || !chatInput.value.trim();
  chatForm.dataset.state = nextState ? "loading" : "idle";
  chatLog.setAttribute("aria-busy", String(nextState));
  chatStatus.textContent = nextState ? "El agente esta escribiendo..." : "Listo para enviar";
}

function syncSendState() {
  sendButton.disabled = isSending || !chatInput.value.trim();
}

refreshMapBtn.addEventListener("click", loadMap);
chatInput.addEventListener("input", syncSendState);
loadMap();
syncSendState();

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message || isSending) return;

  appendBubble(message, "user");
  chatForm.reset();
  setComposerState(true);

  typingMessage = appendBubble("", "bot", { pending: true });

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });

    removeTypingBubble();

    if (res.ok) {
      const data = await res.json();
      appendBubble(data.response || "Sin respuesta", "bot");
      loadMap();
    } else {
      const text = await res.text();
      appendBubble(`Ocurrio un error.\n\n${text}`, "bot");
    }
  } catch (error) {
    removeTypingBubble();
    appendBubble("No pude conectar con el servidor. Intenta nuevamente.", "bot");
  } finally {
    setComposerState(false);
    syncSendState();
    chatInput.focus();
  }
});
