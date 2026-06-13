// WS + controls + render panels.
const $ = id => document.getElementById(id);
const api = (path, body) => fetch(path, { method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}) });

const mapView = new MapView($("map"), (x, y) => {
  // Clicking the map sets a destination and switches to Auto so the robot drives there.
  mapView.goal = [x, y];                 // instant visual feedback at the click
  if (mode !== "auto") setMode("auto");
  api("/api/nav/goal", { x, y }).then(r => r.json()).then(j => {
    $("nav-status").textContent = j.ok ? "NAVIGATING" : "NO PATH";
    if (!j.ok) mapView.goal = null;      // unreachable -> clear marker
  }).catch(() => {});
});
let mode = "manual";
let lastMsg = {};

// ---- WebSocket ----
const ws = new WebSocket(`ws://${location.host}/ws/stream`);
ws.binaryType = "blob";
const urls = { cam: null, depth: null };
ws.onmessage = async ev => {
  if (typeof ev.data === "string") { onTelemetry(JSON.parse(ev.data)); return; }
  const ch = new Uint8Array(await ev.data.slice(0, 1).arrayBuffer())[0];
  const payload = ev.data.slice(1);
  if (ch === 1) setImg("cam", payload);
  else if (ch === 2) setImg("depth", payload);
  else if (ch === 3) mapView.setMap(payload, lastMsg.map_meta);
};
function setImg(id, blob) {
  const old = urls[id];
  urls[id] = URL.createObjectURL(blob);
  $(id).src = urls[id];
  if (old) URL.revokeObjectURL(old);
}

function onTelemetry(m) {
  lastMsg = m;
  $("dot-cam").className = "dot ok"; $("dot-depth").className = "dot ok";
  $("dot-lidar").className = m.lidar_ranges?.length ? "dot ok" : "dot";
  $("dot-imu").className = m.imu ? "dot ok" : "dot";
  if (m.imu) {
    const d = r => (r * 180 / Math.PI).toFixed(1);
    $("roll").textContent = d(m.imu.rpy[0]);
    $("pitch").textContent = d(m.imu.rpy[1]);
    $("yaw").textContent = d(m.imu.rpy[2]);
    ["ax","ay","az"].forEach((id, i) => $(id).textContent = m.imu.accel[i].toFixed(2));
    ["gx","gy","gz"].forEach((id, i) => $(id).textContent = m.imu.gyro[i].toFixed(3));
  }
  $("slam-state").textContent = m.slam_state;
  $("nav-status").textContent = m.nav_status;
  $("tg-status").textContent = m.telegram;
  $("tg-status").style.color = m.telegram === "ARMED" ? "#2c5" : "#c80";
  $("robot-state").textContent = m.fallen ? "FALLEN" : "OK";
  $("robot-state").style.color = m.fallen ? "#c33" : "#2c5";
  // Set the map transform from telemetry directly — a map binary can arrive
  // before the first telemetry text, leaving overlays (path/goal/robot) without
  // a transform; this guarantees they render as soon as telemetry is known.
  if (m.map_meta) mapView.meta = m.map_meta;
  mapView.pose = m.pose; mapView.path = m.path || [];
  if (m.goal) mapView.goal = m.goal;            // keep local click goal until backend confirms
  else if (m.nav_status && m.nav_status.startsWith("IDLE")) mapView.goal = null;
  drawLidar(m.lidar_ranges, m.lidar_angles);
  $("det-list").innerHTML = (m.detections || []).slice(-8).reverse()
    .map(d => `<li>⚠ <b>${d.cls}</b> ${d.conf} @ (${d.x}, ${d.y})</li>`).join("");
}

// ---- Lidar polar ----
function drawLidar(ranges, angles) {
  const cv = $("lidar"), ctx = cv.getContext("2d");
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, cv.width, cv.height);
  const cx = cv.width / 2, cy = cv.height / 2, sc = cv.width / 2 / 8.0;
  ctx.strokeStyle = "#dde";
  for (let r = 2; r <= 8; r += 2) {
    ctx.beginPath(); ctx.arc(cx, cy, r * sc, 0, 7); ctx.stroke();
  }
  ctx.strokeStyle = "#3a3"; ctx.beginPath();
  ctx.moveTo(cx, cy); ctx.lineTo(cx, cy - 8 * sc); ctx.stroke();
  if (!ranges) return;
  ctx.fillStyle = "#36c";
  for (let i = 0; i < ranges.length; i++) {
    if (ranges[i] >= 7.99) continue;
    const a = angles[i] + Math.PI / 2;   // angle 0 = robot's nose = pointing up
    ctx.fillRect(cx + Math.cos(a) * ranges[i] * sc - 1,
                 cy - Math.sin(a) * ranges[i] * sc - 1, 2, 2);
  }
  ctx.fillStyle = "#e22"; ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 7); ctx.fill();
}

// ---- Drive mode + teleop ----
$("btn-manual").onclick = () => setMode("manual");
$("btn-auto").onclick = () => setMode("auto");
function setMode(m) {
  mode = m;
  $("btn-manual").classList.toggle("active", m === "manual");
  $("btn-auto").classList.toggle("active", m === "auto");
  api("/api/mode", { mode: m });
}
const keys = {};
const DRIVE_KEYS = ["w", "a", "s", "d", "q", "e"];
onkeydown = e => {
  const k = e.key.toLowerCase();
  if (DRIVE_KEYS.includes(k)) e.preventDefault();  // don't scroll the page
  keys[k] = true;
};
onkeyup = e => { keys[e.key.toLowerCase()] = false; };

// On-screen D-pad: holding a button contributes to the teleop command and
// forces Manual mode, so movement works without keyboard focus.
const btnCmd = { vx: 0, vy: 0, wz: 0 };
document.querySelectorAll(".drive").forEach(b => {
  const press = e => {
    e.preventDefault();
    if (mode !== "manual") setMode("manual");
    btnCmd.vx = +b.dataset.vx; btnCmd.vy = +b.dataset.vy; btnCmd.wz = +b.dataset.wz;
    b.classList.add("pressed");
  };
  const release = () => {
    btnCmd.vx = btnCmd.vy = btnCmd.wz = 0;
    b.classList.remove("pressed");
  };
  b.addEventListener("mousedown", press);
  b.addEventListener("touchstart", press, { passive: false });
  b.addEventListener("mouseup", release);
  b.addEventListener("mouseleave", release);
  b.addEventListener("touchend", release);
});

setInterval(() => {
  if (mode !== "manual") return;
  const vx = (keys.w ? 1 : 0) + (keys.s ? -1 : 0) + btnCmd.vx;
  const vy = (keys.q ? 1 : 0) + (keys.e ? -1 : 0) + btnCmd.vy;
  const wz = (keys.a ? 1 : 0) + (keys.d ? -1 : 0) + btnCmd.wz;
  api("/api/teleop", { vx, vy, wz });
}, 100);

// ---- Buttons ----
$("btn-reset-robot").onclick = () => api("/api/robot/reset");
$("btn-nav-stop").onclick = () => api("/api/nav/stop");
$("btn-nav-patrol").onclick = () => api("/api/nav/patrol");
$("btn-slam-start").onclick = () => api("/api/slam/start");
$("btn-slam-pause").onclick = () => api("/api/slam/pause");
$("btn-slam-end").onclick = () => api("/api/slam/end");
$("btn-slam-reset").onclick = () => api("/api/slam/reset");
$("btn-slam-save").onclick = async () => {
  const name = prompt("Map name:", "map1");
  if (name) { await api("/api/slam/save", { name }); loadMapList(); }
};
$("btn-slam-load").onclick = () => {
  const name = $("map-select").value;
  if (name) api("/api/slam/load", { name });
};
$("btn-slam-delete").onclick = async () => {
  const name = $("map-select").value;
  if (name && confirm(`Delete saved map "${name}"?`)) {
    await api("/api/slam/delete", { name });
    loadMapList();
  }
};
async function loadMapList() {
  const r = await fetch("/api/slam/maps").then(r => r.json());
  // Placeholder first so no saved map is auto-selected; the live SLAM is shown by default.
  $("map-select").innerHTML = '<option value="">— live SLAM —</option>'
    + r.maps.map(m => `<option>${m}</option>`).join("");
}
loadMapList();

// ---- render loop ----
(function loop() { mapView.draw(); requestAnimationFrame(loop); })();
