// Lander Studio front end: live view, training, evaluation, unit tests and recorded experiments.
"use strict";

const COL = { black: "#111111", blue: "#1550c8", red: "#d6312b", green: "#22a05a", orange: "#f08c00",
  purple: "#7b3fe4", teal: "#0e9aa7", pink: "#e0457b", olive: "#8a7a00", hair: "#e3ebf8" };
const PALETTE = [COL.blue, COL.red, COL.green, COL.orange, COL.purple, COL.teal, COL.pink, COL.olive];
const REWARD_COLORS = { baseline: COL.blue, time_penalty: COL.orange, upright_slow_bonus: COL.red,
  upright_slow_potential: COL.green, upright_slow_fixed: COL.purple };
const OUT_COLORS = { landed: COL.blue, crashed: COL.red, out_of_bounds: COL.pink, timeout: COL.orange };
const ACTIONS = ["do nothing", "left engine", "main engine", "right engine"];
const ACTION_COLORS = [COL.blue, COL.green, COL.red, COL.purple];
const OBS = ["x", "height", "x speed", "y speed", "tilt", "spin", "left leg", "right leg"];

Chart.defaults.color = COL.black;
Chart.defaults.borderColor = COL.hair;
Chart.defaults.font.family = '"Segoe UI", system-ui, sans-serif';
Chart.defaults.animation = false;
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.plugins.legend.labels.boxWidth = 14;

const $ = (id) => document.getElementById(id);
const fmtK = (v) => (Math.abs(v) >= 1000 ? `${Math.round(v / 1000)}k` : `${v}`);
const pct = (x) => `${Math.round(100 * x)}%`;
const mean = (a) => a.reduce((s, x) => s + x, 0) / Math.max(a.length, 1);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const std = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((x) => (x - m) ** 2))); };

let SPEC = [], DEFAULTS = {}, PRESETS = {}, MODELS = [];
let layout = null, lastFrame = null, initialHist = null, evalTotal = 0;
const charts = {};

// ---------------------------------------------------------------- API
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
function say(id, text, bad = false) { const el = $(id); el.textContent = text; el.className = `msg ${bad ? "red" : "blue"}`; }
function setStatus(text, bad = false) { const s = $("status"); s.textContent = text; s.className = bad ? "bad" : ""; }

// ---------------------------------------------------------------- charts
const xLinear = (title = "training steps") => ({ type: "linear", title: { display: true, text: title },
  ticks: { callback: fmtK, maxTicksLimit: 9 }, grid: { color: COL.hair } });
function line(label, color, extra = {}) {
  return { label, data: [], borderColor: color, backgroundColor: color, pointRadius: 0, borderWidth: 2, tension: 0.15, ...extra };
}
function makeCharts() {
  charts.ret = new Chart($("cRet"), { type: "line", data: { datasets: [
    line("real score (plain)", COL.blue), line("what PPO is paid (shaped)", COL.red),
    line("solved = 200", COL.black, { borderDash: [6, 5], borderWidth: 1 })] },
    options: { scales: { x: xLinear(), y: { title: { display: true, text: "return per episode" } } } } });
  charts.len = new Chart($("cLen"), { type: "line", data: { datasets: [
    line("frames per episode", COL.blue), line("% cut at 1,000 frames", COL.red, { yAxisID: "y2" })] },
    options: { scales: { x: xLinear(), y: { title: { display: true, text: "frames" }, min: 0, max: 1000 },
      y2: { position: "right", min: 0, max: 100, grid: { display: false }, title: { display: true, text: "% time-outs" } } } } });
  charts.loss = new Chart($("cLoss"), { type: "line", data: { datasets: [
    line("policy loss", COL.blue), line("value loss", COL.red, { yAxisID: "y2" }), line("entropy", COL.green, { yAxisID: "y3" })] },
    options: { scales: { x: xLinear(), y: { title: { display: true, text: "policy loss" } },
      y2: { position: "right", grid: { display: false }, title: { display: true, text: "value loss" } },
      y3: { position: "right", grid: { display: false }, title: { display: true, text: "entropy" } } } } });
  charts.stab = new Chart($("cStab"), { type: "line", data: { datasets: [
    line("approx. KL", COL.purple), line("clip fraction", COL.orange), line("explained variance", COL.teal, { yAxisID: "y2" })] },
    options: { scales: { x: xLinear(), y: { min: 0, title: { display: true, text: "KL, clip fraction" } },
      y2: { position: "right", min: -1, max: 1, grid: { display: false }, title: { display: true, text: "explained var." } } } } });
  charts.norms = new Chart($("cNorms"), { type: "line", data: { datasets: [] },
    options: { scales: { x: xLinear(), y: { title: { display: true, text: "L2 norm" } } } } });
  charts.change = new Chart($("cChange"), { type: "bar", data: { datasets: [{ label: "weight change", data: [], backgroundColor: COL.blue }] },
    options: { plugins: { legend: { display: false } }, scales: { x: xLinear(), y: { title: { display: true, text: "L2 distance" } } } } });
  const bins = Array.from({ length: 40 }, (_, i) => (-1.2 + (i + 0.5) * 0.06).toFixed(2));
  charts.hist = new Chart($("cHist"), { data: { labels: bins, datasets: [
    { type: "bar", label: "now", data: [], backgroundColor: COL.blue },
    { type: "line", label: "at the start", data: [], borderColor: COL.red, pointRadius: 0, borderWidth: 2 }] },
    options: { scales: { x: { title: { display: true, text: "weight value" }, ticks: { maxTicksLimit: 9 } },
      y: { title: { display: true, text: "count" } } } } });
  charts.eval = new Chart($("cEval"), { data: { labels: [], datasets: [
    { type: "bar", label: "episode score", data: [], backgroundColor: [] },
    { type: "line", label: "running mean", data: [], borderColor: COL.black, pointRadius: 0, borderWidth: 2 },
    { type: "line", label: "solved", data: [], borderColor: COL.black, borderDash: [6, 5], borderWidth: 1, pointRadius: 0 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "episode" } },
      y: { title: { display: true, text: "plain score" } } } } });
}
function push(chart, i, x, y) { if (y !== undefined && y !== null) chart.data.datasets[i].data.push({ x, y }); }
function clearCharts() {
  for (const k of ["ret", "len", "loss", "stab", "change"]) charts[k].data.datasets.forEach((d) => (d.data = []));
  charts.norms.data.datasets = [];
  charts.hist.data.datasets.forEach((d) => (d.data = []));
  Object.values(charts).forEach((c) => c.update("none"));
  initialHist = null;
}

// ---------------------------------------------------------------- parameter form
const FORM_GROUPS = ["Reward", "PPO", "Network", "Training"];
const formSpec = () => SPEC.filter((p) => FORM_GROUPS.includes(p.group));
function inputFor(p) {
  if (p.kind === "choice") return `<select data-key="${p.key}">${p.choices.map((c) => `<option>${c}</option>`).join("")}</select>`;
  if (p.kind === "bool") return `<input type="checkbox" data-key="${p.key}">`;
  if (p.kind === "layers") return `<input type="text" data-key="${p.key}" placeholder="e.g. 64, 64">`;
  return `<input type="number" data-key="${p.key}" min="${p.min}" max="${p.max}" step="${p.step}">`;
}
const shown = (p, v) => (p.kind === "layers" ? v.join(", ") : p.kind === "bool" ? (v ? "on" : "off") : String(v));
function buildForm() {
  $("params").innerHTML = FORM_GROUPS.map((g) => `<div><h3>${g}</h3>${SPEC.filter((p) => p.group === g).map((p) => `
    <div class="param" data-param="${p.key}">
      <div class="lab">${p.label}<span class="def">default ${shown(p, p.default)}</span><span class="changed"></span></div>
      ${inputFor(p)}<div class="why">${p.why}</div></div>`).join("")}</div>`).join("");
  $("params").querySelectorAll("[data-key]").forEach((el) => el.addEventListener("input", refreshForm));
}
const field = (key) => $("params").querySelector(`[data-key="${key}"]`);
function getForm() {
  const out = {};
  for (const p of formSpec()) {
    const el = field(p.key);
    out[p.key] = p.kind === "bool" ? el.checked : p.kind === "choice" || p.kind === "layers" ? el.value : Number(el.value);
  }
  return out;
}
function setForm(values) {
  for (const p of formSpec()) {
    const el = field(p.key), v = values[p.key] ?? p.default;
    if (p.kind === "bool") el.checked = v; else el.value = p.kind === "layers" ? [].concat(v).join(", ") : v;
  }
  refreshForm();
}
function refreshForm() {
  const reward = field("reward").value;
  for (const p of formSpec()) {
    const box = $("params").querySelector(`[data-param="${p.key}"]`), el = field(p.key);
    box.style.display = p.applies && !p.applies.includes(reward) ? "none" : "";
    const cur = p.kind === "bool" ? el.checked : p.kind === "layers" ? el.value.replace(/\s/g, "")
      : p.kind === "choice" ? el.value : Number(el.value);
    const def = p.kind === "layers" ? p.default.join(",") : p.default;
    box.querySelector(".changed").textContent = cur === def ? "" : "changed";
  }
  const f = getForm(), batch = f.n_steps * f.n_envs;
  $("trainInfo").textContent = `${batch.toLocaleString()} transitions per update · ${Math.ceil(f.total_timesteps / batch)} updates · ` +
    `actor 8 → ${f.actor_layers} → 4 · critic 8 → ${f.critic_layers} → 1 · ${f.activation}`;
}

// ---------------------------------------------------------------- network drawing
const net = $("net"), nctx = net.getContext("2d");
function mix(c, t) { // white -> colour c, by t in [0, 1]
  const n = parseInt(c.slice(1), 16), r = n >> 16, g = (n >> 8) & 255, b = n & 255;
  return `rgb(${Math.round(255 + (r - 255) * t)},${Math.round(255 + (g - 255) * t)},${Math.round(255 + (b - 255) * t)})`;
}
const diverge = (v) => (v >= 0 ? mix(COL.red, Math.min(1, v)) : mix(COL.blue, Math.min(1, -v)));
function drawNet(frame) {
  if (!layout) return;
  const W = net.width, H = net.height, L = layout.sizes.length, left = 210, right = W - 250;
  const xs = layout.sizes.map((_, j) => left + (j * (right - left)) / (L - 1));
  const ys = layout.idx.map((ix, j) => { const m = ix.length, top = j === L - 1 ? 250 : 80, bot = j === L - 1 ? 590 : H - 60;
    return ix.map((_, k) => (m === 1 ? (top + bot) / 2 : top + (k * (bot - top)) / (m - 1))); });
  const vals = [
    (frame?.obs || Array(8).fill(0)).map((v) => Math.max(-1, Math.min(1, v / 1.5))),
    ...layout.idx.slice(1, -1).map((ix, j) => frame?.act?.[j] || Array(ix.length).fill(0)),
    frame?.probs || Array(4).fill(0.25)];
  nctx.clearRect(0, 0, W, H);
  for (let j = 0; j < L - 1; j++) { // edges: the strongest signals (weight x source activation)
    const Wm = layout.w[j], src = vals[j], list = [];
    for (let a = 0; a < Wm.length; a++) for (let b = 0; b < Wm[a].length; b++) {
      const s = frame ? Wm[a][b] * src[b] : Wm[a][b]; list.push([Math.abs(s), s, b, a]); }
    list.sort((p, q) => q[0] - p[0]);
    const keep = list.slice(0, j === 0 ? 120 : 220), max = keep[0]?.[0] || 1;
    for (const [m, s, b, a] of keep) {
      const t = m / max;
      nctx.strokeStyle = s >= 0 ? `rgba(214,49,43,${0.08 + 0.7 * t})` : `rgba(21,80,200,${0.08 + 0.7 * t})`;
      nctx.lineWidth = 0.6 + 2.6 * t;
      nctx.beginPath(); nctx.moveTo(xs[j], ys[j][b]); nctx.lineTo(xs[j + 1], ys[j + 1][a]); nctx.stroke();
    }
  }
  nctx.textAlign = "center";
  layout.sizes.forEach((n, j) => {
    nctx.font = "600 28px Segoe UI, system-ui"; nctx.fillStyle = COL.black;
    nctx.fillText(j === 0 ? "8 inputs" : j === L - 1 ? "4 actions" : `hidden ${j} · ${n}`, xs[j], 44);
    if (layout.idx[j].length < n) { nctx.fillStyle = COL.blue; nctx.font = "24px Segoe UI, system-ui";
      nctx.fillText(`showing ${layout.idx[j].length} of ${n}`, xs[j], H - 18); }
  });
  for (let j = 0; j < L; j++) {
    const out = j === L - 1, rad = j === 0 ? 15 : out ? 19 : Math.max(5, Math.min(11, 300 / layout.idx[j].length));
    ys[j].forEach((y, k) => {
      nctx.fillStyle = out ? mix(ACTION_COLORS[k], vals[j][k]) : diverge(vals[j][k]);
      nctx.strokeStyle = COL.black; nctx.lineWidth = out && frame && frame.action === k ? 4 : 1.2;
      nctx.beginPath(); nctx.arc(xs[j], y, rad, 0, 7); nctx.fill(); nctx.stroke();
    });
  }
  nctx.font = "26px Segoe UI, system-ui"; nctx.textAlign = "right";
  ys[0].forEach((y, k) => { nctx.fillStyle = COL.black; nctx.fillText(OBS[k], xs[0] - 26, y - 2);
    nctx.fillStyle = COL.blue; nctx.fillText(frame?.obs ? frame.obs[k].toFixed(2) : "", xs[0] - 26, y + 26); });
  nctx.textAlign = "left";
  ys[L - 1].forEach((y, k) => { const on = frame && frame.action === k;
    nctx.font = `${on ? "700" : "400"} 27px Segoe UI, system-ui`; nctx.fillStyle = on ? COL.red : COL.black;
    nctx.fillText(ACTIONS[k], xs[L - 1] + 30, y - 2);
    nctx.fillStyle = COL.blue; nctx.fillText(frame ? `${Math.round(100 * frame.probs[k])}%` : "", xs[L - 1] + 30, y + 28); });
  if (frame && frame.value !== undefined) { nctx.font = "600 28px Segoe UI, system-ui"; nctx.fillStyle = COL.blue;
    nctx.fillText(`critic V(s) = ${frame.value.toFixed(1)}`, xs[L - 1] - 60, 680); }
}

// ---------------------------------------------------------------- live frames
$("probs").innerHTML = ACTIONS.map((a, i) => `<div class="prob" id="pb${i}"><span>${a}</span><i style="--c:${ACTION_COLORS[i]}"></i><b></b></div>`).join("");
function onFrame(f) {
  const src = "data:image/jpeg;base64," + f.img;
  $("liveImg").src = src;
  if (f.src === "eval") $("evalImg").src = src;
  $("tmModel").textContent = f.label;
  $("tmFrame").textContent = `${f.episode} · ${f.t}`;
  $("tmRet").textContent = f.ret.toFixed(1);
  if (f.outcome) {
    $("tmLast").innerHTML = `<span class="${f.outcome === "landed" ? "blue" : "red"}">${f.outcome.replace(/_/g, " ")}</span>`;
    return;
  }
  $("tmValue").textContent = f.value.toFixed(1);
  f.probs.forEach((p, i) => { const row = $(`pb${i}`); row.classList.toggle("on", i === f.action);
    row.querySelector("i").style.width = `${Math.round(100 * p)}%`; row.querySelector("b").textContent = `${Math.round(100 * p)}%`; });
  lastFrame = f;
  drawNet(f);
}
function onPilot(p) {
  layout = p.layout;
  if (p.training) $("pilotSel").value = "training";
  else if (p.path && [...$("pilotSel").options].some((o) => o.value === p.path)) $("pilotSel").value = p.path;
  $("tmModel").textContent = p.label;
  const sampled = layout.sizes.some((n, j) => layout.idx[j].length < n);
  $("netNote").innerHTML = `Actor network: <b>${layout.sizes.join(" → ")}</b>. ` +
    (sampled ? "Wide layers are sampled evenly for drawing. " : "") +
    "Lines show the strongest signals at this frame (weight × activation).";
  if (!lastFrame) drawNet(null);
}

// ---------------------------------------------------------------- training events
function onProgress(p) {
  push(charts.ret, 0, p.t, p.ep_rew_mean); push(charts.ret, 1, p.t, p.shaped_rew_mean);
  charts.ret.data.datasets[2].data = [{ x: 0, y: 200 }, { x: p.total, y: 200 }];
  push(charts.len, 0, p.t, p.ep_len_mean); push(charts.len, 1, p.t, p.trunc_rate === undefined ? undefined : 100 * p.trunc_rate);
  charts.ret.update("none"); charts.len.update("none");
  $("trainBar").style.width = `${Math.min(100, (100 * p.t) / p.total)}%`;
  const eta = p.fps ? Math.max(0, (p.total - p.t) / p.fps) : 0;
  setStatus(`training ${Math.round((100 * p.t) / p.total)}% · ${fmtK(p.t)} / ${fmtK(p.total)} steps · ` +
    `${p.fps.toLocaleString()} steps/s · ~${Math.ceil(eta / 60)} min left`);
}
function onLosses(l) {
  push(charts.loss, 0, l.t, l.policy_loss); push(charts.loss, 1, l.t, l.value_loss); push(charts.loss, 2, l.t, l.entropy);
  push(charts.stab, 0, l.t, l.approx_kl); push(charts.stab, 1, l.t, l.clip_fraction); push(charts.stab, 2, l.t, l.explained_variance);
  charts.loss.update("none"); charts.stab.update("none");
}
function onWeights(w) {
  Object.entries(w.norms).forEach(([name, v]) => {
    let ds = charts.norms.data.datasets.find((d) => d.label === name);
    if (!ds) { ds = line(name, PALETTE[charts.norms.data.datasets.length % PALETTE.length]); charts.norms.data.datasets.push(ds); }
    ds.data.push({ x: w.t, y: v });
  });
  push(charts.change, 0, w.t, w.change);
  charts.norms.update("none"); charts.change.update("none");
}
function onWeightsDetail(d) {
  if (!initialHist) initialHist = d.hist;
  charts.hist.data.datasets[0].data = d.hist; charts.hist.data.datasets[1].data = initialHist;
  charts.hist.update("none");
  drawHeat(d.w1);
}
function drawHeat(w1) {
  const c = $("heat"), x = c.getContext("2d"), rows = w1[0].length, cols = w1.length;
  const mx = Math.max(...w1.flat().map(Math.abs)) || 1, left = 120, cw = (c.width - left) / cols, ch = (c.height - 10) / rows;
  x.clearRect(0, 0, c.width, c.height); x.font = "20px Segoe UI, system-ui"; x.textAlign = "right"; x.fillStyle = COL.black;
  OBS.forEach((n, i) => x.fillText(n, left - 10, 5 + i * ch + ch / 2 + 7));
  w1.forEach((neuron, j) => neuron.forEach((v, i) => { x.fillStyle = diverge(v / mx); x.fillRect(left + j * cw, 5 + i * ch, cw - 1, ch - 1); }));
}
function trainButtons(running) { $("startBtn").disabled = running; $("stopBtn").disabled = !running; }

// ---------------------------------------------------------------- evaluation
let evalRows = [];
function evalBig(summary) {
  const rets = evalRows.map((e) => e.ret), n = rets.length;
  const share = (o) => evalRows.filter((e) => e.outcome === o).length / Math.max(n, 1);
  const m = summary ? summary.mean_return : mean(rets), s = summary ? summary.std_return : std(rets);
  const succ = summary ? summary.success_rate : rets.filter((x) => x >= 200).length / Math.max(n, 1);
  $("evalBig").innerHTML = `
    <div><span>episodes</span><b>${n}${evalTotal ? " / " + evalTotal : ""}</b></div>
    <div><span>mean score ± std</span><b class="${m >= 200 ? "blue" : "red"}">${n ? `${m.toFixed(1)} ± ${s.toFixed(1)}` : "–"}</b></div>
    <div><span>success (≥ 200)</span><b class="blue">${n ? pct(succ) : "–"}</b></div>
    <div><span>landed</span><b class="blue">${n ? pct(share("landed")) : "–"}</b></div>
    <div><span>crashed / flew away</span><b class="red">${n ? pct(share("crashed") + share("out_of_bounds")) : "–"}</b></div>
    <div><span>cut at 1,000 frames</span><b class="red">${n ? pct(share("timeout")) : "–"}</b></div>`;
}
function onEvalStart(s) {
  evalRows = []; evalTotal = s.n; const c = charts.eval;
  c.data.labels = []; c.data.datasets.forEach((d) => { d.data = []; }); c.data.datasets[0].backgroundColor = []; c.update("none");
  $("evalTable").innerHTML = `<tr><th>#</th><th>seed</th><th class="num">score</th><th class="num">frames</th><th>ending</th></tr>`;
  $("evalBar").style.width = "0"; evalBig(null);
  say("evalMsg", `Testing ${s.label} on ${s.n} episodes…`); evalButtons(true);
}
function onEvalEpisode(e) {
  evalRows.push(e); const c = charts.eval, rets = evalRows.map((x) => x.ret);
  c.data.labels.push(e.i); c.data.datasets[0].data.push(e.ret); c.data.datasets[0].backgroundColor.push(OUT_COLORS[e.outcome]);
  c.data.datasets[1].data.push(mean(rets)); c.data.datasets[2].data.push(200); c.update("none");
  $("evalTable").insertAdjacentHTML("beforeend", `<tr><td>${e.i}</td><td>${e.seed}</td><td class="num">${e.ret.toFixed(1)}</td>
    <td class="num">${e.length}</td><td class="${e.outcome === "landed" ? "blue" : "red"}">${e.outcome.replace(/_/g, " ")}</td></tr>`);
  $("evalBar").style.width = `${(100 * evalRows.length) / (evalTotal || 1)}%`;
  evalBig(null);
}
function onEvalDone(s) {
  evalBig(s.summary); evalButtons(false);
  if (s.summary) say("evalMsg", `${s.status === "stopped" ? "Stopped" : "Done"}: ${s.label}, mean ${s.summary.mean_return.toFixed(1)} over ${s.summary.n_episodes} episodes.`);
}
function evalButtons(running) { $("evalBtn").disabled = running; $("evalStop").disabled = !running; }

// ---------------------------------------------------------------- unit tests
let TESTS = [], testStatus = {}, testFailures = {};
const ST_TEXT = { passed: "✓ passed", failed: "✗ failed", error: "✗ error", skipped: "– skipped", running: "● running", queued: "queued" };
function renderTests() {
  $("testsTable").innerHTML = `<tr><th>status</th><th>test</th><th>file</th><th>what it checks</th><th></th></tr>` + TESTS.map((t, k) => {
    const st = testStatus[t.id] || "", fail = testFailures[t.id];
    return `<tr id="tr${k}"><td class="st ${st}">${ST_TEXT[st] || ""}</td><td class="name">${esc(t.name)}</td><td class="file">${esc(t.file)}</td>
      <td>${esc(t.doc)}${fail ? `<pre>${esc(fail)}</pre>` : ""}</td>
      <td><button class="link" data-run="${k}">run</button></td></tr>`; }).join("");
  $("testsTable").querySelectorAll("[data-run]").forEach((b) => b.addEventListener("click", () => runTests([TESTS[+b.dataset.run].id])));
  testsBig();
}
function setTestRow(id) {
  const k = TESTS.findIndex((t) => t.id === id); if (k < 0) return;
  const cell = $(`tr${k}`).querySelector(".st"), st = testStatus[id] || "";
  cell.className = `st ${st}`; cell.textContent = ST_TEXT[st] || "";
}
function testsBig(summary) {
  const st = TESTS.map((t) => testStatus[t.id]), n = (s) => st.filter((x) => x === s).length;
  $("testsBig").innerHTML = `<div><span>tests</span><b>${TESTS.length}</b></div>
    <div><span>passed</span><b class="blue">${n("passed")}</b></div>
    <div><span>failed</span><b class="red">${n("failed") + n("error")}</b></div>
    ${summary ? `<div><span>last run</span><b>${summary.duration.toFixed(1)} s</b></div>` : ""}`;
}
async function runTests(ids) {
  try { await api("/api/tests/run", ids ? { ids } : {}); } catch (e) { say("testsMsg", e.message, true); }
}

// ---------------------------------------------------------------- recorded experiments
async function loadExperiments() {
  try {
    const d = await api("/api/experiments"), rewards = Object.keys(d.results);
    const agg = rewards.map((rw) => { const s = d.results[rw], m = (k) => mean(s.map((x) => x[k]));
      return { rw, s, ret: m("mean_return"), succ: m("success_rate"), len: m("mean_length"), trunc: m("truncation_rate"),
        landed: m("landed_rate"), crashed: m("crashed_rate"), oob: m("out_of_bounds_rate"), timeout: m("timeout_rate") }; });
    new Chart($("xRet"), { data: { labels: rewards, datasets: [
      { type: "bar", label: "mean of 3 seeds", data: agg.map((a) => a.ret), backgroundColor: agg.map((a) => REWARD_COLORS[a.rw] || COL.blue) },
      { type: "scatter", label: "each seed", data: agg.flatMap((a) => a.s.map((x) => ({ x: a.rw, y: x.mean_return }))), backgroundColor: COL.black, pointRadius: 4 },
      { type: "line", label: "solved", data: rewards.map(() => 200), borderColor: COL.black, borderDash: [6, 5], borderWidth: 1, pointRadius: 0 }] },
      options: { plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: "plain score" } } } } });
    new Chart($("xOut"), { type: "bar", data: { labels: rewards, datasets: [["landed", "landed"], ["crashed", "crashed"], ["oob", "out_of_bounds"], ["timeout", "timeout"]]
      .map(([k, o]) => ({ label: o.replace(/_/g, " "), data: agg.map((a) => 100 * a[k]), backgroundColor: OUT_COLORS[o] })) },
      options: { scales: { x: { stacked: true }, y: { stacked: true, max: 100, title: { display: true, text: "% of test episodes" } } } } });
    const g = d.curves.grid;
    new Chart($("xCurve"), { type: "line", data: { datasets: Object.entries(d.curves.rewards).map(([rw, c]) =>
      ({ ...line(rw, REWARD_COLORS[rw] || COL.black), data: c.r.mean.map((y, i) => ({ x: g[i], y })) })) },
      options: { scales: { x: xLinear(), y: { title: { display: true, text: "plain score (training)" } } } } });
    $("xTable").innerHTML = `<tr><th>reward</th><th class="num">mean score</th><th class="num">worst seed</th><th class="num">success</th>
      <th class="num">frames</th><th class="num">time-outs</th><th class="num">landed</th></tr>` + agg.map((a) => `<tr>
      <td style="color:${REWARD_COLORS[a.rw] || COL.black};font-weight:650">${a.rw}</td><td class="num">${a.ret.toFixed(1)}</td>
      <td class="num">${Math.min(...a.s.map((x) => x.mean_return)).toFixed(1)}</td><td class="num">${pct(a.succ)}</td>
      <td class="num">${a.len.toFixed(0)}</td><td class="num ${a.trunc > 0.2 ? "red" : ""}">${pct(a.trunc)}</td><td class="num">${pct(a.landed)}</td></tr>`).join("");
  } catch (e) { $("xTable").innerHTML = `<tr><td class="red">No recorded experiments found in runs/ (${e.message}).</td></tr>`; }
  loadProgression();
}
async function loadProgression() {
  try {
    const p = await api("/api/experiments/progression");
    if (p.computing) { setTimeout(loadProgression, 5000); return; }
    if (charts.prog) charts.prog.destroy();
    charts.prog = new Chart($("xProg"), { type: "line", data: { datasets: Object.entries(p).map(([label, pts]) =>
      ({ ...line(label, REWARD_COLORS[label.split("/")[0]] || COL.black, { pointRadius: 3 }), data: pts.map((q) => ({ x: q.steps, y: q.mean_return })) })) },
      options: { scales: { x: xLinear(), y: { title: { display: true, text: "test score (10 episodes)" } } } } });
  } catch (e) { /* no progression runs recorded */ }
}

// ---------------------------------------------------------------- model lists
function fillModels() {
  const opts = MODELS.map((m) => `<option value="${esc(m.path)}">${esc(m.label)}</option>`).join("");
  const keep = $("pilotSel").value, keepEval = $("evalModel").value;
  $("pilotSel").innerHTML = `<option value="training">the model being trained (live)</option>${opts}`;
  $("evalModel").innerHTML = `<option value="pilot">the model flying above</option>${opts}`;
  if (keep) $("pilotSel").value = keep;
  if (keepEval) $("evalModel").value = keepEval;
}
async function refreshModels() { MODELS = (await api("/api/models", {})).models; fillModels(); }

// ---------------------------------------------------------------- live stream
function connect() {
  const es = new EventSource("/api/events");
  es.onopen = () => setStatus("connected · live");
  es.onerror = () => setStatus("disconnected: is `python -m lander.studio` still running?", true);
  const on = (name, fn) => es.addEventListener(name, (e) => fn(JSON.parse(e.data)));
  on("frame", onFrame);
  on("pilot", onPilot);
  on("view_episode", (e) => { $("tmLast").innerHTML = `<span class="${e.outcome === "landed" ? "blue" : "red"}">${e.outcome.replace(/_/g, " ")}</span> ${e.ret.toFixed(0)}`; });
  on("train_start", (s) => { clearCharts(); trainButtons(true); $("pilotSel").value = "training";
    say("trainMsg", `Training ${s.cfg.reward}: ${s.n_params.toLocaleString()} parameters, saving to ${s.run_dir}.`); });
  on("progress", onProgress);
  on("losses", onLosses);
  on("weights", onWeights);
  on("weights_detail", onWeightsDetail);
  on("train_done", async (s) => { trainButtons(false); setStatus(`training ${s.status} · ${fmtK(s.timesteps)} steps in ${Math.round(s.elapsed)} s`);
    say("trainMsg", `${s.status === "stopped" ? "Stopped" : "Done"}. Saved ${s.model}. Test it in section 3.`);
    await refreshModels(); $("evalModel").value = s.model; });
  on("train_error", (s) => { trainButtons(false); say("trainMsg", `Training failed: ${s.message}`, true); setStatus("training failed", true); });
  on("eval_start", onEvalStart);
  on("eval_episode", onEvalEpisode);
  on("eval_done", onEvalDone);
  on("eval_error", (s) => { evalButtons(false); say("evalMsg", `Test failed: ${s.message}`, true); });
  on("tests_start", (d) => { Object.assign(testStatus, d.status); d.ids.forEach((i) => delete testFailures[i]); renderTests();
    $("testsBar").style.width = "0"; say("testsMsg", `Running ${d.ids.length} test${d.ids.length > 1 ? "s" : ""}…`);
    $("testsAll").disabled = true; window.testsRunning = d.ids.length; window.testsDone = 0; });
  on("test_result", (d) => { testStatus[d.id] = d.status; setTestRow(d.id); if (d.next) { testStatus[d.next] = "running"; setTestRow(d.next); }
    window.testsDone += 1; $("testsBar").style.width = `${(100 * window.testsDone) / (window.testsRunning || 1)}%`; testsBig(); });
  on("tests_done", (d) => { testStatus = d.status; Object.assign(testFailures, d.failures); renderTests(); testsBig(d.summary);
    $("testsBar").style.width = "100%"; $("testsAll").disabled = false;
    const bad = d.summary.failed + d.summary.error;
    say("testsMsg", bad ? `${bad} failing: see the red details below.` : `All ${d.summary.passed} passed in ${d.summary.duration.toFixed(1)} s.`, bad > 0); });
}

// ---------------------------------------------------------------- start
async function init() {
  makeCharts();
  const p = await api("/api/params");
  SPEC = p.params; DEFAULTS = p.defaults; PRESETS = p.presets; MODELS = p.models;
  buildForm(); setForm(DEFAULTS); fillModels();
  $("presetSel").innerHTML = Object.keys(PRESETS).map((n) => `<option>${n}</option>`).join("");
  $("presetSel").onchange = () => { setForm({ ...DEFAULTS, ...PRESETS[$("presetSel").value] }); say("trainMsg", `Preset loaded: ${$("presetSel").value}.`); };
  $("resetBtn").onclick = () => { setForm(DEFAULTS); $("presetSel").selectedIndex = 0; say("trainMsg", "Back to the project's settings."); };
  $("startBtn").onclick = async () => { try { await api("/api/train/start", getForm()); } catch (e) { say("trainMsg", e.message, true); } };
  $("stopBtn").onclick = () => api("/api/train/stop", {}).then(() => say("trainMsg", "Stopping after the current update…"));
  $("pilotSel").onchange = async () => {
    try { const r = await api("/api/view", { model: $("pilotSel").value }); lastFrame = null; say("pilotMsg", `Now flying: ${r.label}`); }
    catch (e) { say("pilotMsg", e.message, true); } };
  $("viewSpeed").onchange = () => api("/api/view", { speed: Number($("viewSpeed").value) });
  $("viewOn").onchange = () => api("/api/view", { enabled: $("viewOn").checked });
  const ev = Object.fromEntries(SPEC.filter((q) => q.group === "Evaluation").map((q) => [q.key, q.default]));
  $("evalN").value = ev.eval_episodes; $("evalSeed").value = ev.eval_seed; $("evalDet").checked = ev.deterministic;
  $("evalBtn").onclick = async () => {
    try { await api("/api/eval/start", { model: $("evalModel").value, eval_episodes: Number($("evalN").value),
      eval_seed: Number($("evalSeed").value), deterministic: $("evalDet").checked, watch: $("evalWatch").checked,
      speed: Number($("evalSpeed").value) }); }
    catch (e) { say("evalMsg", e.message, true); } };
  $("evalStop").onclick = () => api("/api/eval/stop", {});
  $("testsAll").onclick = () => runTests(null);
  $("testsFailed").onclick = () => { const ids = TESTS.filter((t) => ["failed", "error"].includes(testStatus[t.id])).map((t) => t.id);
    if (ids.length) runTests(ids); else say("testsMsg", "Nothing failed: nothing to rerun."); };

  connect(); // live stream first: the lander appears immediately
  const s = await api("/api/state");
  if (s.pilot) onPilot(s.pilot);
  $("viewSpeed").value = String(s.view.speed); $("viewOn").checked = s.view.enabled;
  s.history.progress.forEach(onProgress); s.history.losses.forEach(onLosses); s.history.weights.forEach(onWeights);
  if (s.history.last_weights) onWeightsDetail(s.history.last_weights);
  const running = ["running", "starting"].includes(s.train.status);
  trainButtons(running);
  if (s.train.cfg) setForm(s.train.cfg);
  if (s.pilot?.training) $("pilotSel").value = "training";
  if (s.eval.episodes?.length) { onEvalStart({ ...s.eval, n: s.eval.n || s.eval.episodes.length }); s.eval.episodes.forEach(onEvalEpisode);
    if (s.eval.status !== "running") onEvalDone(s.eval); }
  if (!running) setStatus("ready · live view on");
  testStatus = s.tests.status; testFailures = s.tests.failures;
  TESTS = s.tests.items; renderTests();
  if (!TESTS.length) api("/api/tests/collect", {}).then((r) => { TESTS = r.items; renderTests(); });
  loadExperiments();
}
init().catch((e) => setStatus(`could not start: ${e.message}`, true));
