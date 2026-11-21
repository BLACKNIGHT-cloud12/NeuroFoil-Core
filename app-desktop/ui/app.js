/* NeuroFoil Flight Deck — UI logic. Vanilla JS, no dependencies.
   Charts are hand-rendered SVG; data comes from the local FastAPI sidecar. */
"use strict";

const $ = (id) => document.getElementById(id);
const API = ""; // same origin — the Electron window and the sidecar share it

const NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}
const fmt = (v, d = 4) => (v == null || !isFinite(v) ? "—" : Number(v).toFixed(d));
const fmtSci = (v, d = 3) =>
  v == null || !isFinite(v) ? "—" : Number(v).toExponential(d).replace("e", "e");

/* ---------- NACA 4-digit geometry (same convention as src/geometry/naca.py) ---------- */
function nacaPoints(m, p, t, n = 121) {
  const upper = [], lower = [];
  for (let i = 0; i <= n; i++) {
    const x = 0.5 * (1 - Math.cos((Math.PI * i) / n));
    const yt =
      5 * t *
      (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x +
       0.2843 * x * x * x - 0.1036 * x * x * x * x);
    let yc = 0, dyc = 0;
    if (m > 0 && x < p) {
      yc = (m / (p * p)) * (2 * p * x - x * x);
      dyc = ((2 * m) / (p * p)) * (p - x);
    } else if (m > 0) {
      yc = (m / ((1 - p) * (1 - p))) * (1 - 2 * p + 2 * p * x - x * x);
      dyc = ((2 * m) / ((1 - p) * (1 - p))) * (p - x);
    }
    const theta = Math.atan(dyc);
    upper.push([x, yc + yt * Math.cos(theta)]);
    lower.push([x, yc - yt * Math.cos(theta)]);
  }
  return { upper, lower };
}

/* ---------- state ---------- */
const state = {
  m: 0.02, p: 0.4, t: 0.12, alpha: 4.0, reynolds: 6000000,
  meta: null, lastPredict: null,
};
const PRESETS = [
  { code: "0012", m: 0.0, p: 0.2, t: 0.12 },
  { code: "2412", m: 0.02, p: 0.4, t: 0.12 },
  { code: "4412", m: 0.04, p: 0.4, t: 0.12 },
  { code: "2415", m: 0.02, p: 0.4, t: 0.15 },
  { code: "4415", m: 0.04, p: 0.4, t: 0.15 },
];
const RES = [1000000, 2000000, 3000000, 6000000];
const MAHAL_P99 = 4.45, MAHAL_CHI2 = 5.78;

/* ---------- helpers ---------- */
let predictTimer = null, polarTimer = null;
function schedulePredict(delay = 90) {
  clearTimeout(predictTimer);
  predictTimer = setTimeout(runPredict, delay);
}
function schedulePolar(delay = 450) {
  clearTimeout(polarTimer);
  polarTimer = setTimeout(runPolar, delay);
}
async function post(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || r.statusText);
  }
  return r.json();
}

/* ---------- linear scale ---------- */
function scale(domain, range) {
  const [d0, d1] = domain, [r0, r1] = range;
  const k = (r1 - r0) / (d1 - d0 || 1);
  return (v) => r0 + (v - d0) * k;
}

/* ---------- Cp chart ---------- */
function drawCp(res) {
  const svg = $("cp-chart");
  svg.innerHTML = "";
  const W = 760, H = 340, padL = 46, padR = 14, padT = 16, padB = 34;
  const xs = scale([0, 1], [padL, W - padR]);
  const cpU = res.cp_upper, cpL = res.cp_lower;
  const bandU = res.uq.cp_upper_ci95, bandUh = res.uq.cp_upper_ci95_high;
  const bandL = res.uq.cp_lower_ci95, bandLh = res.uq.cp_lower_ci95_high;
  // Domain from the predicted curves; conformal bands may spike at edge
  // stations, so they are clipped to this range instead of stretching it.
  let lo = 1.2, hi = -4.2;
  const scan = (a) => { for (const v of a || []) if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } };
  scan(cpU); scan(cpL);
  lo -= 0.45; hi += 0.3;
  const ys = scale([lo, hi], [H - padB, padT]); // inverted: Cp negative up

  // grid + ticks
  for (let g = Math.ceil(lo * 2) / 2; g <= hi; g += 0.5) {
    svgEl("line", { x1: padL, x2: W - padR, y1: ys(g), y2: ys(g),
      stroke: "#1A2940", "stroke-width": 0.6 }, svg);
    svgEl("text", { x: padL - 8, y: ys(g) + 3.5, "text-anchor": "end",
      fill: "#7A8AA5", "font-size": 10,
      "font-family": "JetBrains Mono, monospace" }, svg).textContent = g.toFixed(1);
  }
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    svgEl("line", { x1: xs(g), x2: xs(g), y1: padT, y2: H - padB,
      stroke: "#1A2940", "stroke-width": 0.6 }, svg);
    svgEl("text", { x: xs(g), y: H - padB + 16, "text-anchor": "middle",
      fill: "#7A8AA5", "font-size": 10,
      "font-family": "JetBrains Mono, monospace" }, svg).textContent = g.toFixed(2);
  }
  svgEl("line", { x1: padL, x2: padL, y1: padT, y2: H - padB, stroke: "#24344D", "stroke-width": 1 }, svg);
  svgEl("line", { x1: padL, x2: W - padR, y1: H - padB, y2: H - padB, stroke: "#24344D", "stroke-width": 1 }, svg);
  svgEl("text", { x: W / 2, y: H - 4, "text-anchor": "middle", fill: "#7A8AA5", "font-size": 10.5 }, svg).textContent = "x/c";
  svgEl("text", { x: 12, y: padT + 4, fill: "#7A8AA5", "font-size": 10.5 }, svg).textContent = "Cp";

  const poly = (pts, X, Y) => pts.map(([vx, vy], i) => `${i ? "L" : "M"}${X(vx).toFixed(1)},${Y(vy).toFixed(1)}`).join(" ");

  const clipId = "cp-clip";
  const defs = svgEl("defs", {}, svg);
  const clip = svgEl("clipPath", { id: clipId }, defs);
  svgEl("rect", { x: padL, y: padT, width: W - padL - padR, height: H - padT - padB }, clip);
  const plotG = svgEl("g", { "clip-path": `url(#${clipId})` }, svg);

  // conformal ribbons (upper then lower)
  const ribbon = (cp, loArr, hiArr, fill) => {
    if (!cp || !loArr || !hiArr || !loArr.length) return;
    const host = plotG;
    const up = cp.map((v, i) => `${i ? "L" : "M"}${xs(i / (cp.length - 1)).toFixed(1)},${ys(loArr[i] ?? v).toFixed(1)}`).join(" ");
    const dn = cp.map((v, i) => `${xs(i / (cp.length - 1)).toFixed(1)},${ys(hiArr[i] ?? v).toFixed(1)}`).reverse().map(s => s.replace(/^M/, "L")).join(" ");
    svgEl("path", { d: up + " " + dn + " Z", fill, stroke: "none" }, host);
  };
  ribbon(cpU, bandU, bandUh, "rgba(90,176,255,0.13)");
  ribbon(cpL, bandL, bandLh, "rgba(90,176,255,0.13)");

  if (cpU && cpU.length) {
    svgEl("path", { d: poly(cpU.map((v, i) => [i / (cpU.length - 1), v]), xs, ys),
      fill: "none", stroke: "#5AB0FF", "stroke-width": 1.6, "clip-path": `url(#${clipId})` }, svg);
    svgEl("path", { d: poly(cpL.map((v, i) => [i / (cpL.length - 1), v]), xs, ys),
      fill: "none", stroke: "#5AB0FF", "stroke-width": 1.6, opacity: 0.85,
      "clip-path": `url(#${clipId})` }, svg);
  }
}

/* ---------- airfoil mini + wingbox schematic ---------- */
function drawAirfoilMini(res) {
  const svg = $("airfoil-mini");
  svg.innerHTML = "";
  const { upper, lower } = nacaPoints(state.m, state.p, state.t, 80);
  const xs = scale([0, 1], [12, 228]);
  const ys = scale([-0.12, 0.12], [108, 12]);
  const poly = (pts) => pts.map(([vx, vy], i) => `${i ? "L" : "M"}${xs(vx).toFixed(1)},${ys(vy).toFixed(1)}`).join(" ");
  svgEl("path", { d: poly(upper) + " " + poly(lower.slice().reverse()).replace(/^M/, "L") + " Z",
    fill: "rgba(90,176,255,0.10)", stroke: "#5AB0FF", "stroke-width": 1.1 }, svg);
  const wb = res && res.wingbox;
  if (!wb) return;
  if (Array.isArray(wb.fuel_bay_outline) && wb.fuel_bay_outline.length > 2) {
    svgEl("path", { d: poly(wb.fuel_bay_outline.map(([vx, vy]) => [vx, vy])) + " Z",
      fill: "rgba(111,207,151,0.12)", stroke: "rgba(111,207,151,0.45)",
      "stroke-width": 0.8, "stroke-dasharray": "3 2" }, svg);
  } else {
    const xf = wb.x_front_spar, xr = wb.x_rear_spar;
    if (xf != null && xr != null) {
      const yAt = (x) => {
        const i = Math.round(x * 80);
        return [ys(upper[Math.min(80, i)][1]), ys(lower[Math.min(80, i)][1])];
      };
      const [fu, fl] = yAt(Math.max(xf, 0.05));
      svgEl("rect", { x: xs(xf), y: Math.min(fu, fl), width: Math.max(4, xs(xr) - xs(xf)),
        height: Math.max(2, Math.abs(fl - fu)),
        fill: "rgba(111,207,151,0.12)", stroke: "rgba(111,207,151,0.45)",
        "stroke-width": 0.8, "stroke-dasharray": "3 2" }, svg);
    }
  }
  for (const key of ["front_spar_coords", "rear_spar_coords"]) {
    const seg = wb[key];
    if (Array.isArray(seg) && seg.length > 1) {
      svgEl("path", { d: poly(seg.map(([vx, vy]) => [vx, vy])), fill: "none",
        stroke: "#FFB454", "stroke-width": 2.2 }, svg);
    }
  }
}

function drawWingbox(res) {
  drawAirfoilMini(res);
  const host = $("wingbox-readouts");
  const wb = res && res.wingbox;
  if (!wb) { host.innerHTML = "<div class='opt-bad'>wingbox unavailable for this geometry</div>"; return; }
  const rows = [
    ["front spar x", wb.x_front_spar, 3],
    ["rear spar x", wb.x_rear_spar, 3],
    ["front spar height", wb.h_front_spar, 4],
    ["rear spar height", wb.h_rear_spar, 4],
    ["total area", wb.total_area, 5],
    ["fuel tank area", wb.fuel_tank_area, 5],
    ["fuel area fraction", wb.fuel_area_fraction, 4],
    ["second moment Ixx", wb.i_xx, 5],
    ["min section modulus", wb.min_section_modulus, 5],
    ["bending efficiency", wb.bending_efficiency, 3],
    ["centroid y", wb.y_centroid, 4],
  ];
  host.innerHTML = "";
  for (const [label, v, d] of rows) {
    if (v == null || !isFinite(Number(v))) continue;
    const div = document.createElement("div");
    const dt = document.createElement("dt"); dt.textContent = label;
    const dd = document.createElement("dd"); dd.textContent = fmt(v, d);
    div.append(dt, dd); host.append(div);
  }
}

/* ---------- OOD gauge ---------- */
function drawGauge(mahal) {
  const svg = $("ood-gauge");
  svg.innerHTML = "";
  const cx = 110, cy = 122, R = 78;
  const a0 = Math.PI, a1 = 0;
  const maxV = Math.max(MAHAL_CHI2 * 1.35, (mahal || 0) * 1.15, 2);
  const ang = (v) => a0 + ((Math.min(v, maxV) / maxV) * (a1 - a0));
  const pt = (v, r) => [cx + r * Math.cos(ang(v)), cy + r * Math.sin(ang(v))];
  const arc = (v0, v1, r, stroke, width) => {
    const [x0, y0] = pt(v0, r), [x1, y1] = pt(v1, r);
    svgEl("path", { d: `M${x0.toFixed(1)},${y0.toFixed(1)} A${r},${r} 0 0 1 ${x1.toFixed(1)},${y1.toFixed(1)}`,
      fill: "none", stroke, "stroke-width": width }, svg);
  };
  arc(0, maxV, R, "#1A2940", 10);
  arc(0, MAHAL_P99, R, "rgba(111,207,151,0.75)", 10);
  arc(MAHAL_P99, MAHAL_CHI2, R, "rgba(255,180,84,0.8)", 10);
  arc(MAHAL_CHI2, maxV, R, "rgba(255,107,107,0.8)", 10);
  for (const [v, lbl] of [[0, "0"], [MAHAL_P99, "p99"], [MAHAL_CHI2, "χ²"], [maxV, fmt(maxV, 1)]]) {
    const [tx, ty] = pt(v, R + 18);
    svgEl("text", { x: tx, y: ty + 3, "text-anchor": "middle", fill: "#7A8AA5",
      "font-size": 9, "font-family": "JetBrains Mono, monospace" }, svg).textContent = lbl;
  }
  const [nx, ny] = pt(mahal ?? 0, R - 16);
  const needle = svgEl("line", { x1: cx, y1: cy, x2: nx, y2: ny,
    stroke: "#DCE6F5", "stroke-width": 2, "stroke-linecap": "round" }, svg);
  needle.style.transformOrigin = `${cx}px ${cy}px`;
  needle.style.transformBox = "view-box";
  needle.style.transition = "transform 500ms cubic-bezier(.2,.8,.2,1)";
  requestAnimationFrame(() => { needle.style.transform = "rotate(0deg)"; });
  svgEl("circle", { cx, cy, r: 4, fill: "#DCE6F5" }, svg);
  const [vx, vy] = pt((mahal ?? 0), R - 34);
  svgEl("text", { x: cx, y: vy + 26, "text-anchor": "middle", fill: "#DCE6F5",
    "font-size": 15, "font-family": "JetBrains Mono, monospace", "font-weight": 600 }, svg)
    .textContent = mahal == null ? "—" : fmt(mahal, 2);
}

/* ---------- polar charts ---------- */
function miniChart(svgId, rows, yKey, yLabel, opts = {}) {
  const svg = $(svgId);
  svg.innerHTML = "";
  const W = 400, H = 240, padL = 44, padR = 12, padT = 14, padB = 30;
  if (!rows.length) return;
  const xs = scale([-6, 16], [padL, W - padR]);
  let lo = Infinity, hi = -Infinity;
  for (const r of rows) { const v = r[yKey]; if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
  if (!isFinite(lo)) return;
  const pad = (hi - lo) * 0.12 || 0.1; lo -= pad; hi += pad;
  const ys = scale([lo, hi], [H - padB, padT]);
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    const gy = padT + g * (H - padB - padT);
    const gv = hi - g * (hi - lo);
    svgEl("line", { x1: padL, x2: W - padR, y1: gy, y2: gy, stroke: "#1A2940", "stroke-width": 0.6 }, svg);
    svgEl("text", { x: padL - 6, y: gy + 3, "text-anchor": "end", fill: "#7A8AA5", "font-size": 9,
      "font-family": "JetBrains Mono, monospace" }, svg).textContent = fmt(gv, 2);
  }
  for (const a of [-6, -2, 2, 6, 10, 14, 16]) {
    svgEl("line", { x1: xs(a), x2: xs(a), y1: padT, y2: H - padB, stroke: "#1A2940", "stroke-width": 0.6 }, svg);
    svgEl("text", { x: xs(a), y: H - padB + 14, "text-anchor": "middle", fill: "#7A8AA5", "font-size": 9,
      "font-family": "JetBrains Mono, monospace" }, svg).textContent = a;
  }
  const path = rows.map((r, i) => `${i ? "L" : "M"}${xs(r.alpha).toFixed(1)},${ys(r[yKey]).toFixed(1)}`).join(" ");
  svgEl("path", { d: path, fill: "none", stroke: "#5AB0FF", "stroke-width": 1.6 }, svg);
  if (opts.marker != null && isFinite(opts.marker)) {
    const mx = xs(opts.marker);
    svgEl("line", { x1: mx, x2: mx, y1: padT, y2: H - padB, stroke: "#FFB454", "stroke-width": 1.2, "stroke-dasharray": "4 3" }, svg);
  }
  if (opts.maxMarker) {
    const best = rows.reduce((b, r) => (isFinite(r[yKey]) && (!b || r[yKey] > b[yKey]) ? r : b), null);
    if (best) {
      svgEl("circle", { cx: xs(best.alpha), cy: ys(best[yKey]), r: 3.5, fill: "#6FCF97" }, svg);
      svgEl("text", { x: xs(best.alpha) + 6, y: ys(best[yKey]) - 6, fill: "#6FCF97", "font-size": 9.5,
        "font-family": "JetBrains Mono, monospace" }, svg).textContent = `${fmt(best[yKey], 2)} @ ${fmt(best.alpha, 1)}°`;
    }
  }
  svgEl("text", { x: W / 2, y: H - 2, "text-anchor": "middle", fill: "#7A8AA5", "font-size": 10 }, svg).textContent = "α (deg)";
  svgEl("text", { x: 8, y: padT + 2, fill: "#7A8AA5", "font-size": 10 }, svg).textContent = yLabel;
}

/* ---------- data flows ---------- */
async function runPredict() {
  document.body.classList.add("busy");
  try {
    const res = await post("/api/predict", {
      m: state.m, p: state.p, t: state.t,
      alpha_deg: state.alpha, reynolds: state.reynolds,
    });
    state.lastPredict = res;
    $("ro-cl").textContent = fmt(res.cl, 4);
    $("ro-cd").textContent = fmtSci(res.cd, 4);
    $("ro-cm").textContent = fmt(res.cm, 4);
    $("ro-ld").textContent = fmt(res.lift_to_drag, 2);
    $("band-cl").textContent = `± ${fmt(((res.uq.cl_ci95[1] - res.uq.cl_ci95[0]) / 2), 3)}`;
    $("band-cd").textContent = `± ${fmtSci(((res.uq.cd_ci95[1] - res.uq.cd_ci95[0]) / 2), 2)}`;
    $("band-cm").textContent = `± ${fmt(((res.uq.cm_ci95[1] - res.uq.cm_ci95[0]) / 2), 3)}`;
    drawCp(res);
    drawAirfoilMini(res);
    drawWingbox(res);
    const mahal = res.uq.mahalanobis_dist;
    drawGauge(mahal);
    $("uq-conf").textContent = res.uq.confidence_score == null ? "—" : `${fmt(res.uq.confidence_score, 1)}%`;
    $("uq-mahal").textContent = fmt(mahal, 3);
    const verdict = $("ood-verdict");
    const warnList = $("warn-list");
    warnList.innerHTML = "";
    const warnings = res.uq.warnings || [];
    if (res.uq.is_ood) {
      verdict.textContent = "OUT OF DOMAIN";
      verdict.className = "ood-verdict warn";
    } else if (mahal != null && mahal > MAHAL_P99) {
      verdict.textContent = "NEAR BOUNDARY";
      verdict.className = "ood-verdict caution";
    } else {
      verdict.textContent = "IN DOMAIN";
      verdict.className = "ood-verdict ok";
    }
    if (!warnings.length) {
      const li = document.createElement("li");
      li.className = "warn-empty";
      li.textContent = "All stations nominal.";
      warnList.append(li);
    } else {
      for (const w of warnings) {
        const li = document.createElement("li");
        li.textContent = String(w).replace(/^WARNING:\s*/i, "");
        warnList.append(li);
      }
    }
    $("bk-model").textContent = res.model_backend || "—";
    $("bk-field").textContent = res.field_backend || "—";
    $("bk-prov").textContent = res.model_provenance || "—";
  } catch (err) {
    $("ood-verdict").textContent = "BACKEND ERROR";
    $("ood-verdict").className = "ood-verdict warn";
    const li = document.createElement("li");
    li.className = "warn-level-high";
    li.textContent = String(err.message || err);
    $("warn-list").innerHTML = "";
    $("warn-list").append(li);
  } finally {
    document.body.classList.remove("busy");
  }
}

async function runPolar() {
  try {
    const res = await post("/api/polar", {
      m: state.m, p: state.p, t: state.t, reynolds: state.reynolds,
    });
    miniChart("polar-cl", res.rows, "cl", "CL vs alpha", { marker: state.alpha });
    miniChart("polar-ld", res.rows, "lift_to_drag", "L/D vs alpha", { maxMarker: true, marker: state.alpha });
  } catch { /* polar is decorative; keep last good chart */ }
}

/* ---------- controls ---------- */
function syncLabels() {
  $("v-m").textContent = state.m.toFixed(3);
  $("v-p").textContent = state.p.toFixed(2);
  $("v-t").textContent = state.t.toFixed(3);
  $("v-alpha").textContent = `${state.alpha >= 0 ? "+" : ""}${state.alpha.toFixed(1)}°`;
  const md = Math.round(state.m * 100), pd = state.m > 0 ? Math.round(state.p * 10) : 0;
  const td = String(Math.round(state.t * 100)).padStart(2, "0");
  $("v-code").textContent = `${md}${pd}${td}`;
  for (const b of document.querySelectorAll(".preset-btn")) {
    b.classList.toggle("is-current", PRESETS.some(
      (pr) => pr.m === state.m && pr.p === state.p && pr.t === state.t && b.textContent.startsWith(pr.code)));
  }
  for (const b of document.querySelectorAll(".re-btn")) {
    b.classList.toggle("is-current", Number(b.dataset.re) === state.reynolds);
  }
}

function wireControls() {
  const presets = $("presets");
  for (const pr of PRESETS) {
    const b = document.createElement("button");
    b.className = "preset-btn"; b.textContent = pr.code;
    b.addEventListener("click", () => {
      state.m = pr.m; state.p = pr.p; state.t = pr.t;
      $("in-m").value = pr.m; $("in-p").value = pr.p; $("in-t").value = pr.t;
      syncLabels(); schedulePredict(); schedulePolar();
    });
    presets.append(b);
  }
  const reRow = $("re-row");
  for (const re of RES) {
    const b = document.createElement("button");
    b.className = "re-btn"; b.dataset.re = re;
    b.textContent = re === 1000000 ? "1e6" : `${re / 1e6}e6`;
    b.addEventListener("click", () => {
      state.reynolds = re; syncLabels(); schedulePredict(); schedulePolar();
    });
    reRow.append(b);
  }
  for (const id of ["in-m", "in-p", "in-t", "in-alpha"]) {
    $(id).addEventListener("input", (e) => {
      const key = id.replace("in-", "");
      state[key] = parseFloat(e.target.value);
      syncLabels(); schedulePredict();
      if (key !== "alpha") schedulePolar();
    });
  }
  $("bay-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    for (const t of document.querySelectorAll(".tab")) t.classList.toggle("is-active", t === btn);
    for (const bay of document.querySelectorAll(".bay")) bay.classList.add("is-hidden");
    $(`bay-${btn.dataset.tab}`).classList.remove("is-hidden");
  });
  $("s-run").addEventListener("click", async () => {
    const btn = $("s-run");
    btn.disabled = true;
    $("s-status").textContent = "SLSQP running — maximizing L/D …";
    try {
      const res = await post("/api/optimize", {
        cl_target: parseFloat($("s-cl").value),
        t_min: parseFloat($("s-tmin").value), t_max: parseFloat($("s-tmax").value),
        m_max: parseFloat($("s-mmax").value),
        alpha_low: 1.0, alpha_high: 7.0,
      });
      const bp = res.baseline_params || {}, op = res.opt_params || {};
      const f4 = (v) => fmt(Array.isArray(v) ? v[3] : v?.alpha, 2);
      $("s-result").innerHTML =
        `<div>success: <b style="color:${res.success ? "var(--ok)" : "var(--warn)"}">${res.success}</b></div>` +
        `<div>iterations: ${res.iterations ?? "—"} · ${fmt(res.elapsed_seconds, 2)} s</div>` +
        `<div class="opt-bad">constraints met: ${res.constraints_met ?? "—"}</div>` +
        `<div>start (m,p,t,α): ${arrStr(res.baseline_raw || bp)}</div>` +
        `<div>optimum (m,p,t,α): ${arrStr(op)}</div>`;
      $("s-status").textContent = res.solver_message || "done";
    } catch (err) {
      $("s-result").textContent = String(err.message || err);
      $("s-status").textContent = "search failed";
    } finally {
      btn.disabled = false;
    }
  });
}

function arrStr(v) {
  if (Array.isArray(v)) return `[${v.map((x) => fmt(Number(x), 3)).join(", ")}]`;
  if (v && typeof v === "object") return JSON.stringify(v);
  return "—";
}

/* ---------- boot ---------- */
async function boot() {
  try {
    const meta = await (await fetch("/api/meta")).json();
    state.meta = meta;
    $("provenance-chip").textContent = `provenance: ${meta.model_provenance}`;
    $("fingerprint-chip").textContent = `artifact ${meta.artifact_fingerprint}`;
    $("uq-caln").textContent = meta.calibration_n ?? "—";
    $("bk-model").textContent = meta.model_backend;
    $("bk-prov").textContent = meta.model_provenance;
  } catch {
    $("provenance-chip").textContent = "backend offline";
  }
  wireControls();
  syncLabels();
  const flash = document.querySelector(".pfd");
  flash.classList.add("load-flash");
  setTimeout(() => flash.classList.remove("load-flash"), 600);
  await runPredict();
  runPolar();
}

boot();
