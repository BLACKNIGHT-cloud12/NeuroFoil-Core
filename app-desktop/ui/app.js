/* NeuroFoil Flight Deck v2 — smooth instrument UI.
   - every displayed value eases toward the newest backend response (no snaps)
   - three.js hero: a live-morphing wing section with flow particles
   - hand-rolled SVG charts, no dependencies beyond the vendored three module */
import * as THREE from "three";

const $ = (id) => document.getElementById(id);
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

/* ---------- NACA 4-digit geometry ---------- */
function nacaPoints(m, p, t, n = 60) {
  const upper = [], lower = [];
  for (let i = 0; i <= n; i++) {
    const x = 0.5 * (1 - Math.cos((Math.PI * i) / n));
    const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x +
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
  // closed ring: LE→TE along the upper surface, then TE→LE along the lower
  return { upper, lower, ring: [...upper, ...lower.slice(1).reverse()] };
}

/* ---------- tween core: display chases target every frame ---------- */
const CP_N = 161, POLAR_N = 45;
const zeros = (n) => new Array(n).fill(0);
const display = {
  cl: 0, cd: 0, cm: 0, ld: 0, mahal: 0, conf: 0,
  clCi: [0, 0], cdCi: [0, 0], cmCi: [0, 0],
  cpU: zeros(CP_N), cpL: zeros(CP_N),
  bU: zeros(CP_N), bUh: zeros(CP_N), bL: zeros(CP_N), bLh: zeros(CP_N),
  polarCl: zeros(POLAR_N), polarLd: zeros(POLAR_N),
};
const target = {
  cl: 0, cd: 0, cm: 0, ld: 0, mahal: 0, conf: 0,
  clCi: [0, 0], cdCi: [0, 0], cmCi: [0, 0],
  cpU: zeros(CP_N), cpL: zeros(CP_N),
  bU: zeros(CP_N), bUh: zeros(CP_N), bL: zeros(CP_N), bLh: zeros(CP_N),
  polarCl: zeros(POLAR_N), polarLd: zeros(POLAR_N),
};
let firstFill = true;

function lerpInto(disp, tgt, k) {
  let moving = false;
  for (const key of Object.keys(tgt)) {
    const tv = tgt[key], dv = disp[key];
    if (Array.isArray(tv)) {
      if (!Array.isArray(dv) || dv.length !== tv.length) { disp[key] = tv.slice(); moving = true; continue; }
      for (let i = 0; i < tv.length; i++) {
        const next = dv[i] + (tv[i] - dv[i]) * k;
        if (Math.abs(tv[i] - dv[i]) > 1e-5) moving = true;
        disp[key][i] = next;
      }
    } else if (typeof tv === "number") {
      const next = dv + (tv - dv) * k;
      if (Math.abs(tv - dv) > 1e-5) moving = true;
      disp[key] = next;
    } else { disp[key] = tv; }
  }
  return moving;
}

/* ---------- state ---------- */
const state = { m: 0.02, p: 0.4, t: 0.12, alpha: 4.0, reynolds: 6000000 };
const PRESETS = [
  { code: "0012", m: 0.0, p: 0.2, t: 0.12 },
  { code: "2412", m: 0.02, p: 0.4, t: 0.12 },
  { code: "4412", m: 0.04, p: 0.4, t: 0.12 },
  { code: "2415", m: 0.02, p: 0.4, t: 0.15 },
  { code: "4415", m: 0.04, p: 0.4, t: 0.15 },
];
const RES = [1000000, 2000000, 3000000, 6000000];
const MAHAL_P99 = 4.45, MAHAL_CHI2 = 5.78;

function scale(domain, range) {
  const [d0, d1] = domain, [r0, r1] = range;
  const k = (r1 - r0) / (d1 - d0 || 1);
  return (v) => r0 + (v - d0) * k;
}

/* ---------- Cp chart (drawn from the tweened display state) ---------- */
function drawCp() {
  const svg = $("cp-chart");
  svg.innerHTML = "";
  const W = 760, H = 340, padL = 46, padR = 14, padT = 16, padB = 34;
  const xs = scale([0, 1], [padL, W - padR]);
  const { cpU, cpL, bU, bUh, bL, bLh } = display;
  let lo = 1.2, hi = -4.2;
  const scan = (a) => { for (const v of a) if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } };
  scan(cpU); scan(cpL);
  lo -= 0.45; hi += 0.3;
  const ys = scale([lo, hi], [H - padB, padT]);

  for (let g = Math.ceil(lo * 2) / 2; g <= hi; g += 0.5) {
    svgEl("line", { x1: padL, x2: W - padR, y1: ys(g), y2: ys(g), stroke: "#1A2940", "stroke-width": 0.6 }, svg);
    svgEl("text", { x: padL - 8, y: ys(g) + 3.5, "text-anchor": "end", fill: "#7A8AA5",
      "font-size": 10, "font-family": "JetBrains Mono, monospace" }, svg).textContent = g.toFixed(1);
  }
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    svgEl("line", { x1: xs(g), x2: xs(g), y1: padT, y2: H - padB, stroke: "#1A2940", "stroke-width": 0.6 }, svg);
    svgEl("text", { x: xs(g), y: H - padB + 16, "text-anchor": "middle", fill: "#7A8AA5",
      "font-size": 10, "font-family": "JetBrains Mono, monospace" }, svg).textContent = g.toFixed(2);
  }
  svgEl("line", { x1: padL, x2: padL, y1: padT, y2: H - padB, stroke: "#24344D", "stroke-width": 1 }, svg);
  svgEl("line", { x1: padL, x2: W - padR, y1: H - padB, y2: H - padB, stroke: "#24344D", "stroke-width": 1 }, svg);
  svgEl("text", { x: W / 2, y: H - 4, "text-anchor": "middle", fill: "#7A8AA5", "font-size": 10.5 }, svg).textContent = "x/c";
  svgEl("text", { x: 12, y: padT + 4, fill: "#7A8AA5", "font-size": 10.5 }, svg).textContent = "Cp";

  const clipId = "cp-clip";
  const defs = svgEl("defs", {}, svg);
  const clip = svgEl("clipPath", { id: clipId }, defs);
  svgEl("rect", { x: padL, y: padT, width: W - padL - padR, height: H - padT - padB }, clip);
  const plotG = svgEl("g", { "clip-path": `url(#${clipId})` }, svg);

  const poly = (pts) => pts.map(([vx, vy], i) => `${i ? "L" : "M"}${xs(vx).toFixed(1)},${ys(vy).toFixed(1)}`).join(" ");
  const ribbon = (cp, loArr, hiArr, fill) => {
    if (!loArr || !loArr.length || loArr.length !== cp.length) return;
    const up = cp.map((v, i) => `${i ? "L" : "M"}${xs(i / (cp.length - 1)).toFixed(1)},${ys(loArr[i] ?? v).toFixed(1)}`).join(" ");
    const dn = cp.map((v, i) => `${xs(i / (cp.length - 1)).toFixed(1)},${ys(hiArr[i] ?? v).toFixed(1)}`)
      .reverse().map((s) => s.replace(/^M/, "L")).join(" ");
    svgEl("path", { d: up + " " + dn + " Z", fill, stroke: "none" }, plotG);
  };
  ribbon(cpU, bU, bUh, "rgba(90,176,255,0.13)");
  ribbon(cpL, bL, bLh, "rgba(90,176,255,0.13)");

  svgEl("path", { d: poly(cpU.map((v, i) => [i / (cpU.length - 1), v])), fill: "none",
    stroke: "#5AB0FF", "stroke-width": 1.6, "clip-path": `url(#${clipId})` }, plotG);
  svgEl("path", { d: poly(cpL.map((v, i) => [i / (cpL.length - 1), v])), fill: "none",
    stroke: "#5AB0FF", "stroke-width": 1.6, opacity: 0.85, "clip-path": `url(#${clipId})` }, plotG);
}

/* ---------- gauge ---------- */
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
  svgEl("line", { x1: cx, y1: cy, x2: nx, y2: ny, stroke: "#DCE6F5", "stroke-width": 2, "stroke-linecap": "round" }, svg);
  svgEl("circle", { cx, cy, r: 4, fill: "#DCE6F5" }, svg);
  const [, vy] = pt(mahal ?? 0, R - 34);
  svgEl("text", { x: cx, y: vy + 26, "text-anchor": "middle", fill: "#DCE6F5",
    "font-size": 15, "font-family": "JetBrains Mono, monospace", "font-weight": 600 }, svg)
    .textContent = mahal == null ? "—" : fmt(mahal, 2);
}

/* ---------- airfoil mini + wingbox (small panels snap) ---------- */
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
  } else if (wb.x_front_spar != null && wb.x_rear_spar != null) {
    const yAt = (x) => {
      const i = Math.round(x * 80);
      return [ys(upper[Math.min(80, i)][1]), ys(lower[Math.min(80, i)][1])];
    };
    const [fu, fl] = yAt(Math.max(wb.x_front_spar, 0.05));
    svgEl("rect", { x: xs(wb.x_front_spar), y: Math.min(fu, fl),
      width: Math.max(4, xs(wb.x_rear_spar) - xs(wb.x_front_spar)),
      height: Math.max(2, Math.abs(fl - fu)),
      fill: "rgba(111,207,151,0.12)", stroke: "rgba(111,207,151,0.45)",
      "stroke-width": 0.8, "stroke-dasharray": "3 2" }, svg);
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

/* ---------- polar charts (tweened) ---------- */
function miniChart(svgId, series, yLabel, opts = {}) {
  const svg = $(svgId);
  svg.innerHTML = "";
  const W = 400, H = 240, padL = 44, padR = 12, padT = 14, padB = 30;
  let lo = Infinity, hi = -Infinity;
  for (const v of series) if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!isFinite(lo)) return;
  const pad = (hi - lo) * 0.12 || 0.1; lo -= pad; hi += pad;
  const xs = scale([-6, 16], [padL, W - padR]);
  const ys = scale([lo, hi], [H - padB, padT]);
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    const gy = padT + g * (H - padB - padT);
    svgEl("line", { x1: padL, x2: W - padR, y1: gy, y2: gy, stroke: "#1A2940", "stroke-width": 0.6 }, svg);
    svgEl("text", { x: padL - 6, y: gy + 3, "text-anchor": "end", fill: "#7A8AA5", "font-size": 9,
      "font-family": "JetBrains Mono, monospace" }, svg).textContent = fmt(hi - g * (hi - lo), 2);
  }
  for (const a of [-6, -2, 2, 6, 10, 14, 16]) {
    svgEl("line", { x1: xs(a), x2: xs(a), y1: padT, y2: H - padB, stroke: "#1A2940", "stroke-width": 0.6 }, svg);
    svgEl("text", { x: xs(a), y: H - padB + 14, "text-anchor": "middle", fill: "#7A8AA5", "font-size": 9,
      "font-family": "JetBrains Mono, monospace" }, svg).textContent = a;
  }
  const path = series.map((v, i) => `${i ? "L" : "M"}${xs(-6 + (32 * i) / (series.length - 1)).toFixed(1)},${ys(v).toFixed(1)}`).join(" ");
  svgEl("path", { d: path, fill: "none", stroke: "#5AB0FF", "stroke-width": 1.6 }, svg);
  if (opts.marker != null && isFinite(opts.marker)) {
    const mx = xs(opts.marker);
    svgEl("line", { x1: mx, x2: mx, y1: padT, y2: H - padB, stroke: "#FFB454", "stroke-width": 1.2, "stroke-dasharray": "4 3" }, svg);
  }
  if (opts.maxMarker) {
    let bi = 0;
    for (let i = 1; i < series.length; i++) if (series[i] > series[bi]) bi = i;
    const bx = -6 + (32 * bi) / (series.length - 1);
    svgEl("circle", { cx: xs(bx), cy: ys(series[bi]), r: 3.5, fill: "#6FCF97" }, svg);
    svgEl("text", { x: xs(bx) + 6, y: ys(series[bi]) - 6, fill: "#6FCF97", "font-size": 9.5,
      "font-family": "JetBrains Mono, monospace" }, svg).textContent = `${fmt(series[bi], 2)} @ ${fmt(bx, 1)}°`;
  }
  svgEl("text", { x: W / 2, y: H - 2, "text-anchor": "middle", fill: "#7A8AA5", "font-size": 10 }, svg).textContent = "α (deg)";
  svgEl("text", { x: 8, y: padT + 2, fill: "#7A8AA5", "font-size": 10 }, svg).textContent = yLabel;
}

/* ---------- three.js hero: morphing wing + flow particles ---------- */
const hero = (() => {
  const canvas = $("wing-canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x0b1220, 0.10);
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);

  const key = new THREE.DirectionalLight(0xffffff, 2.4); key.position.set(2.5, 3, 4);
  const rim = new THREE.DirectionalLight(0x5ab0ff, 1.6); rim.position.set(-3, 1.2, -2.5);
  scene.add(key, rim, new THREE.AmbientLight(0xffffff, 0.38));

  const grid = new THREE.GridHelper(7, 28, 0x24344d, 0x1a2940);
  grid.position.y = -0.42;
  grid.material.transparent = true; grid.material.opacity = 0.35;
  scene.add(grid);

  // ---- morphable wing: rings of profile points along the span ----
  const PROF_N = 121, SPAN = 1.25, RINGS = 26;
  const ringZ = Array.from({ length: RINGS }, (_, i) => -SPAN / 2 + (SPAN * i) / (RINGS - 1));
  const taper = (z) => 1 - 0.24 * Math.pow(Math.abs(z) / (SPAN / 2), 1.6);

  const positions = new Float32Array(RINGS * PROF_N * 3 + 6);
  const indices = [];
  for (let r = 0; r < RINGS - 1; r++) {
    for (let i = 0; i < PROF_N; i++) {
      const a = r * PROF_N + i, b = r * PROF_N + ((i + 1) % PROF_N);
      const c = (r + 1) * PROF_N + i, d = (r + 1) * PROF_N + ((i + 1) % PROF_N);
      indices.push(a, c, b, b, c, d);
    }
  }
  const capL = RINGS * PROF_N, capR = RINGS * PROF_N + 1;
  for (let i = 0; i < PROF_N; i++) {
    indices.push(capL, i, (i + 1) % PROF_N);
    const base = (RINGS - 1) * PROF_N;
    indices.push(capR, base + ((i + 1) % PROF_N), base + i);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(indices);

  const material = new THREE.MeshStandardMaterial({
    color: 0x4a70a0, metalness: 0.5, roughness: 0.3, side: THREE.DoubleSide,
  });
  const wingGroup = new THREE.Group();
  wingGroup.add(new THREE.Mesh(geometry, material));
  scene.add(wingGroup);

  function profileRing(m, p, t) {
    // upper n+1 points + lower n points = 2n+1 closed ring, must equal PROF_N
    return nacaPoints(m, p, t, Math.round((PROF_N - 1) / 2)).ring;
  }
  const wingFrom = profileRing(state.m, state.p, state.t).map(([x, y]) => [x, y]);
  const wingTo = wingFrom.map(([x, y]) => [x, y]);

  function writeWing(points) {
    const pos = geometry.attributes.position.array;
    let vi = 0;
    for (let r = 0; r < RINGS; r++) {
      const z = ringZ[r], s = taper(z);
      for (let i = 0; i < PROF_N; i++) {
        pos[vi++] = points[i][0] * s - 0.5;
        pos[vi++] = points[i][1] * s;
        pos[vi++] = z;
      }
    }
    const mid = points[Math.floor(PROF_N / 2)];
    pos[capL * 3] = mid[0] * taper(-SPAN / 2) - 0.5;
    pos[capL * 3 + 1] = mid[1] * taper(-SPAN / 2);
    pos[capL * 3 + 2] = -SPAN / 2;
    pos[capR * 3] = mid[0] * taper(SPAN / 2) - 0.5;
    pos[capR * 3 + 1] = mid[1] * taper(SPAN / 2);
    pos[capR * 3 + 2] = SPAN / 2;
    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();
  }
  writeWing(wingFrom);

  // flow particles
  const PN = 550;
  const pGeo = new THREE.BufferGeometry();
  const pPos = new Float32Array(PN * 3);
  const seed = [];
  for (let i = 0; i < PN; i++) {
    seed.push({
      x: -2.8 + Math.random() * 6.0,
      y: (Math.random() * 2 - 1) * 0.85,
      z: (Math.random() * 2 - 1) * 0.75,
      v: 0.85 + Math.random() * 0.5,
    });
  }
  pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
  scene.add(new THREE.Points(pGeo, new THREE.PointsMaterial({
    color: 0x5ab0ff, size: 0.022, transparent: true, opacity: 0.5,
    blending: THREE.AdditiveBlending, depthWrite: false,
  })));

  // camera orbit
  const orbit = { yaw: 0.9, pitch: 0.3, radius: 3.1, dragging: false, lastX: 0, lastY: 0, idle: 0 };
  function applyCamera() {
    camera.position.set(
      orbit.radius * Math.cos(orbit.pitch) * Math.cos(orbit.yaw),
      orbit.radius * Math.sin(orbit.pitch) + 0.1,
      orbit.radius * Math.cos(orbit.pitch) * Math.sin(orbit.yaw));
    camera.lookAt(-0.1, 0, 0);
  }
  canvas.addEventListener("pointerdown", (e) => {
    orbit.dragging = true; orbit.lastX = e.clientX; orbit.lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!orbit.dragging) return;
    orbit.yaw += (e.clientX - orbit.lastX) * 0.006;
    orbit.pitch = THREE.MathUtils.clamp(orbit.pitch + (e.clientY - orbit.lastY) * 0.005, -0.5, 1.1);
    orbit.lastX = e.clientX; orbit.lastY = e.clientY;
  });
  canvas.addEventListener("pointerup", () => { orbit.dragging = false; orbit.idle = 0; });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    orbit.radius = THREE.MathUtils.clamp(orbit.radius + e.deltaY * 0.0022, 1.6, 7.5);
  }, { passive: false });

  let alphaRad = 0;
  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width < 4) return;
    renderer.setSize(rect.width, rect.height, false);
    camera.aspect = rect.width / Math.max(1, rect.height);
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  function update(dt) {
    let moved = false;
    const lim = Math.min(PROF_N, wingFrom.length, wingTo.length);
    for (let i = 0; i < lim; i++) {
      wingFrom[i][0] += (wingTo[i][0] - wingFrom[i][0]) * Math.min(1, dt * 7);
      wingFrom[i][1] += (wingTo[i][1] - wingFrom[i][1]) * Math.min(1, dt * 7);
      if (Math.abs(wingTo[i][0] - wingFrom[i][0]) > 1e-4) moved = true;
    }
    if (moved) writeWing(wingFrom);
    wingGroup.rotation.z = THREE.MathUtils.degToRad(-alphaRad);
    if (!orbit.dragging) { orbit.idle += dt; if (orbit.idle > 2.2) orbit.yaw += dt * 0.07; }
    applyCamera();
    for (let i = 0; i < PN; i++) {
      const s = seed[i];
      s.x += s.v * dt;
      if (s.x > 3.2) { s.x = -3.0; s.y = (Math.random() * 2 - 1) * 0.85; s.z = (Math.random() * 2 - 1) * 0.75; }
      let y = s.y;
      if (Math.abs(s.z) < SPAN / 2 + 0.05) {
        const sT = taper(s.z);
        const xi = THREE.MathUtils.clamp(s.x + 0.5, 0, 1);
        const prof = wingFrom[Math.floor(xi * (PROF_N - 1))];
        const surface = Math.abs(prof[1]) * sT + 0.02;
        if (Math.abs(y) < surface + 0.16) {
          y = (surface + 0.16) * Math.sign(y || 1) * (0.4 + 0.6 * Math.exp(-Math.abs(y) * 6));
        }
      }
      pPos[i * 3] = s.x - 0.1; pPos[i * 3 + 1] = y; pPos[i * 3 + 2] = s.z;
    }
    pGeo.attributes.position.needsUpdate = true;
    renderer.render(scene, camera);
    return moved;
  }

  return {
    update, resize,
    setWingTarget(m, p, t) {
      const nxt = profileRing(m, p, t);
      for (let i = 0; i < PROF_N; i++) wingTo[i] = nxt[i];
    },
    setAlpha(a) { alphaRad = a; },
  };
})();

/* ---------- master loop ---------- */
let lastT = performance.now();
let cpDirty = true, polarDirty = true, gaugeDirty = true;
function halfBand(lo, hi) {
  let s = 0, n = 0;
  for (let i = 0; i < lo.length; i++) { s += Math.abs((hi[i] ?? 0) - (lo[i] ?? 0)); n++; }
  return n ? s / (2 * n) : 0;
}
function loop(now) {
  const dt = Math.min(0.05, (now - lastT) / 1000 || 0.016);
  lastT = now;
  const k = 1 - Math.exp(-dt * 7);
  const moving = lerpInto(display, target, k);
  if (moving || cpDirty) {
    drawCp();
    $("ro-cl").textContent = fmt(display.cl, 4);
    $("ro-cd").textContent = fmtSci(display.cd, 4);
    $("ro-cm").textContent = fmt(display.cm, 4);
    $("ro-ld").textContent = fmt(display.ld, 2);
    $("band-cl").textContent = `± ${fmt((display.clCi[1] - display.clCi[0]) / 2, 3)}`;
    $("band-cd").textContent = `± ${fmtSci((display.cdCi[1] - display.cdCi[0]) / 2, 2)}`;
    $("band-cm").textContent = `± ${fmt((display.cmCi[1] - display.cmCi[0]) / 2, 3)}`;
    cpDirty = false;
  }
  if (moving || polarDirty) {
    miniChart("polar-cl", display.polarCl, "CL vs alpha", { marker: state.alpha });
    miniChart("polar-ld", display.polarLd, "L/D vs alpha", { maxMarker: true, marker: state.alpha });
    polarDirty = false;
  }
  if (moving || gaugeDirty) {
    drawGauge(display.mahal);
    $("uq-conf").textContent = `${fmt(display.conf, 1)}%`;
    $("uq-mahal").textContent = fmt(display.mahal, 3);
    gaugeDirty = false;
  }
  hero.update(dt);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

/* ---------- data flows ---------- */
let predictTimer = null, polarTimer = null;
function schedulePredict(delay = 80) { clearTimeout(predictTimer); predictTimer = setTimeout(runPredict, delay); }
function schedulePolar(delay = 420) { clearTimeout(polarTimer); polarTimer = setTimeout(runPolar, delay); }
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

function setPredictTargets(res) {
  const set = (key, arr, n) => {
    if (Array.isArray(arr) && arr.length === n) target[key] = arr.map(Number);
  };
  target.cl = Number(res.cl) || 0;
  target.cd = Number(res.cd) || 0;
  target.cm = Number(res.cm) || 0;
  target.ld = Number(res.lift_to_drag) || 0;
  const uq = res.uq || {};
  target.mahal = Number(uq.mahalanobis_dist) || 0;
  target.conf = Number(uq.confidence_score) || 0;
  target.clCi = (uq.cl_ci95 || [0, 0]).map(Number);
  target.cdCi = (uq.cd_ci95 || [0, 0]).map(Number);
  target.cmCi = (uq.cm_ci95 || [0, 0]).map(Number);
  set("cpU", res.cp_upper, CP_N);
  set("cpL", res.cp_lower, CP_N);
  set("bU", uq.cp_upper_ci95, CP_N);
  set("bUh", uq.cp_upper_ci95_high, CP_N);
  set("bL", uq.cp_lower_ci95, CP_N);
  set("bLh", uq.cp_lower_ci95_high, CP_N);
  if (firstFill) {
    for (const key of Object.keys(target)) {
      if (Array.isArray(target[key])) display[key] = target[key].slice();
      else display[key] = target[key];
    }
    firstFill = false;
    cpDirty = polarDirty = gaugeDirty = true;
  }
  const verdict = $("ood-verdict");
  const mahal = target.mahal;
  if (uq.is_ood) { verdict.textContent = "OUT OF DOMAIN"; verdict.className = "ood-verdict warn"; }
  else if (mahal > MAHAL_P99) { verdict.textContent = "NEAR BOUNDARY"; verdict.className = "ood-verdict caution"; }
  else { verdict.textContent = "IN DOMAIN"; verdict.className = "ood-verdict ok"; }
  const warnList = $("warn-list");
  warnList.innerHTML = "";
  const warnings = uq.warnings || [];
  if (!warnings.length) {
    const li = document.createElement("li"); li.className = "warn-empty";
    li.textContent = "All stations nominal."; warnList.append(li);
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
  drawWingbox(res);
}

async function runPredict() {
  try {
    const res = await post("/api/predict", {
      m: state.m, p: state.p, t: state.t,
      alpha_deg: state.alpha, reynolds: state.reynolds,
    });
    setPredictTargets(res);
  } catch (err) {
    const verdict = $("ood-verdict");
    verdict.textContent = "BACKEND ERROR"; verdict.className = "ood-verdict warn";
    const li = document.createElement("li"); li.className = "warn-level-high";
    li.textContent = String(err.message || err);
    $("warn-list").innerHTML = ""; $("warn-list").append(li);
  }
}

async function runPolar() {
  try {
    const res = await post("/api/polar", {
      m: state.m, p: state.p, t: state.t, reynolds: state.reynolds,
    });
    const rows = res.rows || [];
    if (rows.length === POLAR_N) {
      target.polarCl = rows.map((r) => (isFinite(r.cl) ? r.cl : 0));
      target.polarLd = rows.map((r) => (isFinite(r.lift_to_drag) ? r.lift_to_drag : 0));
      polarDirty = true;
    }
  } catch { /* keep last polar */ }
}

/* ---------- controls ---------- */
function syncLabels() {
  $("v-m").textContent = state.m.toFixed(3);
  $("v-p").textContent = state.p.toFixed(2);
  $("v-t").textContent = state.t.toFixed(3);
  $("v-alpha").textContent = `${state.alpha >= 0 ? "+" : ""}${state.alpha.toFixed(1)}°`;
  $("hero-alpha").textContent = `α ${state.alpha >= 0 ? "+" : ""}${state.alpha.toFixed(1)}°`;
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
      hero.setWingTarget(state.m, state.p, state.t);
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
      syncLabels();
      if (key === "alpha") { hero.setAlpha(state.alpha); schedulePredict(); }
      else { hero.setWingTarget(state.m, state.p, state.t); schedulePredict(); schedulePolar(); }
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
      const op = res.opt_params || {};
      $("s-result").innerHTML =
        `<div>success: <b style="color:${res.success ? "var(--ok)" : "var(--warn)"}">${res.success}</b></div>` +
        `<div>iterations: ${res.iterations ?? "—"} · ${fmt(res.elapsed_seconds, 2)} s</div>` +
        `<div class="opt-bad">constraints met: ${res.constraints_met ?? "—"}</div>` +
        `<div>optimum (m,p,t,α): ${arrStr(op)}</div>`;
      $("s-status").textContent = res.solver_message || "done";
    } catch (err) {
      $("s-result").textContent = String(err.message || err);
      $("s-status").textContent = "search failed";
    } finally { btn.disabled = false; }
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
  hero.setWingTarget(state.m, state.p, state.t);
  hero.setAlpha(state.alpha);
  await runPredict();
  runPolar();
}

boot();
