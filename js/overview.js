// Method overview: a schematic of PACE drawn natively in SVG, in four stages.
// Toy geometry (not data): three snapshots along a curved manifold.
(() => {
  'use strict';

  const svg = document.getElementById('ov');
  if (!svg) return;
  const fig = document.getElementById('overview');
  const cap = document.getElementById('ov-cap');
  const tabs = [...fig.querySelectorAll('[role="tab"]')];
  const cards = [...document.querySelectorAll('.step[data-stage]')];
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const NS = 'http://www.w3.org/2000/svg';

  const C = { t: ['#93a9cd', '#4f6fa6', '#10264a'], red: '#d1343f', ink: '#0f1d36', muted: '#8a92a0' };
  const STAGES = [
    { ms: 3600, cap: 'Unpaired snapshots: which cell became which is never observed.' },
    { ms: 4000, cap: 'Local PCA on each snapshot gives a tangent (cheap to move along) and a normal (costly).' },
    { ms: 6000, cap: 'Straight Euclidean pairings cut across the bend. PACE re-couples cells along bridges that follow the manifold.' },
    { ms: 5600, cap: 'The bridges are distilled into one velocity field and integrated as an ODE.' },
  ];

  // ---------- geometry ----------
  const P = [[50, 292], [540, 292], [220, 58], [710, 58]];
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
  const smooth = (t) => { t = clamp01(t); return t * t * (3 - 2 * t); };
  function bez(t) {
    const u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return [a * P[0][0] + b * P[1][0] + c * P[2][0] + d * P[3][0], a * P[0][1] + b * P[1][1] + c * P[2][1] + d * P[3][1]];
  }
  const M = 600, curve = [], len = [0];
  for (let i = 0; i <= M; i++) curve.push(bez(i / M));
  for (let i = 1; i <= M; i++) len.push(len[i - 1] + Math.hypot(curve[i][0] - curve[i - 1][0], curve[i][1] - curve[i - 1][1]));
  const L = len[M];

  // Point, unit tangent and unit normal at arc-length fraction s.
  function at(s) {
    const target = clamp01(s) * L;
    let lo = 0, hi = M - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (len[mid] <= target) lo = mid; else hi = mid - 1; }
    const a = curve[lo], b = curve[lo + 1], f = (target - len[lo]) / (len[lo + 1] - len[lo] || 1);
    const dx = b[0] - a[0], dy = b[1] - a[1], k = Math.hypot(dx, dy) || 1;
    const tx = dx / k, ty = dy / k;
    return { x: lerp(a[0], b[0], f), y: lerp(a[1], b[1], f), tx, ty, nx: ty, ny: -tx };
  }
  const pos = (s, n) => { const q = at(s); return [q.x + n * q.nx, q.y + n * q.ny]; };

  let seed = 11;
  const rng = () => { seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const RANGES = [[0.05, 0.22], [0.42, 0.58], [0.78, 0.95]];
  const N = 11;
  const cells = RANGES.map(([a, b], k) => Array.from({ length: N }, (_, i) => {
    const s = a + (b - a) * (i + 0.5 + (rng() - 0.5) * 0.7) / N;
    const n = (rng() * 2 - 1) * 10;
    const [x, y] = pos(s, n);
    return { s, n, k, x, y };
  }));

  // Couplings between adjacent snapshots: a scrambled start, and the order-preserving refinement.
  function couplings(A, B) {
    const scrambled = A.map((_, i) => i);
    for (let i = scrambled.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [scrambled[i], scrambled[j]] = [scrambled[j], scrambled[i]]; }
    const byA = A.map((c, i) => i).sort((i, j) => A[i].n - A[j].n);
    const byB = B.map((c, i) => i).sort((i, j) => B[i].n - B[j].n);
    const refined = []; byA.forEach((i, r) => { refined[i] = byB[r]; });
    return A.map((a, i) => ({ a, old: B[scrambled[i]], neu: B[refined[i]] }));
  }
  const pairs = [...couplings(cells[0], cells[1]), ...couplings(cells[1], cells[2])];

  // ---------- svg helpers ----------
  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }
  const f1 = (v) => v.toFixed(1);
  const pathD = (pts) => 'M' + pts.map((p) => `${f1(p[0])},${f1(p[1])}`).join('L');
  function label(parent, x, y, text, cls, anchor) {
    const t = el('text', { x: f1(x), y: f1(y), class: cls || '', 'text-anchor': anchor || 'start' }, parent);
    t.textContent = text;
    return t;
  }
  function tLabel(parent, x, y, sub) {
    const t = el('text', { x: f1(x), y: f1(y), class: 'tlabel', 'text-anchor': 'middle' }, parent);
    t.appendChild(document.createTextNode('T'));
    const s = el('tspan', { class: 'sub', dy: '0.32em' }, t);
    s.textContent = sub;
    return t;
  }

  const defs = el('defs', {}, svg);
  for (const [id, color] of [['ov-ink', C.ink], ['ov-muted', C.muted], ['ov-red', C.red]]) {
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: '8', refY: '5', markerWidth: '7', markerHeight: '7', orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,1 L9,5 L0,9 z', fill: color }, m);
  }

  // manifold band (support of the data)
  el('path', { d: pathD(curve.filter((_, i) => i % 6 === 0)), class: 'band' }, svg);

  // ---------- stage 0: unknown correspondences ----------
  const gWeb = el('g', { class: 'layer' }, svg);
  for (let k = 0; k < 2; k++) for (const a of cells[k]) for (const b of cells[k + 1]) {
    el('line', { x1: f1(a.x), y1: f1(a.y), x2: f1(b.x), y2: f1(b.y), class: 'web' }, gWeb);
  }
  for (const s of [0.33, 0.67]) {
    const [x, y] = pos(s, 46);
    label(gWeb, x, y, '?', 'qm', 'middle');
  }
  label(gWeb, 50, 40, 'correspondences unknown', 'note');

  // ---------- stage 1: anisotropic metric ----------
  const gMetric = el('g', { class: 'layer' }, svg);
  // The metric only exists where cells were observed: one local PCA per snapshot.
  cells.forEach((group, idx) => {
    const mx = group.reduce((a, c) => a + c.x, 0) / group.length;
    const my = group.reduce((a, c) => a + c.y, 0) / group.length;
    let sxx = 0, sxy = 0, syy = 0;
    for (const c of group) { const dx = c.x - mx, dy = c.y - my; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
    const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy); // leading principal axis
    const flow = at((RANGES[idx][0] + RANGES[idx][1]) / 2); // orient the axis along the time direction
    let tx = Math.cos(theta), ty = Math.sin(theta);
    if (tx * flow.tx + ty * flow.ty < 0) { tx = -tx; ty = -ty; }
    const nx = ty, ny = -tx;
    el('ellipse', { cx: f1(mx), cy: f1(my), rx: 40, ry: 11, transform: `rotate(${f1(Math.atan2(ty, tx) * 180 / Math.PI)} ${f1(mx)} ${f1(my)})`, class: 'glyph' }, gMetric);
    const tip = [mx + tx * 54, my + ty * 54], ntip = [mx + nx * 28, my + ny * 28];
    el('line', { x1: f1(mx), y1: f1(my), x2: f1(tip[0]), y2: f1(tip[1]), class: 'arrow-ink', 'marker-end': 'url(#ov-ink)' }, gMetric);
    el('line', { x1: f1(mx), y1: f1(my), x2: f1(ntip[0]), y2: f1(ntip[1]), class: 'arrow-muted', 'marker-end': 'url(#ov-muted)' }, gMetric);
    if (idx === 1) {
      label(gMetric, tip[0] + 12, tip[1] + 6, 'tangent · low cost', 'note ink minor');
      label(gMetric, ntip[0] - 8, ntip[1] - 4, 'normal · high cost', 'note minor', 'end');
    }
  });
  label(gMetric, 50, 40, 'G(x,t) = I + α·P_N', 'note');

  // ---------- stage 2: couplings and bridges ----------
  const gCouple = el('g', { class: 'layer' }, svg);
  const bridgeEls = pairs.map(() => el('path', { class: 'bridge' }, gCouple));
  const eucl = label(gCouple, 50, 40, 'Euclidean pairing · r = 0', 'note');
  const pace = label(gCouple, 50, 40, 'PACE bridges · r = 1, 2, …', 'note red');
  const TAU = 18;
  function drawCouplings(alpha) {
    const g = [lerp(138, 209, alpha), lerp(146, 52, alpha), lerp(160, 63, alpha)].map(Math.round);
    pairs.forEach((p, i) => {
      const pts = [];
      for (let j = 0; j <= TAU; j++) {
        const t = j / TAU;
        const straight = [lerp(p.a.x, p.old.x, t), lerp(p.a.y, p.old.y, t)];
        const bridge = pos(lerp(p.a.s, p.neu.s, t), lerp(p.a.n, p.neu.n, t));
        pts.push([lerp(straight[0], bridge[0], alpha), lerp(straight[1], bridge[1], alpha)]);
      }
      const e = bridgeEls[i];
      e.setAttribute('d', pathD(pts));
      e.setAttribute('stroke', `rgb(${g.join(',')})`);
      e.setAttribute('stroke-dasharray', alpha > 0.98 ? 'none' : `4 ${f1(4 * (1 - alpha) + 0.01)}`);
      e.setAttribute('stroke-opacity', f1(lerp(0.6, 0.85, alpha)));
    });
    eucl.setAttribute('opacity', f1(1 - smooth(alpha * 2)));
    pace.setAttribute('opacity', f1(smooth(alpha * 2 - 1)));
  }

  // ---------- stage 3: velocity field ----------
  const gField = el('g', { class: 'layer' }, svg);
  for (let gx = 34; gx <= 730; gx += 38) for (let gy = 26; gy <= 316; gy += 38) {
    let best = 0, bd = Infinity;
    for (let i = 0; i <= M; i += 4) { const d = (curve[i][0] - gx) ** 2 + (curve[i][1] - gy) ** 2; if (d < bd) { bd = d; best = i; } }
    const dist = Math.sqrt(bd);
    if (dist > 72) continue;
    const q = at(len[best] / L), dn = (gx - q.x) * q.nx + (gy - q.y) * q.ny;
    let vx = q.tx - (dn / 70) * q.nx, vy = q.ty - (dn / 70) * q.ny;
    const k = Math.hypot(vx, vy); vx /= k; vy /= k;
    const h = 7;
    el('line', { x1: f1(gx - vx * h), y1: f1(gy - vy * h), x2: f1(gx + vx * h), y2: f1(gy + vy * h), class: 'field', 'stroke-opacity': f1(0.12 + 0.5 * (1 - dist / 72)), 'marker-end': 'url(#ov-muted)' }, gField);
  }
  const flows = cells[0].map(() => ({ trail: el('path', { class: 'trail' }, gField), head: el('circle', { r: 4.2, class: 'head' }, gField) }));
  label(gField, 50, 40, 'v_φ(x, t) · ODE rollout', 'note red');
  const TRAVEL = 0.73;
  function drawFlow(u) {
    cells[0].forEach((c, i) => {
      const pts = [];
      const steps = Math.max(1, Math.round(u * 40));
      for (let j = 0; j <= steps; j++) {
        const t = (j / steps) * u;
        pts.push(pos(c.s + TRAVEL * t, c.n * (1 - 0.3 * t)));
      }
      flows[i].trail.setAttribute('d', pathD(pts));
      const h = pts[pts.length - 1];
      flows[i].head.setAttribute('cx', f1(h[0])); flows[i].head.setAttribute('cy', f1(h[1]));
    });
  }

  // ---------- cells (always on top) ----------
  const gCells = el('g', {}, svg);
  cells.forEach((group, k) => {
    for (const c of group) el('circle', { cx: f1(c.x), cy: f1(c.y), r: 5, fill: C.t[k], class: 'cell' }, gCells);
    const mid = at((RANGES[k][0] + RANGES[k][1]) / 2);
    const out = k === 2 ? -1 : 1; // labels sit below the curve, except on the top arm
    tLabel(gCells, mid.x - mid.nx * 40 * out, mid.y - mid.ny * 40 * out + (k === 2 ? -6 : 8), ['k', 'k+1', 'k+2'][k]);
  });

  // ---------- playback ----------
  const layers = [gWeb, gMetric, gCouple, gField];
  let stage = 0, start = 0, auto = !reduceMotion, visible = false;

  function show(i, now) {
    stage = i; start = now;
    layers.forEach((g, k) => g.classList.toggle('on', k === i));
    tabs.forEach((t, k) => { t.setAttribute('aria-selected', String(k === i)); t.tabIndex = k === i ? 0 : -1; if (k !== i) t.style.setProperty('--p', 0); });
    cards.forEach((c) => c.classList.toggle('active', Number(c.dataset.stage) === i));
    cap.textContent = STAGES[i].cap;
    gCells.classList.toggle('dim', i === 1);
  }

  function frame(now) {
    const t = now - start;
    if (stage === 2) drawCouplings(reduceMotion ? 1 : smooth((t - 1200) / 2200));
    if (stage === 3) drawFlow(reduceMotion ? 1 : smooth(((t - 400) % 5600) / 3800));
    if (auto) {
      const p = clamp01(t / STAGES[stage].ms);
      tabs[stage].style.setProperty('--p', p.toFixed(3));
      if (p >= 1) show((stage + 1) % STAGES.length, now);
    }
  }

  function loop(now) {
    if (visible) frame(now);
    requestAnimationFrame(loop);
  }

  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => { auto = false; show(i, performance.now()); tab.style.setProperty('--p', 1); });
    tab.addEventListener('keydown', (e) => {
      const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (!d) return;
      e.preventDefault();
      const next = (stage + d + STAGES.length) % STAGES.length;
      tabs[next].focus(); tabs[next].click();
    });
  });

  new IntersectionObserver(([e]) => {
    const was = visible; visible = e.isIntersecting;
    if (visible && !was) start = performance.now(); // restart the current stage when scrolled into view
  }, { threshold: 0.25 }).observe(fig);

  drawCouplings(0);
  drawFlow(0);
  show(0, performance.now());
  if (reduceMotion) { drawCouplings(1); drawFlow(1); tabs.forEach((t) => t.style.setProperty('--p', 1)); }
  requestAnimationFrame(loop);
})();
