// Hero: renders the cached PACE rollouts on WOT reprogramming (data/wot.json).
// Every particle is a real model prediction; nothing here is simulated.
(() => {
  'use strict';

  const canvas = document.getElementById('flow');
  const stage = document.getElementById('stage');
  const hero = canvas.parentElement;
  const track = document.getElementById('track');
  const fill = track.querySelector('.fill');
  const head = track.querySelector('.head');
  const readout = document.getElementById('readout');
  const playBtn = document.getElementById('play');
  const chips = [...document.querySelectorAll('.chip')];
  const ctx = canvas.getContext('2d');
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const PLAY_MS = 8000, HOLD_MS = 1600, FADE_MS = 700;
  const DUST_LO = [40, 68, 120], DUST_HI = [170, 192, 228];
  const TRAIL = 'rgba(240, 82, 92, ALPHA)';

  let data = null, dpr = 1, W = 0, H = 0;
  let fit = { x: 0, y: 0, s: 1 }, cam = { x: 0, y: 0, s: 1 }, target = null;
  let focus = -1, u = 0, playing = !reduceMotion, visible = true;
  let clock = 0, last = 0, dustKey = '';
  const dust = document.createElement('canvas');
  const dctx = dust.getContext('2d');

  const u16 = (b64) => { const s = atob(b64), b = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i); return new Uint16Array(b.buffer); };
  const u8 = (b64) => { const s = atob(b64), b = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i); return b; };
  const toWorld = (q) => { const f = new Float32Array(q.length); for (let i = 0; i < q.length; i++) f[i] = q[i] / 65535; return f; };
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (t) => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

  function decode(raw) {
    const ctxQ = toWorld(u16(raw.context.q));
    const day = u8(raw.context.day);
    const nDays = raw.days.length;
    const byDay = raw.days.map(() => []);
    for (let i = 0; i < day.length; i++) byDay[raw.days.indexOf(day[i])].push(ctxQ[2 * i], ctxQ[2 * i + 1]);
    const colors = raw.days.map((_, k) => {
      const t = k / (nDays - 1);
      return `rgb(${DUST_LO.map((c, j) => Math.round(lerp(c, DUST_HI[j], t))).join(',')})`;
    });
    const intervals = raw.intervals.map((iv) => {
      const path = toWorld(u16(iv.q));
      const held = toWorld(u16(raw.held[String(iv.held)]));
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const arr of [path, held]) for (let i = 0; i < arr.length; i += 2) {
        x0 = Math.min(x0, arr[i]); x1 = Math.max(x1, arr[i]); y0 = Math.min(y0, arr[i + 1]); y1 = Math.max(y1, arr[i + 1]);
      }
      return { ...iv, path, held, box: [x0, y0, x1, y1] };
    });
    return { extent: raw.extent, byDay: byDay.map((a) => Float32Array.from(a)), colors, intervals };
  }

  // ---------- camera ----------
  function stageRect() {
    const c = canvas.getBoundingClientRect(), s = stage.getBoundingClientRect();
    return { x: s.left - c.left, y: s.top - c.top, w: s.width, h: s.height };
  }
  function camFor(box, r, pad) {
    const bw = Math.max(box[2] - box[0], 1e-3), bh = Math.max(box[3] - box[1], 1e-3);
    return { x: (box[0] + box[2]) / 2, y: (box[1] + box[3]) / 2, s: Math.min(r.w / bw, r.h / bh) * pad };
  }
  // Resizes snap the camera; interval chips (animate = true) glide to the new view.
  function layout(animate = false) {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = hero.clientWidth; H = hero.clientHeight;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    dust.width = canvas.width; dust.height = canvas.height;
    if (!data || !W || !H) return;
    const r = stageRect();
    fit = camFor([0, 0, data.extent[0], data.extent[1]], r, 0.96);
    fit.cx = r.x + r.w / 2; fit.cy = r.y + r.h / 2;
    target = focus < 0 ? { ...fit } : { ...camFor(data.intervals[focus].box, r, 0.82), cx: fit.cx, cy: fit.cy };
    if (!cam.cx || animate !== true) cam = { ...target };
    cam.cx = fit.cx; cam.cy = fit.cy;
    dustKey = '';
  }
  const sx = (x) => (cam.cx + (x - cam.x) * cam.s) * dpr;
  const sy = (y) => (cam.cy - (y - cam.y) * cam.s) * dpr;

  // ---------- drawing ----------
  function drawDust() {
    const key = [cam.x, cam.y, cam.s, W, H, focus].map((v) => v.toFixed(4)).join();
    if (key === dustKey) return;
    dustKey = key;
    dctx.clearRect(0, 0, dust.width, dust.height);
    const zoom = cam.s / fit.s;
    const size = Math.min(1.1 + 0.45 * Math.log2(Math.max(zoom, 1)), 2.4) * dpr;
    dctx.globalAlpha = 0.45;
    data.byDay.forEach((pts, k) => {
      dctx.fillStyle = data.colors[k];
      for (let i = 0; i < pts.length; i += 2) dctx.fillRect(sx(pts[i]) - size / 2, sy(pts[i + 1]) - size / 2, size, size);
    });
    dctx.globalAlpha = 1;
  }

  // Screen positions of every cell at fractional frame f, written into iv.head.
  function heads(iv, f) {
    const F = iv.frames, N = iv.cells, p = iv.path, h = iv.head || (iv.head = new Float32Array(N * 2));
    const a = Math.min(Math.floor(f), F - 1), b = Math.min(a + 1, F - 1), t = f - a;
    for (let c = 0; c < N; c++) {
      const ia = (a * N + c) * 2, ib = (b * N + c) * 2;
      h[2 * c] = sx(lerp(p[ia], p[ib], t));
      h[2 * c + 1] = sy(lerp(p[ia + 1], p[ib + 1], t));
    }
    return h;
  }

  function draw(fade) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawDust();
    ctx.drawImage(dust, 0, 0);
    const bump = Math.exp(-Math.pow((u - 0.5) / 0.14, 2));
    // Small screens pack the same cells into fewer pixels: shrink marks and thin the glow.
    const markScale = Math.min(Math.max(fit.s / 680, 0.5), 1.25), dens = Math.sqrt(markScale);

    data.intervals.forEach((iv, idx) => {
      const dim = focus < 0 || focus === idx ? 1 : 0.12;
      const N = iv.cells, F = iv.frames, p = iv.path, f = u * (F - 1);

      // held-out cells: hollow rings that light up as the flow crosses them
      ctx.lineWidth = 1 * dpr;
      ctx.strokeStyle = `rgba(222, 232, 247, ${(0.16 + 0.5 * bump) * dim})`;
      const r = (focus === idx ? 2.6 : 1.9) * dpr;
      ctx.beginPath();
      for (let i = 0; i < iv.held.length; i += 2) {
        const x = sx(iv.held[i]), y = sy(iv.held[i + 1]);
        ctx.moveTo(x + r, y); ctx.arc(x, y, r, 0, Math.PI * 2);
      }
      ctx.stroke();

      if (fade <= 0) return;
      const h = heads(iv, f), whole = Math.floor(f);
      ctx.globalCompositeOperation = 'lighter';

      // accumulated paths
      ctx.lineWidth = 0.8 * dpr;
      ctx.strokeStyle = TRAIL.replace('ALPHA', (0.11 * dens * dim * fade).toFixed(3));
      ctx.beginPath();
      for (let c = 0; c < N; c++) {
        ctx.moveTo(sx(p[c * 2]), sy(p[c * 2 + 1]));
        for (let j = 1; j <= whole; j++) { const o = (j * N + c) * 2; ctx.lineTo(sx(p[o]), sy(p[o + 1])); }
        ctx.lineTo(h[2 * c], h[2 * c + 1]);
      }
      ctx.stroke();

      // recent tail
      const from = Math.max(0, whole - 4);
      ctx.lineWidth = 1.2 * dpr;
      ctx.strokeStyle = `rgba(255, 128, 136, ${(0.32 * dens * dim * fade).toFixed(3)})`;
      ctx.beginPath();
      for (let c = 0; c < N; c++) {
        const o0 = (from * N + c) * 2;
        ctx.moveTo(sx(p[o0]), sy(p[o0 + 1]));
        for (let j = from + 1; j <= whole; j++) { const o = (j * N + c) * 2; ctx.lineTo(sx(p[o]), sy(p[o + 1])); }
        ctx.lineTo(h[2 * c], h[2 * c + 1]);
      }
      ctx.stroke();

      // heads: soft halo + bright core
      const halo = 4.5 * markScale * dpr, core = 1.7 * Math.max(markScale, 0.8) * dpr;
      ctx.fillStyle = `rgba(240, 82, 92, ${(0.16 * dens * dim * fade).toFixed(3)})`;
      ctx.beginPath();
      for (let c = 0; c < N; c++) { const x = h[2 * c], y = h[2 * c + 1]; ctx.moveTo(x + halo, y); ctx.arc(x, y, halo, 0, Math.PI * 2); }
      ctx.fill();
      ctx.fillStyle = `rgba(255, 236, 237, ${(0.92 * dim * fade).toFixed(3)})`;
      for (let c = 0; c < N; c++) ctx.fillRect(h[2 * c] - core / 2, h[2 * c + 1] - core / 2, core, core);
      ctx.globalCompositeOperation = 'source-over';
    });
  }

  let hudText = '';
  function hud() {
    const pct = (u * 100).toFixed(1) + '%';
    fill.style.width = pct; head.style.left = pct;
    const show = focus < 0 ? data.intervals : [data.intervals[focus]];
    const days = show.map((iv) => (iv.start + u * (iv.end - iv.start)).toFixed(2)).join(' · ');
    const atHeld = Math.abs(u - 0.5) < 0.02;
    const text = `${days}|${atHeld}`;
    if (text === hudText) return;
    hudText = text;
    // The tag is always rendered and only toggles visibility, so the readout never changes width.
    readout.innerHTML = `day ${days} <small style="visibility:${atHeld ? 'visible' : 'hidden'}">held out</small>`;
    track.setAttribute('aria-valuenow', String(Math.round(u * 100)));
    track.setAttribute('aria-valuetext', `day ${days}${atHeld ? ', held-out snapshot' : ''}`);
  }

  // ---------- loop ----------
  let drawnKey = '';
  function tick(now) {
    requestAnimationFrame(tick);
    if (!visible || !data || !canvas.width || !canvas.height) { last = now; return; }
    const dt = Math.min(now - (last || now), 64); last = now;
    let fade = 1;
    if (playing) {
      clock = (clock + dt) % (PLAY_MS + HOLD_MS + FADE_MS);
      u = Math.min(clock / PLAY_MS, 1);
      if (clock > PLAY_MS + HOLD_MS) fade = 1 - (clock - PLAY_MS - HOLD_MS) / FADE_MS;
    }
    if (target) {
      const k = reduceMotion ? 1 : 1 - Math.pow(0.0025, dt / 1000);
      cam.x = lerp(cam.x, target.x, k); cam.y = lerp(cam.y, target.y, k); cam.s = lerp(cam.s, target.s, k);
      if (Math.abs(cam.s - target.s) / target.s < 1e-3 && Math.abs(cam.x - target.x) < 1e-5) { cam.x = target.x; cam.y = target.y; cam.s = target.s; }
    }
    // Skip the redraw when nothing visible changed (paused, reduced motion, settled camera).
    const key = [u, fade, cam.x, cam.y, cam.s, canvas.width, canvas.height, focus].join();
    if (key !== drawnKey) { drawnKey = key; draw(ease(Math.max(fade, 0))); }
    hud();
  }

  // ---------- controls ----------
  function setPlaying(on) {
    playing = on;
    playBtn.setAttribute('aria-label', on ? 'Pause' : 'Play');
    playBtn.innerHTML = on
      ? '<svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor"><rect x="1" y="0" width="3" height="10"/><rect x="6" y="0" width="3" height="10"/></svg>'
      : '<svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor"><path d="M1.5 0v10L9.5 5z"/></svg>';
    if (on) clock = u * PLAY_MS;
  }
  playBtn.addEventListener('click', () => setPlaying(!playing));

  chips.forEach((chip) => chip.addEventListener('click', () => {
    focus = Number(chip.dataset.focus);
    chips.forEach((c) => c.setAttribute('aria-pressed', String(c === chip)));
    layout(true);
  }));

  function scrub(e) {
    const r = track.getBoundingClientRect();
    u = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
    if (playing) setPlaying(false);
  }
  track.addEventListener('pointerdown', (e) => { track.setPointerCapture(e.pointerId); scrub(e); });
  track.addEventListener('pointermove', (e) => { if (track.hasPointerCapture(e.pointerId)) scrub(e); });
  const KEYS = { ArrowRight: 0.02, ArrowUp: 0.02, ArrowLeft: -0.02, ArrowDown: -0.02, PageUp: 0.1, PageDown: -0.1 };
  track.addEventListener('keydown', (e) => {
    let next;
    if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = 1;
    else if (e.key in KEYS) next = u + KEYS[e.key];
    else return;
    e.preventDefault(); u = Math.min(Math.max(next, 0), 1); if (playing) setPlaying(false);
  });

  new ResizeObserver(() => layout()).observe(hero);
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(hero);

  fetch('data/wot.json')
    .then((r) => { if (!r.ok) throw new Error(`data/wot.json: HTTP ${r.status}`); return r.json(); })
    .then((raw) => {
      data = decode(raw);
      if (reduceMotion) { u = 1; setPlaying(false); }
      layout();
      requestAnimationFrame((t) => { last = t; canvas.classList.add('ready'); tick(t); });
    })
    .catch((err) => { console.error('PACE hero:', err); readout.textContent = ''; });
})();
