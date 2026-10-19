// Results chart: time-averaged held-out reconstruction errors from the paper's tables.
(() => {
  'use strict';

  const METRICS = ['MMD', 'W₁', 'W₂'];
  const OURS = 'PACE';
  // [MMD, W1, W2]; null = not reported. Ocean and Two-Branch are means of the per-snapshot tables.
  const SETTINGS = [
    { name: 'Ocean', sub: 'Ocean currents · 2D positions · held out t = 1, 3, 5, 7', v: {
      'PACE': [0.2651, 0.0393, 0.0442], 'Action Matching': [0.9711, 0.3957, 0.4019], 'Aligned CFM': [0.8772, 0.1972, 0.1999],
      'CURLY': [0.8362, 0.2251, 0.2264], 'DMSB': [0.6213, 0.1422, 0.1457], 'MFM': [0.7273, 0.1258, 0.1283], 'OT-CFM': [0.7678, 0.1316, 0.1339] } },
    { name: 'Two-Branch', sub: 'Branching toy · 2D · held out t = 1, 4', v: {
      'PACE': [0.1057, 0.1569, 0.1961], 'Action Matching': [0.4517, 0.3037, 0.3234], 'Aligned CFM': [0.4981, 0.3482, 0.3658],
      'CURLY': [0.1716, 0.1750, 0.2105], 'DMSB': [1.0909, 3.7095, 3.7126], 'MFM': [0.2481, 0.1775, 0.1974], 'OT-CFM': [0.3261, 0.2137, 0.2314] } },
    { name: 'EB PHATE', sub: 'Embryoid bodies · 2D PHATE · held out t = 3', v: {
      'PACE': [0.0674, 0.2087, 0.2803], 'Action Matching': [0.2436, 0.4553, 0.5273], 'Aligned CFM': [0.1056, 0.2657, 0.3243],
      'CURLY': [0.1384, 0.3223, 0.4422], 'DMSB': [0.1325, 0.3053, 0.4341], 'MFM': [0.1518, 0.3704, 0.4451], 'OT-CFM': [0.1069, 0.2784, 0.3622] } },
    { name: 'Schiebinger2019', sub: 'Mouse reprogramming · 2D FLE · held out t = 6, 11, 16', v: {
      'PACE': [0.2084, 0.2067, 0.2840], 'Action Matching': [0.4057, 0.4161, 0.4754], 'Aligned CFM': [0.3239, 0.3175, 0.3931],
      'CURLY': [0.7687, 1.3684, 1.4660], 'DMSB': [0.3030, 0.2774, 0.3662], 'MFM': [0.6990, 1.5897, 1.7942], 'OT-CFM': [0.8072, 1.9095, 2.0235] } },
    { name: 'iPSC-Liu 2D', sub: 'Human iPSC reprogramming · 2D PCA · held out t = 4, 16', v: {
      'PACE': [0.4385, 0.4853, 0.6238], 'Action Matching': [0.5230, 0.8005, 0.9730], 'Aligned CFM': [0.5428, 0.6594, 0.8320],
      'CURLY': [0.5613, 0.7186, 0.9216], 'DMSB': [0.6456, 1.0244, 1.1023], 'MFM': [0.5533, 0.6657, 0.8838], 'OT-CFM': [0.5228, 0.7166, 0.9261] } },
    { name: 'iPSC-Liu 10D', sub: 'Human iPSC reprogramming · 10D PCA · held out t = 4, 8, 12, 16', v: {
      'PACE': [0.4095, 2.9226, 3.1879], 'Action Matching': [0.5555, 3.7801, 4.3743], 'Aligned CFM': [0.4644, 2.9405, 3.3825],
      'CURLY': [0.5369, 3.6128, 4.2102], 'DMSB': [0.7713, 6.3088, 6.4593], 'MFM': [0.5273, 3.4066, 3.8382], 'OT-CFM': [0.5145, 3.3850, 3.9447] } },
    { name: 'iPSC-Liu 50D', sub: 'Human iPSC reprogramming · 50D PCA · held out t = 4, 8, 12, 16', v: {
      'PACE': [0.2550, 6.7537, 7.2836], 'Action Matching': [0.3125, 7.6126, 8.4868], 'Aligned CFM': [0.2624, 7.3133, 8.0552],
      'CURLY': [0.3144, 8.1144, 8.9096], 'DMSB': [0.5638, 10.9368, 11.2423], 'MFM': [0.2853, 7.7924, 8.5454], 'OT-CFM': [0.2898, 7.8392, 8.5954] } },
    { name: 'OP-Cite 100D', sub: 'Open Problems CITE-seq · 100D PCA · held out t = 3', v: {
      'PACE': [0.1409, 10.4574, 10.5605], 'Action Matching': [0.1753, 9.9388, 10.0448], 'Aligned CFM': [0.1434, 10.7009, 10.8094],
      'CURLY': [0.2529, 11.5163, 11.6015], 'DMSB': null, 'MFM': [0.1435, 10.4301, 10.5409], 'OT-CFM': [0.1448, 10.4447, 10.5491] } },
    { name: 'OP-Multi 100D', sub: 'Open Problems Multiome · 100D PCA · held out t = 3', v: {
      'PACE': [0.1693, 10.5824, 10.6293], 'Action Matching': [0.1622, 10.7711, 10.8160], 'Aligned CFM': [0.1720, 10.8330, 10.8826],
      'CURLY': [0.2786, 12.0015, 12.0440], 'DMSB': null, 'MFM': [0.1699, 10.6274, 10.6702], 'OT-CFM': [0.1662, 10.6128, 10.6564] } },
  ];

  const ROW_H = 40;
  const bars = document.getElementById('bars');
  const delta = document.getElementById('delta');
  const sub = document.getElementById('chart-sub');
  const tip = document.getElementById('tip');
  const dsBox = document.getElementById('datasets');
  const mBox = document.getElementById('metrics');
  const methods = Object.keys(SETTINGS[0].v);
  let ds = 0, metric = 1;

  const fmt = (x) => (x >= 10 ? x.toFixed(2) : x.toFixed(4));

  function seg(box, labels, onPick, active) {
    labels.forEach((label, i) => {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = label; b.setAttribute('aria-pressed', String(i === active));
      b.addEventListener('click', () => {
        [...box.children].forEach((c) => c.setAttribute('aria-pressed', String(c === b)));
        onPick(i);
      });
      box.appendChild(b);
    });
  }

  const rows = new Map(methods.map((m) => {
    const row = document.createElement('div');
    row.className = 'row' + (m === OURS ? ' ours' : '');
    row.setAttribute('role', 'row');
    row.innerHTML = `<span class="name" role="rowheader">${m}</span><span class="lane"><span class="bar"></span></span><span class="val" role="cell"></span>`;
    row.addEventListener('pointermove', (e) => showTip(e, m));
    row.addEventListener('pointerleave', () => tip.classList.remove('on'));
    bars.appendChild(row);
    return [m, row];
  }));
  bars.style.height = `${methods.length * ROW_H}px`;

  function showTip(e, m) {
    const s = SETTINGS[ds], x = s.v[m] ? s.v[m][metric] : null;
    if (x == null) { tip.classList.remove('on'); return; }
    const ours = s.v[OURS][metric];
    const rel = m === OURS ? '' : `<br><span>vs PACE</span> ${x > ours ? '+' : ''}${((x / ours - 1) * 100).toFixed(1)}%`;
    tip.innerHTML = `${m}<br><span>${METRICS[metric]}</span> ${fmt(x)}${rel}`;
    tip.style.left = `${e.clientX}px`; tip.style.top = `${e.clientY}px`;
    tip.classList.add('on');
  }

  function render() {
    const s = SETTINGS[ds];
    sub.textContent = s.sub;
    const entries = methods.map((m) => [m, s.v[m] ? s.v[m][metric] : null]);
    const known = entries.filter(([, x]) => x != null).map(([, x]) => x).sort((a, b) => a - b);
    // Clip a lone outlier so the remaining bars stay readable; it is marked with a break.
    const top = known[known.length - 1], second = known[known.length - 2];
    const domain = top > 2.2 * second ? second * 1.18 : top;
    const order = entries.slice().sort((a, b) => (a[1] ?? Infinity) - (b[1] ?? Infinity));

    order.forEach(([m, x], rank) => {
      const row = rows.get(m);
      row.style.top = `${rank * ROW_H}px`;
      row.classList.toggle('na', x == null);
      const bar = row.querySelector('.bar');
      bar.style.width = x == null ? '0%' : `${Math.min(x / domain, 1) * 100}%`;
      bar.classList.toggle('clipped', x != null && x > domain);
      row.querySelector('.val').textContent = x == null ? 'n/r' : fmt(x);
    });

    const ours = s.v[OURS][metric];
    const best = entries.filter(([m, x]) => m !== OURS && x != null).sort((a, b) => a[1] - b[1])[0];
    const change = (ours / best[1] - 1) * 100;
    const better = change < 0;
    delta.innerHTML = `<span>PACE vs best baseline (${best[0]})</span><strong class="${better ? '' : 'up'}">${change > 0 ? '+' : '−'}${Math.abs(change).toFixed(1)}%</strong><span>${better ? 'lower error' : 'higher error'}</span>`;
  }

  seg(dsBox, SETTINGS.map((s) => s.name), (i) => { ds = i; render(); }, ds);
  seg(mBox, METRICS, (i) => { metric = i; render(); }, metric);
  render();
})();
