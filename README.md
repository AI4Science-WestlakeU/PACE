# PACE project page

Static site for *PACE: Geometry-Aware Bridge Transport for Single-Cell Trajectory Inference* (NeurIPS 2026).
No build step: plain HTML, CSS and JavaScript. Kept outside the PACE code repository on purpose.

```
index.html          page
css/site.css        styles
js/hero.js          hero animation (renders data/wot.json)
js/results.js       results chart (numbers transcribed from the paper's tables)
js/page.js          KaTeX, scroll reveals, BibTeX copy
data/wot.json       PACE rollouts on WOT reprogramming, quantized (generated)
assets/             logo variants, favicon, Figure 1 (generated)
tools/              scripts that regenerate data/ and assets/ from the PACE repo
```

## Preview

```bash
python3 -m http.server 8765 --directory .
```

Open http://localhost:8765. Opening `index.html` as a file will not load `data/wot.json`.

## Regenerate data and assets

Both scripts read the PACE repository and never write to it.

```bash
PACE=/path/to/PACE
PYTHONDONTWRITEBYTECODE=1 $PACE/.venv/bin/python tools/export_rollouts.py --pace $PACE
PYTHONDONTWRITEBYTECODE=1 $PACE/.venv/bin/python tools/make_assets.py --pace $PACE
```

`export_rollouts.py` needs the cached demo rollouts at
`results/reprogramming_demo/.../metrics/readme_reprogramming_rollouts.npz`
(produced by `scripts/render_reprogramming_demo.py` in the PACE repo).
`make_assets.py` needs `cwebp`.

## What the hero shows

Real PACE predictions, not a simulation: the README demo model on Schiebinger2019 (WOT) in its FLE
embedding, integrated from observed snapshots 5, 10 and 15 to 7, 12 and 17. Snapshots 6, 11 and 16
are held out and drawn as rings. Background dots are 1,400 seeded cells per snapshot, coloured by day.
That demo model was selected with holdout feedback (see `docs/development-demo.md` in the PACE repo),
so it illustrates the method rather than reproducing the paper's scores.
