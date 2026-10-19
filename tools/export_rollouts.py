#!/usr/bin/env python3
"""Export the cached PACE WOT rollouts into a compact file for the project page.

Reads (never writes) the PACE repository:
  <pace>/results/reprogramming_demo/.../pace/metrics/readme_reprogramming_rollouts.npz
  <pace>/data/scnode/wot_fle/wot_fle.h5ad

Writes data/wot.json next to this tool. All coordinates are in the run's
training-only standardized FLE space, the same space the rollouts live in.
Paths and cells are quantized to uint16 over a shared bounding box and stored
as base64 so the browser can decode them into typed arrays.

Usage:
  PYTHONDONTWRITEBYTECODE=1 <pace>/.venv/bin/python tools/export_rollouts.py --pace <pace>
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

RUN = 'results/reprogramming_demo/results/wot_fle_three_interpolation_dim2_test6-11-16/pace'
INTERVALS = ((5, 6, 7), (10, 11, 12), (15, 16, 17))
TIME_STRIDE = 4          # 201 Euler points -> 51 stored frames
CONTEXT_PER_DAY = 1400   # background cells per snapshot
HELD_PER_DAY = 512
Q = 65535


def b64(values: np.ndarray, dtype: npt.DTypeLike) -> str:
    """Encode a flat numeric array as base64 little-endian bytes."""
    return base64.b64encode(np.ascontiguousarray(values, dtype=dtype).tobytes()).decode('ascii')


def quantize(points: np.ndarray, lo: np.ndarray, span: float) -> np.ndarray:
    """Map 2D points into [0, Q] with one isotropic scale so the aspect ratio is kept."""
    scaled = np.rint((points - lo) / span * Q)
    if scaled.min() < 0 or scaled.max() > Q:
        raise ValueError('points fall outside the export bounds; refusing to clip them silently')
    return scaled.astype('<u2')


def load(pace: Path) -> tuple[dict, dict[int, np.ndarray], list[np.ndarray]]:
    """Rebuild the run's datamodule to recover its scaler, then load all cells and rollouts."""
    sys.path.insert(0, str(pace))
    from omegaconf import OmegaConf
    from src.methods.mfm.runner import build_datamodule

    run = pace / RUN
    cfg = OmegaConf.load(run / 'resolved_config.yaml')
    cfg.data_path = str(pace / 'data/scnode/wot_fle/wot_fle.h5ad')
    dm = build_datamodule(cfg)
    dm.setup()
    raw = dm._standardize_frames(dm._load_timepoint_frames())
    cells = {int(round(float(t))): dm.scaler.transform(v).astype(np.float32) for t, v in raw.items()}
    cached = np.load(run / 'metrics/readme_reprogramming_rollouts.npz')
    paths = [cached[f'interval_{s}_{e}'] for s, _, e in INTERVALS]
    meta = {'seed': int(cfg.get('seed', 42)), 'run': RUN}
    return meta, cells, paths


def main() -> None:
    """Export the rollouts and a seeded sample of every snapshot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pace', type=Path, required=True, help='PACE repository root (read-only)')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'data/wot.json')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    meta, cells, paths = load(args.pace.resolve())
    rng = np.random.default_rng(0)
    days = sorted(cells)
    context_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for day in days:
        frame = cells[day]
        keep = rng.choice(len(frame), size=min(CONTEXT_PER_DAY, len(frame)), replace=False)
        context_parts.append(frame[keep])
        label_parts.append(np.full(len(keep), day, dtype=np.uint8))
    context = np.concatenate(context_parts)
    labels = np.concatenate(label_parts)

    held_cells = {}
    for _, h, _ in INTERVALS:
        frame = cells[h]
        keep = rng.choice(len(frame), size=min(HELD_PER_DAY, len(frame)), replace=False)
        held_cells[h] = frame[keep]

    # Shared bounds: 0.05-99.95th percentile of the background cells (trims stray dots),
    # widened so every rollout point and every held-out cell fits without clipping.
    must_fit = np.concatenate([p.reshape(-1, 2) for p in paths] + list(held_cells.values()))
    lo = np.minimum(np.percentile(context, 0.05, axis=0), must_fit.min(0))
    hi = np.maximum(np.percentile(context, 99.95, axis=0), must_fit.max(0))
    span = float((hi - lo).max())
    extent = ((hi - lo) / span).tolist()
    inside = np.all((context >= lo) & (context <= hi), axis=1)
    context, labels = context[inside], labels[inside]
    logger.info('dropped %d background outliers outside the bounds', int((~inside).sum()))

    held = {str(h): b64(quantize(v, lo, span), '<u2') for h, v in held_cells.items()}

    intervals = []
    for (s, h, e), path in zip(INTERVALS, paths):
        sub = path[::TIME_STRIDE]
        if (len(path) - 1) % TIME_STRIDE:
            sub = np.concatenate([sub, path[-1:]])
        if not np.isfinite(sub).all():
            raise ValueError(f'non-finite rollout in interval {s}->{e}')
        intervals.append({'start': s, 'held': h, 'end': e, 'frames': int(sub.shape[0]),
                          'cells': int(sub.shape[1]), 'q': b64(quantize(sub, lo, span), '<u2')})

    out = {
        'source': 'PACE WOT reprogramming demo (Schiebinger et al., 2019), FLE embedding',
        'meta': meta,
        'extent': extent,
        'days': days,
        'context': {'n': int(len(context)), 'q': b64(quantize(context, lo, span), '<u2'),
                    'day': b64(labels, 'u1')},
        'held': held,
        'intervals': intervals,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(',', ':')))
    logger.info('wrote %s (%.0f KB): %d context cells over %d days, %d intervals x %d cells x %d frames',
                args.out, args.out.stat().st_size / 1024, len(context), len(days),
                len(intervals), intervals[0]['cells'], intervals[0]['frames'])


if __name__ == '__main__':
    main()
