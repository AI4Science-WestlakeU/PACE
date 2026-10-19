#!/usr/bin/env python3
"""Build page assets from the PACE repository (read-only).

- assets/pace-mark-dark.png : logo with transparent background, navy -> near-white (for the dark hero)
- assets/favicon.png        : swoosh only, original colours
- assets/swoosh-dark.png    : swoosh only, dark-hero colours
- assets/fig1.webp          : paper Figure 1 (needs `cwebp` on PATH)

Usage:
  <pace>/.venv/bin/python tools/make_assets.py --pace <pace>
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
LOGO_WIDTH = 960
LIGHT_INK = np.array([236, 240, 246], dtype=np.float32)


def cut_background(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (alpha, is_red) from a logo drawn on white; alpha = distance from white."""
    dist = 255.0 - rgb.min(axis=2)
    alpha = np.clip(dist / 200.0, 0.0, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    is_red = (r > g + 60) & (r > b + 60)
    return alpha, is_red


def logo_variants(src: Path, out_dir: Path) -> None:
    """Write the light and dark transparent logo variants, trimmed to content."""
    rgb = np.asarray(Image.open(src).convert('RGB'), dtype=np.float32)
    alpha, is_red = cut_background(rgb)
    ys, xs = np.nonzero(alpha > 0.02)
    pad = 12
    box = (max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad)

    def save(colour: np.ndarray, name: str) -> None:
        rgba = np.dstack([colour, alpha * 255.0])[box[0]:box[1], box[2]:box[3]]
        image = Image.fromarray(rgba.clip(0, 255).astype(np.uint8))
        image.thumbnail((LOGO_WIDTH, LOGO_WIDTH), Image.Resampling.LANCZOS)
        image.save(out_dir / name, optimize=True)
        logger.info('wrote %s', out_dir / name)

    navy = np.array([16, 38, 74], dtype=np.float32)
    red = np.array([209, 52, 63], dtype=np.float32)
    light = np.where(is_red[..., None], red, navy)
    dark = np.where(is_red[..., None], np.array([240, 82, 92], dtype=np.float32), LIGHT_INK)
    save(dark, 'pace-mark-dark.png')

    # Swoosh only (left of the wordmark), square-padded, for the favicon and nav.
    cols = np.nonzero((alpha > 0.02).any(axis=0))[0]
    gap = np.nonzero(np.diff(cols) > 20)[0]
    right = cols[gap[0]] + pad if len(gap) else cols.max()
    for colour, name in ((light, 'favicon.png'), (dark, 'swoosh-dark.png')):
        swoosh = np.dstack([colour, alpha * 255.0])[box[0]:box[1], box[2]:right]
        h, w = swoosh.shape[:2]
        side = max(h, w)
        square = np.zeros((side, side, 4), dtype=np.float32)
        square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = swoosh
        icon = Image.fromarray(square.clip(0, 255).astype(np.uint8)).resize((128, 128), Image.Resampling.LANCZOS)
        icon.save(out_dir / name, optimize=True)
        logger.info('wrote %s', out_dir / name)


def main() -> None:
    """Generate all page assets."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pace', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    out = Path(__file__).resolve().parents[1] / 'assets'
    out.mkdir(exist_ok=True)
    logo_variants(args.pace / 'assets/pace-logo.png', out)
    if shutil.which('cwebp') is None:
        raise FileNotFoundError('cwebp not found; install libwebp to build fig1.webp')
    subprocess.run(['cwebp', '-quiet', '-q', '88', str(args.pace / 'assets/fig1.png'), '-o', str(out / 'fig1.webp')],
                   check=True)
    logger.info('wrote %s', out / 'fig1.webp')


if __name__ == '__main__':
    main()
