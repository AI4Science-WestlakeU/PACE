#!/usr/bin/env python3
"""Render the README two-panel WOT animation (days 10->12 and 15->17).

Reads the rollouts cached by scripts/render_reprogramming_demo.py in
<run_dir>/metrics/readme_reprogramming_rollouts.npz; the model is not re-run.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from src.methods.mfm.runner import build_datamodule

# (start, held-out, end) snapshot labels and PACE logo colours.
WINDOWS = [((10, 11, 12), '#24467C'), ((15, 16, 17), '#D1343F')]
BG, CONTEXT, HELD, RULE, TICK, TEXT = 'white', '#C3C9D0', '#59626D', '#D9DEE3', '#C6CCD3', '#6A737D'
W, H = 10.0, 5.4
STEPS = 200  # cached rollouts hold 201 Euler points per interval


def load(run_dir):
    cache = run_dir/'metrics/readme_reprogramming_rollouts.npz'
    if not cache.exists():
        raise FileNotFoundError(f'{cache} is missing; run scripts/render_reprogramming_demo.py first.')
    cfg = OmegaConf.load(run_dir/'resolved_config.yaml')
    cfg.data_path = str(ROOT/'data/scnode/wot_fle/wot_fle.h5ad')
    dm = build_datamodule(cfg)
    dm.setup()
    snapshots = {t: v.numpy() for t, v in {**dm.train_frames, **dm.test_frames}.items()}
    cached = np.load(cache)
    return snapshots, [cached[f'interval_{s}_{e}'] for (s, _, e), _ in WINDOWS]


def limits(points, ratio):
    low, high = points.min(0), points.max(0)
    center, span = (low+high)/2, np.maximum(high-low, 1e-6)*1.08
    span[0] = max(span[0], span[1]*ratio)
    span[1] = max(span[1], span[0]/ratio)
    return center-span/2, center+span/2


def build_figure(snapshots, paths):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 14,
                         'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']})
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    context = np.concatenate(list(snapshots.values()))[::2]
    pw, py, ph = .43, 1.45/H, 3.8/H
    panels = []
    for k, (((start, held, end), color), path) in enumerate(zip(WINDOWS, paths)):
        x = .05+k*.47
        ax = fig.add_axes([x, py, pw, ph])
        points = np.concatenate([snapshots[t] for t in (start, held, end)]+[path.reshape(-1, 2)])
        lo, hi = limits(points, (pw*W)/(ph*H))
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect='equal')
        ax.axis('off')
        ax.scatter(*context.T, s=1.6, c=CONTEXT, alpha=.75, linewidths=0, zorder=0)
        ax.scatter(*snapshots[held].T, s=13, facecolors='none', edgecolors=HELD,
                   alpha=.45, linewidths=.55, zorder=2)
        lines = LineCollection([], colors=color, linewidths=.6, alpha=.3, zorder=3)
        ax.add_collection(lines)
        cells = ax.scatter(*path[0].T, s=8, c=color, alpha=.9, linewidths=0, zorder=4)

        bar = fig.add_axes([x, .97/H, pw, .3/H])
        bar.set(xlim=(-.015, 1.015), ylim=(-1, 1))
        bar.axis('off')
        bar.plot([0, 1], [0, 0], color=RULE, lw=1.8, solid_capstyle='round')
        for u in (0, 1):
            bar.plot([u, u], [-.45, .45], color=TICK, lw=1.2)
        progress, = bar.plot([], [], color=color, lw=2.8, solid_capstyle='round')
        # Hollow ring marks the held-out snapshot, same glyph as the held-out cells.
        ring = bar.scatter([.5], [0], s=56, facecolors=BG, edgecolors=HELD, linewidths=1.2, zorder=3)
        cursor = bar.scatter([0], [0], s=40, c=color, zorder=4, linewidths=0)
        labels = [fig.text(x+u*pw, .75/H, str(label), color=TEXT, fontsize=18, ha=ha)
                  for u, label, ha in ((0, start, 'left'), (.5, held, 'center'), (1, end, 'right'))]
        panels.append(dict(path=path, color=color, lines=lines, cells=cells, progress=progress,
                           cursor=cursor, ring=ring, held_label=labels[1]))

    handles = [Line2D([], [], ls='', marker='o', ms=7.5, mfc='none', mec=HELD, mew=1.1),
               Line2D([], [], ls='', marker='o', ms=5.5, mfc=CONTEXT, mec='none')]
    fig.legend(handles, ['held out', 'observed'], loc='lower center', bbox_to_anchor=(.5, .05/H), ncol=2,
               frameon=False, fontsize=16.5, handletextpad=.45, columnspacing=2.4, labelcolor=TEXT)

    def update(index):
        u, at_held = index/STEPS, index == STEPS//2
        for p in panels:
            # Lines for every third cell keep the accumulated paths readable; dots show all cells.
            p['lines'].set_segments(list(p['path'][:index+1, ::3].transpose(1, 0, 2)))
            p['cells'].set_offsets(p['path'][index])
            p['progress'].set_data([0, u], [0, 0])
            p['cursor'].set_offsets([[u, 0]])
            p['held_label'].set_color(p['color'] if at_held else TEXT)
            p['ring'].set_edgecolor(p['color'] if at_held else HELD)
    return fig, update


def frame_schedule():
    ids = np.linspace(0, STEPS, 141).round().astype(int).tolist()
    mid = ids.index(STEPS//2)
    # Short holds at the start, at the held-out snapshot and at the end.
    return [0]*10 + ids[:mid] + [STEPS//2]*24 + ids[mid:] + [STEPS]*24


def save_gif(frames, output, fps):
    # One shared palette from the fullest (last) frame avoids colour flicker between frames.
    palette = frames[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    rgb = np.array(palette.getpalette()[:128*3]).reshape(-1, 3)
    rgb[rgb.sum(1).argmax()] = 255
    quantized = []
    for frame in frames:
        # Exact nearest-colour lookup; Pillow's palette mapping shifts pure white to 252.
        pixels = np.asarray(frame).reshape(-1, 3).astype(np.int32)
        colours, inverse = np.unique(pixels, axis=0, return_inverse=True)
        nearest = ((colours[:, None, :]-rgb[None])**2).sum(-1).argmin(1)
        image = Image.frombytes('P', frame.size, nearest[inverse.ravel()].astype(np.uint8).tobytes())
        image.putpalette(rgb.ravel().tolist())
        quantized.append(image)
    quantized[0].save(output, save_all=True, append_images=quantized[1:], duration=int(1000/fps),
                      loop=0, optimize=True, disposal=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'assets/reprogramming-two-panel.gif')
    parser.add_argument('--fps', type=int, default=20)
    args = parser.parse_args()
    snapshots, paths = load(args.run_dir)
    fig, update = build_figure(snapshots, paths)
    fig.set_dpi(100)
    frames = []
    for index in frame_schedule():
        update(index)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    plt.close(fig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, args.output, args.fps)
    print(args.output)


if __name__ == '__main__':
    main()
