#!/usr/bin/env python3
"""Render an EB developmental-data animation from a trained PACE velocity field."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from src.inference import load_run, rollout
from src.methods.mfm.runner import build_datamodule
from src.methods.mfm.eval_metrics import gaussian_mmd, wasserstein_2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'assets/eb-development.gif')
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, cfg, checkpoint = load_run(args.run_dir)
    if (cfg.data_name != 'eb_phate' or cfg.dim != 2
            or list(cfg.train_timepoint_labels) != [0, 1, 2, 4]
            or list(cfg.test_timepoint_labels) != [3]):
        parser.error('Expected EB PHATE with train labels 0,1,2,4 and held-out label 3.')
    data_path = ROOT/'data/eb_velocity_v5.npz'
    portable = OmegaConf.to_container(cfg, resolve=True)
    portable.update(data_path='data/eb_velocity_v5.npz', working_dir='results/eb_development_demo',
                    load_stage1_ckpt=None, load_stage2_ckpt=None)
    cfg.data_path = str(data_path)
    dm = build_datamodule(cfg)
    dm.setup()  # Reconstruct the exact training-only scaler and seeded sampling.
    path = rollout(model, dm.train_frames[2], t_start=0.5, t_end=1.0, steps=401).numpy()
    if not np.isfinite(path).all():
        raise ValueError('Non-finite model trajectory; cannot render.')
    data = np.load(data_path)
    observed = {i: data['phate'][data['sample_labels'] == i].astype(np.float32) for i in [2, 3, 4]}
    if dm.scaler is not None:
        observed = {i: dm.scaler.transform(x).astype(np.float32) for i, x in observed.items()}
    # No particle selection by model quality: a fixed evenly spaced display subset.
    display_ids = np.linspace(0, path.shape[1]-1, min(768, path.shape[1]), dtype=int)
    shown = path[:, display_ids]
    reference = {**dm.train_frames, **dm.test_frames}
    diagnostics = []
    for label in [3, 4]:
        pred = torch.tensor(path[(label-2)*200])
        diagnostics.append({'label': label, 'held_out': label == 3,
                            'w2': wasserstein_2(pred, reference[label]),
                            'mmd': gaussian_mmd(pred, reference[label])})

    ink, muted, paper = '#142D4B', '#64778E', '#F6F8FC'
    cmap = LinearSegmentedColormap.from_list('development', ['#536DE8', '#3F9DD6', '#1BB5A6', '#F2B65D', '#E97076'])
    colors = [cmap(i/4) for i in range(5)]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig = plt.figure(figsize=(12, 6.3), facecolor=paper)
    left = fig.add_axes([.045, .22, .43, .56], facecolor='white')
    right = fig.add_axes([.525, .22, .43, .56], facecolor='white')
    fig.text(.045, .925, 'Development, in motion.', color=ink, fontsize=25, weight='bold')
    fig.text(.045, .865, 'Reconstructing the unseen state between developmental snapshots', color=muted, fontsize=12)
    fig.text(.955, .935, 'PACE', ha='right', color=ink, fontsize=18, weight='bold')
    fig.text(.955, .889, 'EMBRYOID BODIES  /  PHATE', ha='right', color=muted, fontsize=9)
    all_points = np.concatenate([*observed.values(), path.reshape(-1, 2)])
    low, high = all_points.min(0), all_points.max(0)
    center = (low+high)/2
    # Equal coordinates and limits in both panels; every trajectory stays visible.
    span = np.maximum(high-low, 1e-6)*1.13
    ratio = (.43*12)/(.56*6.3)
    span[0] = max(span[0], span[1]*ratio)
    span[1] = max(span[1], span[0]/ratio)
    for ax, title in [(left, '01   SNAPSHOTS 2, 3 AND 4'), (right, '02   PACE-INFERRED MOTION')]:
        ax.set_title(title, loc='left', fontsize=10, color=ink, weight='bold', pad=13)
        ax.set_xlim(center[0]-span[0]/2, center[0]+span[0]/2)
        ax.set_ylim(center[1]-span[1]/2, center[1]+span[1]/2)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_color('#E2E8F1')
    backdrop = np.concatenate(list(observed.values()))
    right.scatter(*backdrop[::2].T, s=1.5, c='#8B9CAF', alpha=.12, linewidths=0, rasterized=True)
    for label, points in observed.items():
        ids = np.linspace(0, len(points)-1, min(1600, len(points)), dtype=int)
        if label == 3:
            left.scatter(*points[ids].T, s=6, facecolors='none', edgecolors=[colors[label]],
                         linewidths=.4, alpha=.43)
        else:
            left.scatter(*points[ids].T, s=4, color=colors[label], alpha=.42, linewidths=0)
    left.text(.025, .025, 'Color = snapshot   •   Outline = held out',
              transform=left.transAxes, fontsize=8, color=muted,
              bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.85, 'pad':2})
    right.text(.025, .025, 'Continuous model rollout: snapshot 2 → 4',
               transform=right.transAxes, fontsize=7.7, color=muted,
               bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.85, 'pad':2})
    tails = LineCollection([], linewidths=.8, alpha=.24)
    right.add_collection(tails)
    glow = right.scatter(*shown[0].T, s=23, c=[colors[0]], alpha=.065, linewidths=0)
    moving = right.scatter(*shown[0].T, s=5.5, c=[colors[0]], alpha=.75, linewidths=0)
    timeline = fig.add_axes([.1, .075, .8, .072], facecolor=paper)
    timeline.set_xlim(1.85, 4.15); timeline.set_ylim(-.7, .6); timeline.axis('off')
    timeline.plot([2, 4], [0, 0], c='#D5DFEC', lw=2, zorder=1)
    progress, = timeline.plot([], [], c=colors[0], lw=3, zorder=2)
    cursor = timeline.scatter([2], [0], s=75, color=[colors[2]], zorder=4, edgecolor='white', linewidth=1.5)
    for label in [2, 3, 4]:
        timeline.scatter([label], [0], s=34, facecolors='white' if label == 3 else [colors[label]],
                         edgecolors=[colors[label]], linewidth=1.4, zorder=3)
        timeline.text(label, -.43, str(label)+(' · held out' if label == 3 else ''),
                      ha='center', fontsize=8, color=ink if label == 3 else muted)
    clock = fig.text(.955, .17, '', ha='right', color=ink, fontsize=10)
    fig.text(.045, .17, 'DEVELOPMENTAL SNAPSHOT', color=muted, fontsize=8, weight='bold')
    fig.text(.5, .027, 'Learned population transport in an embedding; trajectories are not measured cell lineages.',
              ha='center', color=muted, fontsize=8)

    def update(index):
        t = 2+index/200
        color = cmap(t/4)
        moving.set_offsets(shown[index]); moving.set_color(color)
        glow.set_offsets(shown[index]); glow.set_color(color)
        history = shown[max(0,index-22):index+1, ::3]
        tails.set_segments(history.transpose(1, 0, 2)); tails.set_color(color)
        progress.set_data([2, t], [0, 0]); progress.set_color(color)
        cursor.set_offsets([[t, 0]]); cursor.set_color(color)
        clock.set_text(f't = {t:.2f}  |  interval 2 → 4'+('  ·  held out' if index == 200 else ''))
        return moving, glow, tails, progress, cursor, clock

    frames = np.linspace(0, 400, 161, dtype=int).tolist()
    frames[frames.index(200):frames.index(200)] = [200]*16
    frames += [400]*20
    animation = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=20), dpi=100)
    update(200)
    fig.savefig(args.output.with_suffix('.png'), dpi=140)
    plt.close(fig)
    with Image.open(args.output) as gif:
        duration = 0
        for i in range(gif.n_frames):
            gif.seek(i); duration += gif.info.get('duration', 0)
        encoded = gif.n_frames
    config_path = args.output.with_name(args.output.stem+'-config.yaml')
    OmegaConf.save(OmegaConf.create(portable), config_path)
    report = {
        'description': 'Fresh EB developmental-data PACE run using the unchanged release preset; illustrative population dynamics, not a paper-table reproduction or measured lineage movie.',
        'training_command': 'python train.py experiment=eb_phate_pace_ode seed=42 enable_progress_bar=false working_dir=results/eb_development_demo',
        'render_command': 'python scripts/render_development_demo.py results/eb_development_demo/results/eb_phate_dim2_test3/pace',
        'configuration': config_path.name, 'seed': int(cfg.seed),
        'training_accelerator': cfg.accelerator,
        'stage1_epochs': int(cfg.total_epochs_stage1), 'stage2_epochs': int(cfg.epochs),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'train_labels': list(cfg.train_timepoint_labels), 'heldout_labels': list(cfg.test_timepoint_labels),
        'integration': {'solver': 'Euler', 'points': 401, 'normalized_time': 'snapshot label / 4', 'source': 'selected training snapshot 2', 'start_label': 2, 'end_label': 4, 'resets_to_observations': False},
        'coordinates': 'PHATE coordinates standardized with the original training-only scaler; no display-time refitting.',
        'display': {'model_particles_integrated': path.shape[1], 'model_particles_shown': len(display_ids),
                    'selection': 'Evenly spaced indices; no selection by prediction quality.',
                    'color': 'Developmental snapshot time, not cell type or lineage.',
                    'encoded_frames': encoded, 'duration_ms': duration},
        'interval_rollout_diagnostics': diagnostics,
    }
    args.output.with_name(args.output.stem+'-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    metrics_dir = args.run_dir/'metrics'
    metrics_dir.mkdir(exist_ok=True)
    np.savez_compressed(metrics_dir/'readme_development_rollout.npz', positions=path, times=np.linspace(2,4,401), display_ids=display_ids)
    print(args.output)


if __name__ == '__main__':
    main()
