#!/usr/bin/env python3
"""Render a trained 2D Ocean flow and save its portable configuration/provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from src.inference import load_run, rollout
from src.methods.mfm.runner import build_datamodule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'assets/ocean-demo.gif')
    parser.add_argument('--steps', type=int, default=801)
    parser.add_argument('--selection-note', default='Illustrative training run; not a published-table reproduction.')
    args = parser.parse_args()
    if args.steps < 9 or (args.steps - 1) % 8:
        parser.error('--steps minus one must be a positive multiple of eight.')
    torch.set_num_threads(1)
    model, cfg, checkpoint = load_run(args.run_dir)
    if cfg.data_name != 'oceans_positions' or cfg.dim != 2:
        parser.error('This layout expects the 2D Ocean benchmark.')
    # Preserve a portable, complete configuration before resolving local paths.
    portable = OmegaConf.to_container(cfg, resolve=True)
    portable['data_path'] = 'data/oceans/oceans.npz'
    portable['working_dir'] = 'results/ocean_demo'
    portable['load_stage1_ckpt'] = None
    portable['load_stage2_ckpt'] = None
    cfg.data_path = str(ROOT/portable['data_path'])
    dm = build_datamodule(cfg)
    dm.setup()
    labels = sorted(dm.train_frames)
    path = rollout(model, dm.train_frames[labels[0]], steps=args.steps).numpy()
    observed = np.load(cfg.data_path)['positions']
    bg, ink, red, teal = '#F5F7FA', '#102B4A', '#E33845', '#168B9D'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), facecolor=bg)
    fig.subplots_adjust(left=.045, right=.965, top=.77, bottom=.14, wspace=.12)
    fig.text(.05, .92, 'PACE', size=23, weight='bold', color=ink)
    fig.text(.05, .855, 'Continuous dynamics from unpaired snapshots', size=13, color=ink)
    fig.text(.96, .91, 'OCEAN  /  2D', ha='right', size=11, color=teal, weight='bold')
    cloud = np.concatenate([observed.reshape(-1, 2), path.reshape(-1, 2)])
    low, high = cloud.min(0), cloud.max(0)
    pad = (high-low)*.08
    for ax, title in zip(axes, ['01  OBSERVED SNAPSHOTS', '02  LEARNED VELOCITY FLOW']):
        ax.set_facecolor('white')
        ax.set_title(title, loc='left', color=ink, size=10, pad=12, weight='bold')
        ax.set_xlim(low[0]-pad[0], high[0]+pad[0])
        ax.set_ylim(low[1]-pad[1], high[1]+pad[1])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#DFE5ED')
        for frame in observed:
            ax.scatter(*frame.T, s=5, color=ink, alpha=.045)
    for lab in labels:
        axes[0].scatter(*observed[lab].T, s=8, c=np.full(len(observed[lab]), lab),
                        vmin=0, vmax=8, cmap='viridis', alpha=.7)
    for lab in cfg.test_timepoint_labels:
        axes[0].scatter(*observed[lab].T, s=13, facecolors='none', edgecolors=red,
                        alpha=.45, linewidths=.6)
    axes[0].text(.04, .035, 'Filled: training   /   Outline: held out',
                 transform=axes[0].transAxes, size=8, color=ink)
    for j in range(0, path.shape[1], 3):
        axes[1].plot(path[:, j, 0], path[:, j, 1], color=teal, alpha=.18, lw=.7)
    # t=7 is shown as a reference cloud; it is not used for model training.
    axes[1].scatter(*observed[7].T, s=12, facecolors='none', edgecolors=ink,
                    alpha=.4, linewidths=.6)
    axes[1].text(.04, .035, 'Outlined reference: held-out t = 7',
                 transform=axes[1].transAxes, size=8, color=ink)
    moving = axes[1].scatter(*path[0].T, s=12, color=red, alpha=.9, edgecolors='none')
    timer = fig.text(.96, .055, '', ha='right', size=10, color=ink)
    fig.text(.05, .055, 'Continuous rollout from t = 0  •  Demonstration model', size=9, color=ink)

    def update(i):
        moving.set_offsets(path[i])
        timer.set_text(f't = {8*i/(args.steps-1):.2f} / 8')
        return moving, timer

    frames = np.linspace(0, args.steps-1, 161, dtype=int).tolist()
    focus_index = 7*(args.steps-1)//8
    focus_position = frames.index(focus_index)
    frames[focus_position:focus_position] = [focus_index]*18
    frames += [args.steps-1]*12
    # Figure-level time text must also be included in saved frames/posters.
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.output, writer=PillowWriter(fps=20), dpi=100)
    update(focus_index)
    fig.savefig(args.output.with_suffix('.png'), dpi=140)
    plt.close(fig)
    with Image.open(args.output) as gif:
        encoded_frames = gif.n_frames
        duration_ms = 0
        for index in range(encoded_frames):
            gif.seek(index)
            duration_ms += gif.info.get('duration', 0)
    config_path = args.output.with_name(args.output.stem+'-config.yaml')
    OmegaConf.save(OmegaConf.create(portable), config_path)
    report = {
        'description': args.selection_note,
        'configuration': config_path.name,
        'command': f'python train.py --config-path {os.path.relpath(config_path.parent.resolve(), ROOT)} --config-name {config_path.stem}',
        'seed': int(cfg.seed),
        'training_accelerator': cfg.accelerator,
        'stage1_epochs': int(cfg.total_epochs_stage1),
        'stage2_epochs': int(cfg.epochs),
        'sigma': float(cfg.sigma),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'data_sha256': hashlib.sha256(Path(cfg.data_path).read_bytes()).hexdigest(),
        'integration_steps': args.steps,
        'display_frames': len(frames),
        'encoded_gif_frames': encoded_frames,
        'duration_ms': duration_ms,
        'time_labels': 'Original Ocean labels 0..8; model time is label / 8.',
        'render_command': f'python scripts/render_demo.py {portable["working_dir"]}/results/oceans_positions_dim2_test1-3-5-7/pace --steps {args.steps}',
    }
    args.output.with_name('demo-provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
