#!/usr/bin/env python3
"""Render three genuine WOT held-out reconstruction intervals from a PACE run."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from src.inference import load_run, rollout
from src.methods.mfm.runner import build_datamodule
from scripts.evaluate_reprogramming_demo import distribution_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'assets/reprogramming-demo.gif')
    parser.add_argument('--selection-note', default='See the accompanying training record for model selection.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, cfg, checkpoint = load_run(args.run_dir)
    if cfg.data_name != 'wot_fle_three_interpolation' or cfg.dim != 2 or list(cfg.test_timepoint_labels) != [6, 11, 16]:
        parser.error('Expected WOT FLE 2D with held-out labels 6,11,16.')
    portable = OmegaConf.to_container(cfg, resolve=True)
    portable.update(data_path='data/scnode/wot_fle/wot_fle.h5ad', working_dir='results/reprogramming_demo',
                    load_stage1_ckpt=None, load_stage2_ckpt=None)
    data_path = ROOT/portable['data_path']
    cfg.data_path = str(data_path)
    dm = build_datamodule(cfg)
    dm.setup()
    snapshots = {**dm.train_frames, **dm.test_frames}
    context = torch.cat(list(snapshots.values())).numpy()
    windows = [(5, 6, 7), (10, 11, 12), (15, 16, 17)]
    paths, metrics, endpoint_metrics = [], [], []
    for start, held, end in windows:
        path = rollout(model, dm.train_frames[start], t_start=start/18, t_end=end/18, steps=201)
        if not torch.isfinite(path).all():
            raise ValueError('Non-finite trajectory; cannot render.')
        metrics.append({'start': start, 'held_out': held, 'end': end,
                        **distribution_diagnostics(path[100], dm.test_frames[held])})
        endpoint_metrics.append({'start': start, 'end': end,
                                 **distribution_diagnostics(path[-1], dm.train_frames[end])})
        paths.append(path.numpy())

    bg, panel, ink, muted = '#0C182B', '#111F35', '#EEF4FC', '#91A5C3'
    colors = ['#70B8FF', '#5BDEC7', '#FC93AC']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig = plt.figure(figsize=(12.6, 6.5), facecolor=bg)
    fig.text(.045, .925, 'Reprogramming, in motion.', color=ink, fontsize=25, weight='bold')
    fig.text(.045, .868, 'Three unseen snapshots. One learned velocity field.', color=muted, fontsize=12)
    fig.text(.955, .931, 'PACE', ha='right', color=ink, fontsize=19, weight='bold')
    fig.text(.955, .886, 'SCHIEBINGER2019  /  WOT FLE', ha='right', color=muted, fontsize=8.5)
    artists = []
    for k, ((start, held, end), path, color) in enumerate(zip(windows, paths, colors)):
        x = .045+k*.31
        ax = fig.add_axes([x, .255, .29, .49], facecolor=panel)
        ax.set_title(f'{["EARLY", "INTERMEDIATE", "LATE"][k]}  ·  {start} → {end}',
                     loc='left', color=color, fontsize=10, weight='bold', pad=15)
        points = np.concatenate([snapshots[t].numpy() for t in [start, held, end]]+[path.reshape(-1, 2)])
        low, high = points.min(0), points.max(0)
        center, span = (low+high)/2, np.maximum(high-low, 1e-6)*1.14
        ratio = (.29*12.6)/(.49*6.5)
        span[0] = max(span[0], span[1]*ratio)
        span[1] = max(span[1], span[0]/ratio)
        ax.set_xlim(center[0]-span[0]/2, center[0]+span[0]/2)
        ax.set_ylim(center[1]-span[1]/2, center[1]+span[1]/2)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_color('#243751')
        ax.scatter(*context[::2].T, s=2, c='#758AA8', alpha=.15, linewidths=0)
        ref = snapshots[held].numpy()
        ax.scatter(*ref.T, s=9, facecolors='none', edgecolors='#CAD8EC', alpha=.25, linewidths=.45)
        tails = LineCollection([], colors=color, linewidths=.8, alpha=.4)
        ax.add_collection(tails)
        glow = ax.scatter(*path[0].T, s=27, c=color, alpha=.08, linewidths=0)
        moving = ax.scatter(*path[0].T, s=6, c=color, alpha=.85, linewidths=0)
        time_text = fig.text(x+.145, .198, '', ha='center', color=ink, fontsize=10)
        bar = fig.add_axes([x+.025, .13, .24, .035], facecolor=bg)
        bar.set_xlim(0,1); bar.set_ylim(-.25,.25); bar.axis('off')
        bar.plot([0,1],[0,0],color='#2A3B54',lw=2)
        bar.scatter([.5],[0],s=28,facecolors=bg,edgecolors='#CBD8EA',lw=.9,zorder=3)
        progress, = bar.plot([],[],color=color,lw=2.5)
        cursor = bar.scatter([0],[0],s=30,c=color,zorder=4)
        fig.text(x+.025,.103,str(start),color=muted,fontsize=8,ha='center')
        fig.text(x+.145,.103,f'{held} · held out',color=muted,fontsize=8,ha='center')
        fig.text(x+.265,.103,str(end),color=muted,fontsize=8,ha='center')
        artists.append((tails,glow,moving,time_text,progress,cursor))
    fig.text(.5,.045,'Filled points: model rollout   ·   Outlines: held-out cells   ·   Each interval starts from an observed training snapshot',
             ha='center',color=muted,fontsize=8.3)
    fig.text(.5,.018,'Panels advance by the same fraction of their labeled intervals. Paths are inferred population transport, not measured cell lineages.',
             ha='center',color=muted,fontsize=7.6)

    def update(index):
        updated=[]
        for (start,held,end),path,(tails,glow,moving,label,progress,cursor) in zip(windows,paths,artists):
            moving.set_offsets(path[index]); glow.set_offsets(path[index])
            tails.set_segments(path[max(0,index-20):index+1,::3].transpose(1,0,2))
            label.set_text(f'snapshot {start+(end-start)*index/200:.2f}'+('  ·  held out' if index==100 else ''))
            progress.set_data([0,index/200],[0,0]);cursor.set_offsets([[index/200,0]])
            updated.extend([tails,glow,moving,label,progress,cursor])
        return updated
    frame_ids = np.linspace(0,200,141,dtype=int).tolist()
    frame_ids[frame_ids.index(100):frame_ids.index(100)] = [100]*20
    frame_ids += [200]*20
    animation = FuncAnimation(fig,update,frames=frame_ids,interval=50,blit=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    animation.save(args.output,writer=PillowWriter(fps=20),dpi=100)
    update(100);fig.savefig(args.output.with_suffix('.png'),dpi=140);plt.close(fig)
    with Image.open(args.output) as gif:
        duration=0
        for i in range(gif.n_frames):gif.seek(i);duration+=gif.info.get('duration',0)
        encoded=gif.n_frames
    config_file=args.output.with_name(args.output.stem+'-config.yaml')
    OmegaConf.save(OmegaConf.create(portable),config_file)
    config_parent = config_file.resolve().parent
    if config_parent.is_relative_to(ROOT):
        config_parent = config_parent.relative_to(ROOT)
    report={
        'description':'Fresh PACE run on WOT reprogramming data. Three independent observed-anchor reconstructions; not a continuous rollout across all 19 snapshots or measured cell lineages.',
        'training_command':shlex.join(['python', 'train.py', '--config-path', str(config_parent), '--config-name', config_file.stem]),
        'selection_note':args.selection_note,
        'render_command':'python scripts/render_reprogramming_demo.py results/reprogramming_demo/results/wot_fle_three_interpolation_dim2_test6-11-16/pace',
        'configuration':config_file.name,'seed':int(cfg.seed),'training_accelerator':cfg.accelerator,
        'stage1_epochs':int(cfg.total_epochs_stage1),'stage2_epochs':int(cfg.epochs),
        'sigma':float(cfg.sigma),'matching_method':cfg.matching_method,
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'train_labels':list(cfg.train_timepoint_labels),'heldout_labels':list(cfg.test_timepoint_labels),
        'integration':{'solver':'Euler','points_per_interval':201,'normalized_time':'snapshot label / 18','intervals':windows,'resets_within_intervals':False},
        'coordinates':'Original training-only standardization; fixed local limits per panel contain all model trajectories.',
        'display':{'particles_per_panel':[p.shape[1] for p in paths],'selection':'All selected source cells; no quality-based filtering.','background':'Seeded training/evaluation frame samples.', 'encoded_frames':encoded,'duration_ms':duration},
        'heldout_diagnostics':metrics,
        'endpoint_diagnostics':endpoint_metrics,
    }
    args.output.with_name(args.output.stem+'-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.run_dir/'metrics/readme_reprogramming_rollouts.npz',**{f'interval_{s}_{e}':p for (s,h,e),p in zip(windows,paths)})
    print(args.output)


if __name__=='__main__':main()
