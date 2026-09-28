#!/usr/bin/env python3
"""Evaluate WOT demo holdouts AND endpoints, including spread/off-support checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.inference import load_run, rollout
from src.methods.mfm.runner import build_datamodule
from src.methods.mfm.eval_metrics import gaussian_mmd, wasserstein_2


def distribution_diagnostics(pred, target):
    pred, target = pred.float(), target.float()
    rms = lambda x: (x-x.mean(0)).square().sum(1).mean().sqrt()
    return {'w2': wasserstein_2(pred, target), 'mmd': gaussian_mmd(pred, target),
            'spread_ratio': float(rms(pred)/rms(target).clamp_min(1e-12)),
            'p95_nearest_distance': float(torch.quantile(torch.cdist(pred,target).min(1).values, .95))}


def evaluate(run_dir, output_dir=None, checkpoint_path=None):
    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    model, cfg, checkpoint = load_run(run_dir)
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        raw = torch.load(checkpoint,map_location='cpu',weights_only=False)
        state = {k.removeprefix('flow_net.'):v for k,v in raw['state_dict'].items() if k.startswith('flow_net.')}
        model.load_state_dict(state,strict=True)
    else:
        raw = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if cfg.data_name != 'wot_fle_three_interpolation' or cfg.dim != 2 or list(cfg.test_timepoint_labels) != [6,11,16]:
        raise ValueError('Expected WOT FLE 2D with holdouts 6,11,16.')
    cfg.data_path = str(ROOT/'data/scnode/wot_fle/wot_fle.h5ad')
    dm = build_datamodule(cfg); dm.setup()
    rows, paths = [], {}
    for start, held, end in [(5,6,7),(10,11,12),(15,16,17)]:
        path = rollout(model,dm.train_frames[start],t_start=start/18,t_end=end/18,steps=201)
        if not torch.isfinite(path).all():raise ValueError('Non-finite trajectory.')
        rows.append({'start':start,'held_out':held,'end':end,
                     'midpoint':distribution_diagnostics(path[100],dm.test_frames[held]),
                     'endpoint':distribution_diagnostics(path[200],dm.train_frames[end])})
        paths[f'interval_{start}_{end}'] = path.numpy()
    result = {'checkpoint':checkpoint.name,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'checkpoint_epoch_count':int(raw['epoch'])+1, 'seed':int(cfg.seed),'sigma':float(cfg.sigma),
              'matching_method':cfg.matching_method,'flow_activation':cfg.activation_flow,
              'data_sha256':hashlib.sha256(Path(cfg.data_path).read_bytes()).hexdigest(),
              'protocol':{'solver':'Euler','points_per_interval':201,'source':'All 512 selected observed source cells','coordinates':'Original training-only standardization'},
              'intervals':rows}
    output = Path(output_dir) if output_dir else run_dir/'metrics/reprogramming_diagnostics'
    output.mkdir(parents=True,exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/'predictions.npz',**paths)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    print(json.dumps(evaluate(args.run_dir,args.output_dir,args.checkpoint),indent=2))
