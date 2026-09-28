#!/usr/bin/env python3
"""Evaluate an Ocean checkpoint using continuous and bracketed rollouts.

The README uses a continuous trajectory from label 0. Paper-style reconstruction
restarts from the observed preceding training frame at each held-out label.
"""
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
from src.methods.mfm.eval_metrics import compute_distribution_metrics


def evaluate(run_dir, output_dir=None):
    torch.set_num_threads(1)
    model, cfg, checkpoint = load_run(run_dir)
    if (cfg.data_name != 'oceans_positions' or cfg.dim != 2 or cfg.whiten
            or list(cfg.train_timepoint_labels) != [0, 2, 4, 6, 8]
            or list(cfg.test_timepoint_labels) != [1, 3, 5, 7]):
        raise ValueError('Expected the unwhitened Ocean 2D split: train 0,2,4,6,8; test 1,3,5,7.')
    data_path = ROOT/'data/oceans/oceans.npz'
    points = np.load(data_path)['positions']
    path = rollout(model, points[0], steps=801)
    continuous = []
    for label in [1, 3, 5, 7, 8]:
        metrics = compute_distribution_metrics(path[label*100], torch.tensor(points[label]).float())
        continuous.append({'label': label, **metrics})
    bracketed, bracket_predictions = [], []
    for label in [1, 3, 5, 7]:
        bracket = rollout(model, points[label-1], t_start=(label-1)/8,
                          t_end=(label+1)/8, steps=101)
        prediction = bracket[50]
        metrics = compute_distribution_metrics(prediction, torch.tensor(points[label]).float())
        bracketed.append({'label': label, 'source_label': label-1, **metrics})
        bracket_predictions.append(prediction.numpy())
    result = {
        'continuous_from_t0': continuous,
        'bracketed': bracketed,
        'integration': {'solver': 'Euler', 'continuous_grid_points': 801,
                        'bracket_grid_points': 101, 'model_time': 'label / 8'},
        'checkpoint': checkpoint.name,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'seed': int(cfg.seed),
    }
    destination = Path(output_dir) if output_dir else Path(run_dir)/'metrics'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'ocean_demo_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(destination/'ocean_demo_predictions.npz',
                        continuous=path.numpy(), times=np.linspace(0, 8, 801),
                        bracketed=np.stack(bracket_predictions), labels=[1, 3, 5, 7])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run_dir, args.output_dir), indent=2))
