#!/usr/bin/env python3
"""Re-evaluate an explicit, trusted Stage 2 checkpoint without training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from src.methods.mfm.ema import EMA
from src.methods.mfm.eval_metrics import (
    _find_train_bracket, _run_ode_rollout, gaussian_mmd,
    wasserstein_1, wasserstein_2,
)
from src.methods.mfm.flow_matcher import labels_to_timesteps
from src.methods.mfm.networks import VelocityNet
from src.methods.mfm.runner import build_datamodule


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(config_path, checkpoint_path, ot_max_iter=None):
    cfg = OmegaConf.load(config_path)
    # Inference only needs the explicitly supplied velocity weights. Remove
    # training-only interpolations before dataloaders resolve the full config.
    cfg.load_stage1_ckpt = None
    cfg.load_stage2_ckpt = None
    data_path = Path(cfg.data_path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    cfg.data_path = str(data_path)
    torch.set_num_threads(1)
    dm = build_datamodule(cfg)
    dm.setup()
    frames = getattr(dm, 'selected_train_frames', dm.train_frames)
    labels = sorted(frames)
    times = dict(zip(labels, labels_to_timesteps(labels)))
    model = VelocityNet(
        dim=cfg.dim, hidden_dims=list(cfg.hidden_dims_flow),
        activation=cfg.activation_flow, batch_norm=False,
    )
    if cfg.get('ema_decay') is not None:
        model = EMA(model, decay=cfg.ema_decay)
    # Lightning checkpoints contain Python metadata: use only trusted files.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict({
        key.removeprefix('flow_net.'): value
        for key, value in checkpoint['state_dict'].items()
        if key.startswith('flow_net.')
    }, strict=True)
    model.eval()
    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for label in sorted(dm.test_frames):
            before, after = _find_train_bracket(label, labels)
            if before is None or after is None:
                raise ValueError(f'{label} has no observed bracket; use forecasting diagnostics.')
            trajectory = _run_ode_rollout(model, frames[before], times[before], times[after], 101, 'cpu')
            prediction = trajectory[round((label - before) / (after - before) * 100)]
            reference = dm.test_frames[label]
            if not torch.isfinite(prediction).all():
                raise ValueError(f'Nonfinite prediction at {label}')
            if ot_max_iter is None:
                w1 = wasserstein_1(prediction, reference)
                w2 = wasserstein_2(prediction, reference)
            else:
                import ot
                cost = torch.cdist(prediction.float(), reference.float()).numpy()
                a, b = ot.unif(len(prediction)), ot.unif(len(reference))
                w1 = float(ot.emd2(a, b, cost, numItermax=ot_max_iter))
                w2 = float(ot.emd2(a, b, cost ** 2, numItermax=ot_max_iter)) ** 0.5
            rows.append({
                'label': float(label), 'n_prediction': len(prediction),
                'n_reference': len(reference), 'mmd': gaussian_mmd(prediction, reference),
                'w1': w1, 'w2': w2,
            })
    if not rows:
        raise ValueError('The configuration has no evaluable held-out frames.')
    return {
        'checkpoint_file': Path(checkpoint_path).name,
        'checkpoint_sha256': sha256(checkpoint_path),
        'checkpoint_epoch_count': int(checkpoint['epoch']) + 1,
        'config_sha256': sha256(config_path), 'data_sha256': sha256(data_path),
        'protocol': '101-point observed-anchor Euler grid; all configured frame samples; CPU.',
        'ot_max_iter': ot_max_iter if ot_max_iter is not None else 'inherited POT default',
        'metrics': rows, 'warnings': sorted({str(item.message) for item in caught}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Resolved YAML, e.g. configs_archive/<experiment>.yaml')
    parser.add_argument('--checkpoint', type=Path, required=True, help='An explicit trusted Stage 2 Lightning checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ot-max-iter', type=int, help='Optional OT convergence diagnostic; leave unset for historical evaluation')
    args = parser.parse_args()
    if args.ot_max_iter is not None and args.ot_max_iter <= 0:
        parser.error('--ot-max-iter must be positive')
    record = evaluate(args.config, args.checkpoint, args.ot_max_iter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
