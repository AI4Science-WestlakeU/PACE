#!/usr/bin/env python3
"""Exercise both training stages, checkpointing and held-out evaluation on CPU."""
import csv
import math
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
for method in ['pace', 'mfm']:
    command = [sys.executable, str(ROOT/'train.py'), f'experiment=oceans_positions_{method}_ode',
               'trainer=cpu', 'epochs=2', 'samples_per_timepoint=24',
               'batch_size=24', 'enable_progress_bar=false',
               f'working_dir={ROOT / "results" / "smoke"}']
    command += ['total_epochs_stage1=3', 'rematch_every=1', 'stage1_batch_size=24', 'T_steps=3', 't_probe_steps=3'] if method == 'pace' else ['epochs_geopath=2']
    subprocess.run(command, cwd=ROOT, check=True)
    metrics = ROOT/'results/smoke/results/oceans_positions_dim2_test1-3-5-7'/method/'metrics/distribution_metrics.csv'
    with metrics.open() as f: rows = list(csv.DictReader(f))
    rows = [r for r in rows if r['stage'] == 'stage2' and r.get('predictor') == 'ode_rollout']
    assert {int(r['test_label']) for r in rows} == {1, 3, 5, 7}, rows
    assert all(math.isfinite(float(r[k])) for r in rows for k in ['mmd', 'w1', 'w2']), rows
    print(f'{method}: both stages and all four holdouts passed.', flush=True)
