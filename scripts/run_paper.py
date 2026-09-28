#!/usr/bin/env python3
"""List or run the PACE/MFM paper experiments, each with a resolved config."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]

def experiments(method):
    names = [f"{d}_{method}_ode" for d in ["oceans_positions", "toy_bifurcation", "eb_phate", "wot_fle_three_interpolation", "ipsc_liu"]]
    names += [f"ipsc_liu_{method}_ode_{d}d_test{t}" for d in [10, 50] for t in [4, 8, 12, 16]]
    names += [f"op_{d}_{method}_ode_100d" for d in ["cite", "multi"]]
    return names

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['pace', 'mfm', 'both'], default='both')
    p.add_argument('--run', action='store_true', help='Execute training (otherwise only print commands).')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    for method in (['pace', 'mfm'] if args.method == 'both' else [args.method]):
        for experiment in experiments(method):
            cmd = [sys.executable, str(ROOT/'train.py'), f'experiment={experiment}', f'seed={args.seed}']
            # Isolate repeated seeds and paper runs from tutorial output.
            cmd += [f'working_dir={ROOT / "results" / "paper_runs" / ("seed_" + str(args.seed))}']
            if args.cpu: cmd += ['trainer=cpu']
            print(' '.join(cmd), flush=True)
            if args.run: subprocess.run(cmd, cwd=ROOT, check=True)
if __name__ == '__main__':
    main()
