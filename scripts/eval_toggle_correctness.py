#!/usr/bin/env python
"""Toggle-switch (ControlledToggle2D) trajectory-correctness evaluator.

Answers reviewer Q2: does the normal-motion penalty suppress *real*
transverse transitions?  One fine ODE rollout from the source snapshot gives:

1. **Basin accuracy** per snapshot (predicted basin via the operational
   separatrix x>y vs GT basin labels, transit labels excluded);
2. **Switch-time error** for the init-A cohort (first predicted x<y crossing,
   in physical time units, vs GT switch times);
3. **Spurious transitions** (init-B particles predicted to enter the x>y side)
   and **missed transitions** (init-A particles that never cross);
4. **Velocity cosine** against the exact analytic drift field;
5. **ADE/FDE** and terminal **W2** (same conventions as the CurvedPitchfork
   evaluator).

Usage:
    python scripts/eval_toggle_correctness.py <run_dir> --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_trajectory_correctness import (  # noqa: E402
    build_velocity_model,
    compute_w2_terminal,
    infer_stage2_checkpoint,
    load_yaml,
    make_datamodule,
)
from src.methods.pace.stage2.eval_metrics import _run_ode_rollout  # noqa: E402
from src.methods.pace.stage2.flow_matcher import labels_to_timesteps  # noqa: E402

log = logging.getLogger("toggle_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=str)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-ode-steps", type=int, default=101)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def load_toggle_gt(cfg: dict[str, Any]) -> dict[str, Any]:
    data_path = Path(str(cfg["data_path"]))
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    data = np.load(data_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def basin_of_points(x: np.ndarray) -> np.ndarray:
    """Operational separatrix: x>y -> basin A, else basin B."""
    return np.where(x[:, 0] > x[:, 1], "A", "B")


def first_crossing_times(traj: np.ndarray, times: np.ndarray, cohort_mask: np.ndarray) -> np.ndarray:
    """First time (in label units) when x<y along each trajectory.

    traj: (S, N, 2). Returns per-particle label-time of first crossing or nan.
    """
    diff = traj[:, :, 0] - traj[:, :, 1]  # (S, N)
    out = np.full(traj.shape[1], np.nan)
    for i in np.where(cohort_mask)[0]:
        below = np.where(diff[:, i] < 0)[0]
        if len(below) == 0:
            continue
        j = below[0]
        if j == 0:
            out[i] = times[0]
        else:
            d0, d1 = diff[j - 1, i], diff[j, i]
            frac = d0 / max(d0 - d1, 1e-12)
            out[i] = times[j - 1] + frac * (times[j] - times[j - 1])
    return out


def eval_toggle_metrics(
    flow_net: torch.nn.Module,
    cfg: dict[str, Any],
    frames: dict[Any, np.ndarray],
    labels: list,
    n_ode_steps: int,
    device: str,
) -> dict[str, Any]:
    gt = load_toggle_gt(cfg)
    sorted_labels = sorted(labels)
    label_to_t = {label: labels_to_timesteps(sorted_labels)[i] for i, label in enumerate(sorted_labels)}
    t0, t1 = label_to_t[sorted_labels[0]], label_to_t[sorted_labels[-1]]

    gt_times = np.asarray(gt["times"], dtype=np.float64)  # physical times per label
    label_duration = float(gt_times[-1] - gt_times[0]) if len(gt_times) > 1 else 1.0

    source = torch.from_numpy(frames[sorted_labels[0]]).float()
    traj = _run_ode_rollout(flow_net, source, t0, t1, n_ode_steps, device)
    traj_np = traj.detach().cpu().numpy()  # (S, N, 2)
    times_label = np.linspace(t0, t1, traj_np.shape[0])

    step_idx = [int(np.argmin(np.abs(times_label - label_to_t[l]))) for l in sorted_labels]
    pred_at_labels = traj_np[step_idx]  # (L, N, 2)
    gt_at_labels = np.stack([frames[l] for l in sorted_labels], axis=0)

    init_basin = np.asarray(gt["init_basin"], dtype=object)
    basin_by_time = np.asarray(gt["basin_labels_by_time"], dtype=object)
    switch_true_phys = np.asarray(gt["switch_times"], dtype=np.float64)

    out: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Basin accuracy per snapshot
    # ------------------------------------------------------------------
    per_snap = []
    for li, label in enumerate(sorted_labels):
        gt_b = basin_by_time[label]
        mask = (gt_b == "A") | (gt_b == "B")
        pred_b = basin_of_points(pred_at_labels[li])
        acc = float((pred_b[mask] == gt_b[mask]).mean()) if mask.any() else float("nan")
        per_snap.append({"label": int(label), "t_phys": float(gt_times[label]), "basin_acc": acc, "n": int(mask.sum())})
    out["basin_accuracy_by_time"] = per_snap
    out["basin_accuracy_terminal"] = per_snap[-1]["basin_acc"]
    out["basin_accuracy_mean"] = float(np.nanmean([s["basin_acc"] for s in per_snap]))

    # ------------------------------------------------------------------
    # 2. Switch-time error (init-A cohort, physical units)
    # ------------------------------------------------------------------
    cohort_a = init_basin == "A"
    pred_switch_label = first_crossing_times(traj_np, times_label, cohort_a)
    pred_switch_phys = pred_switch_label * label_duration  # label grid spans [0,1]
    true_a = cohort_a & ~np.isnan(switch_true_phys)
    both = true_a & ~np.isnan(pred_switch_phys)
    missed = true_a & np.isnan(pred_switch_phys)
    out["switch_time"] = {
        "n_true_switchers": int(true_a.sum()),
        "n_missed": int(missed.sum()),
        "missed_rate": float(missed.sum() / max(true_a.sum(), 1)),
        "mae": float(np.abs(pred_switch_phys[both] - switch_true_phys[both]).mean()) if both.any() else float("nan"),
        "median_ae": float(np.median(np.abs(pred_switch_phys[both] - switch_true_phys[both]))) if both.any() else float("nan"),
        "gt_median": float(np.nanmedian(switch_true_phys)),
        "pred_median": float(np.nanmedian(pred_switch_phys[true_a])) if (~np.isnan(pred_switch_phys[true_a])).any() else float("nan"),
    }

    # ------------------------------------------------------------------
    # 3. Spurious transitions (init-B entering x>y) / early switches (init-A)
    # ------------------------------------------------------------------
    cohort_b = init_basin == "B"
    b_diff = traj_np[:, :, 0] - traj_np[:, :, 1]
    spurious_b = np.zeros(traj_np.shape[1], dtype=bool)
    for i in np.where(cohort_b)[0]:
        # predicted to ever sit strictly on the A side after the initial phase
        spurious_b[i] = bool((b_diff[1:, i] > 0).any())
    early_a = np.zeros(traj_np.shape[1], dtype=bool)
    for i in np.where(cohort_a & ~np.isnan(switch_true_phys))[0]:
        if not np.isnan(pred_switch_phys[i]) and pred_switch_phys[i] < switch_true_phys[i] - 2.5:
            early_a[i] = True
    out["spurious"] = {
        "initB_enter_A_rate": float(spurious_b[cohort_b].mean()) if cohort_b.any() else float("nan"),
        "n_initB_spurious": int(spurious_b[cohort_b].sum()),
        "initA_early_switch_rate": float(early_a[cohort_a].mean()) if cohort_a.any() else float("nan"),
        "n_initA_early": int(early_a[cohort_a].sum()),
    }

    # ------------------------------------------------------------------
    # 4. Velocity cosine vs exact analytic drift
    # ------------------------------------------------------------------
    gt_vel = np.asarray(gt["velocity"], dtype=np.float64)  # (T_all, N, 2)
    vel_rows = []
    for li, label in enumerate(sorted_labels):
        pts = torch.from_numpy(gt_at_labels[li]).float().to(device)
        tval = torch.full((pts.shape[0],), label_to_t[label], dtype=torch.float32, device=device)
        with torch.no_grad():
            v_pred = flow_net(tval, pts).cpu().numpy().astype(np.float64)
        v_true = gt_vel[label] if gt_vel.shape[0] > label else gt_vel[li]
        denom = (np.linalg.norm(v_pred, axis=-1) * np.linalg.norm(v_true, axis=-1) + 1e-12)
        cos = float(((v_pred * v_true).sum(-1) / denom).mean())
        vel_rows.append({"label": int(label), "cosine": cos})
    out["velocity_cosine_by_time"] = vel_rows
    out["velocity_cosine_mean"] = float(np.mean([r["cosine"] for r in vel_rows]))

    # ------------------------------------------------------------------
    # 5. ADE / FDE
    # ------------------------------------------------------------------
    dists = np.linalg.norm(pred_at_labels - gt_at_labels, axis=-1)
    out["ade"] = {
        "ade": float(dists.mean()),
        "fde": float(dists[-1].mean()),
        "median_ade": float(np.median(dists.mean(axis=0))),
    }
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    dm = make_datamodule(cfg)
    dm.setup()

    ckpt_path = infer_stage2_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No stage-2 checkpoint found in {run_dir}")
    model = build_velocity_model(cfg, ckpt_path, args.device)

    frames = {
        **{l: t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t) for l, t in dm.selected_train_frames.items()},
        **{l: t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t) for l, t in dm.test_frames.items()},
    }
    labels = sorted(frames.keys())

    output: dict[str, Any] = {
        "run_dir": str(run_dir),
        "data_name": cfg.get("data_name"),
        "n_ode_steps": args.n_ode_steps,
    }
    output.update(eval_toggle_metrics(model, cfg, frames, labels, args.n_ode_steps, args.device))
    output["w2"] = compute_w2_terminal(model, frames, labels, args.n_ode_steps, args.device)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "toggle_correctness.json", "w") as f:
        json.dump(output, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log.info("Saved %s", output_dir / "toggle_correctness.json")
    print(
        f"basin_acc(term)={output['basin_accuracy_terminal']:.4f}  "
        f"switch_mae={output['switch_time']['mae']:.3f}  "
        f"missed={output['switch_time']['missed_rate']:.3f}  "
        f"spuriousB={output['spurious']['initB_enter_A_rate']:.3f}  "
        f"vel_cos={output['velocity_cosine_mean']:.4f}  "
        f"ADE={output['ade']['ade']:.4f}"
    )


if __name__ == "__main__":
    main()
