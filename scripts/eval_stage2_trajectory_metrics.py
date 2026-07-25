#!/usr/bin/env python
"""Evaluate Stage-2 ODE rollout against ground-truth trajectory labels.

This script computes the Stage-2 metrics from the GT protocol:
    - clone/fate probability error over time (top-1, Brier, CE, TV, JS, ECE)
    - binary-fate AUC-ROC / average precision (LARRY Mo/Neu)
    - branch accuracy (clone -> terminal fate/cell-type)
    - pseudotime correlation (Spearman / Pearson / MAE)

Usage
-----
    python scripts/eval_stage2_trajectory_metrics.py \\
        results/larry_pca2_dim2_test1/pace \\
        --output-dir results/larry_pca2_dim2_test1/pace/metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.pace.stage2.eval_metrics import (
    evaluate_branch_accuracy,
    evaluate_cell_fate_error,
    evaluate_pseudotime_correlation,
    evaluate_stage2_fate_probability_over_time,
    evaluate_stage2_label_probability_over_time,
)
from src.methods.pace.stage2.networks import VelocityNet

log = logging.getLogger(__name__)


SUPPORTED_METHODS = {"pace"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Stage-2 ground-truth trajectory metrics for PACE."
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="PACE run directory containing resolved_config.yaml and checkpoints/stage2_flow.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write JSON results. Defaults to <run-dir>/metrics.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of neighbors for kNN label probability estimation.",
    )
    parser.add_argument(
        "--n-ode-steps",
        type=int,
        default=101,
        help="Number of ODE steps for rollout.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def as_namespace(data: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def make_datamodule(cfg: dict[str, Any]):
    data_name = str(cfg.get("data_name", ""))
    args = as_namespace({**cfg, "data_path": str(resolve_data_path(cfg))})
    if data_name == "eb_phate":
        from src.dataloaders.eb_data import EBDataModule
        return EBDataModule(args)
    if data_name in ("larry_pca2", "morris_celltag") or "larry" in data_name or "morris" in data_name:
        from src.dataloaders.larry_data import LARRYDataModule
        return LARRYDataModule(args)
    raise ValueError(f"Unsupported data_name={data_name!r}")


def resolve_data_path(cfg: dict[str, Any]) -> Path:
    data_path = Path(str(cfg["data_path"]))
    if data_path.is_absolute():
        return data_path
    return REPO_ROOT / data_path


def infer_checkpoint(run_dir: Path) -> Path:
    candidates = list((run_dir / "checkpoints" / "stage2_flow").glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}/checkpoints/stage2_flow")
    for ckpt in candidates:
        if ckpt.name == "last.ckpt":
            return ckpt

    def sort_key(path: Path) -> tuple[int, int, float]:
        match = re.search(r"epoch=(\d+)", path.name)
        epoch = int(match.group(1)) if match else -1
        version_match = re.search(r"-v(\d+)\.ckpt$", path.name)
        version = int(version_match.group(1)) if version_match else 0
        return epoch, version, path.stat().st_mtime

    return max(candidates, key=sort_key)


def build_velocity_model(cfg: dict[str, Any], ckpt_path: Path, device: str) -> torch.nn.Module:
    dim = int(cfg["dim"])
    hidden_dims = list(cfg.get("hidden_dims_flow", [64, 64, 64]))
    activation = str(cfg.get("activation_flow", "selu"))

    model = VelocityNet(
        dim=dim,
        hidden_dims=hidden_dims,
        activation=activation,
        batch_norm=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state = ckpt.get("state_dict", ckpt)
    state = _remap_state_dict(raw_state, set(model.state_dict().keys()))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_non_buffers = [key for key in missing if not key.endswith("num_batches_tracked")]
    if missing_non_buffers:
        raise RuntimeError(f"Could not load {ckpt_path}; missing keys include {missing_non_buffers[:5]}")
    if unexpected:
        log.warning("Ignored unexpected checkpoint keys: %s", unexpected[:5])
    model.to(device)
    model.eval()
    return model


def _remap_state_dict(raw_state: dict[str, torch.Tensor], target_keys: set[str]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for target_key in target_keys:
        if target_key in raw_state:
            remapped[target_key] = raw_state[target_key]
            continue
        suffix = "." + target_key
        matches = [key for key in raw_state if key.endswith(suffix)]
        if matches:
            remapped[target_key] = raw_state[sorted(matches, key=len)[0]]
    return remapped


def load_raw_label_frames(
    cfg: dict[str, Any],
    dm,
    label_key: str,
) -> dict[Any, np.ndarray]:
    """Load a per-timepoint label array, respecting the datamodule's subsampling.

    This mirrors ``build_per_timepoint_arrays`` in ``eval_larry_barcode.py`` but
    uses the datamodule's selected indices so the label arrays line up with the
    train/test point clouds used by PACE.
    """
    data_path = resolve_data_path(cfg)
    data = np.load(data_path, allow_pickle=True)
    if label_key not in data:
        return {}

    raw_labels = data[label_key]
    raw_timepoints = data[str(cfg.get("timepoint_key", "timepoints"))]

    # Remap string time labels exactly as the datamodule does.
    unique_labels = np.unique(raw_timepoints)
    if all(isinstance(l, (int, float, np.integer, np.floating)) for l in unique_labels):
        label_to_num = {label: label for label in unique_labels}
    else:
        sorted_labels = sorted(unique_labels, key=_label_sort_key)
        label_to_num = {label: idx for idx, label in enumerate(sorted_labels)}
    numeric_timepoints = np.array([label_to_num[label] for label in raw_timepoints])

    selected_indices = getattr(dm, "selected_train_indices", {})
    test_indices = getattr(dm, "test_indices", {})

    label_frames: dict[Any, np.ndarray] = {}
    for label in sorted(label_to_num.values()):
        mask = numeric_timepoints == label
        idx = np.where(mask)[0]
        if label in selected_indices:
            idx = idx[selected_indices[label]]
        elif label in test_indices:
            idx = idx[test_indices[label]]
        label_frames[label] = raw_labels[idx]
    return label_frames


def _label_sort_key(value: Any):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, float(value))
    text = str(value)
    m = re.search(r"(\d+(?:\.\d+)?)$", text)
    if m:
        return (0, float(m.group(1)))
    return (1, text)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    method = cfg.get("method", {}).get("name", "pace")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")

    dm = make_datamodule(cfg)
    dm.setup()

    ckpt_path = infer_checkpoint(run_dir)
    log.info("Loading checkpoint from %s", ckpt_path)
    model = build_velocity_model(cfg, ckpt_path, args.device)

    train_frames = (
        dm.selected_train_frames
        if hasattr(dm, "selected_train_frames")
        else dm.train_frames
    )
    test_frames = dm.test_frames
    train_labels = sorted(dm.unique_train_labels)
    test_labels = sorted(dm.unique_test_labels)

    output: dict[str, Any] = {
        "run_dir": str(run_dir),
        "data_name": cfg.get("data_name"),
        "k": args.k,
        "n_ode_steps": args.n_ode_steps,
    }

    # ------------------------------------------------------------------
    # Clone probability error over time
    # ------------------------------------------------------------------
    clone_id_key = str(cfg.get("clone_id_key", "clone_ids"))
    clone_frames = load_raw_label_frames(cfg, dm, clone_id_key)
    if clone_frames:
        log.info("Evaluating clone probability error...")
        output["clone_probability"] = evaluate_stage2_label_probability_over_time(
            flow_net=model,
            train_frames=train_frames,
            test_frames=test_frames,
            train_labels=train_labels,
            test_labels=test_labels,
            label_frames=clone_frames,
            label_name="clone",
            k=args.k,
            n_ode_steps=args.n_ode_steps,
            device=args.device,
        )

    # ------------------------------------------------------------------
    # Fate / cell-type probability error over time
    # ------------------------------------------------------------------
    fate_label_key = str(cfg.get("fate_label_key", "fate_labels"))
    fate_frames = load_raw_label_frames(cfg, dm, fate_label_key)
    if fate_frames:
        log.info("Evaluating fate probability error...")
        output["fate_probability"] = evaluate_stage2_fate_probability_over_time(
            flow_net=model,
            train_frames=train_frames,
            test_frames=test_frames,
            train_labels=train_labels,
            test_labels=test_labels,
            fate_frames=fate_frames,
            k=args.k,
            n_ode_steps=args.n_ode_steps,
            device=args.device,
        )
        output["cell_fate_error"] = evaluate_cell_fate_error(
            flow_net=model,
            train_frames=train_frames,
            test_frames=test_frames,
            train_labels=train_labels,
            test_labels=test_labels,
            fate_frames=fate_frames,
            k=args.k,
            n_ode_steps=args.n_ode_steps,
            device=args.device,
        )

    # ------------------------------------------------------------------
    # Branch accuracy (clone -> terminal branch)
    # ------------------------------------------------------------------
    if clone_frames and fate_frames:
        log.info("Evaluating branch accuracy...")
        source_label = min(train_labels)
        terminal_label = max(train_labels)
        output["branch_accuracy"] = evaluate_branch_accuracy(
            flow_net=model,
            frames=train_frames,
            labels=train_labels,
            source_label=source_label,
            terminal_label=terminal_label,
            branch_label_frames=fate_frames,
            clone_label_frames=clone_frames,
            n_ode_steps=args.n_ode_steps,
            device=args.device,
        )

    # ------------------------------------------------------------------
    # Pseudotime correlation
    # ------------------------------------------------------------------
    log.info("Evaluating pseudotime correlation...")
    output["pseudotime_correlation"] = evaluate_pseudotime_correlation(
        flow_net=model,
        frames={**train_frames, **test_frames},
        labels=sorted(set(train_labels) | set(test_labels)),
        source_label=min(train_labels),
        target_labels=sorted(set(train_labels) | set(test_labels)),
        n_ode_steps=args.n_ode_steps,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stage2_trajectory_metrics.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Stage-2 trajectory metrics saved to %s", output_path)


if __name__ == "__main__":
    main()
