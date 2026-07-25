#!/usr/bin/env python
"""Evaluate a trained PACE model on the SphereRot synthetic benchmark.

Computes the synthetic GT metrics from the rebuttal protocol:
    - Stage-2 path L2 / geodesic error against true S^2 rotation
    - Endpoint W2 and MMD (also available from distribution_metrics.csv)
    - Learned metric recovery error (Frobenius, tangent block, normal eigenvalue)
    - Tangent-normal velocity ratio using the true velocity field
    - Condition number of the learned metric

Usage
-----
    python scripts/eval_sphere_rot.py results/sphere_rot/pace
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
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

from src.dataloaders.sphere_rot_data import SphereRotDataModule, true_sphere_metric, true_sphere_velocity
from src.methods.pace.geometry import (
    build_anchor_normal_bank,
    estimate_segment_geometric_bandwidths,
    evaluate_metric_field,
)
from src.methods.pace.stage2.eval_metrics import _run_ode_rollout, compute_distribution_metrics
from src.methods.pace.stage2.flow_matcher import labels_to_timesteps
from src.methods.pace.stage2.networks import VelocityNet

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PACE on SphereRot")
    parser.add_argument("run_dir", type=str, help="PACE run directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <run-dir>/metrics.",
    )
    parser.add_argument(
        "--n-query",
        type=int,
        default=2048,
        help="Number of query points for metric recovery.",
    )
    parser.add_argument(
        "--n-ode-steps",
        type=int,
        default=101,
        help="ODE steps for path-error evaluation.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def make_datamodule(cfg: dict[str, Any]) -> SphereRotDataModule:
    args = SimpleNamespace(**{**cfg, "data_path": str(REPO_ROOT / "data" / "sphere_rot_dummy.npz")})
    return SphereRotDataModule(args)


def infer_checkpoint(run_dir: Path, stage: str) -> Path | None:
    ckpt_dir = run_dir / "checkpoints" / stage
    if not ckpt_dir.exists():
        return None
    candidates = list(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        return None
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
    model = VelocityNet(dim=dim, hidden_dims=hidden_dims, activation=activation, batch_norm=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state = ckpt.get("state_dict", ckpt)
    state = _remap_state_dict(raw_state, set(model.state_dict().keys()))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("Missing velocity keys: %s", missing[:5])
    if unexpected:
        log.warning("Unexpected velocity keys: %s", unexpected[:5])
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


def _geodesic_distance_s2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise geodesic distance on S^2."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    cos_angle = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.arccos(np.abs(cos_angle))  # use abs to avoid numerical issues near pi


def compute_path_error(
    flow_net: torch.nn.Module,
    train_frames: dict,
    train_labels: list,
    omega: float,
    n_ode_steps: int,
    device: str,
) -> dict[str, float]:
    """Compute path-L2 and endpoint distribution error for Stage-2 rollouts."""
    sorted_train = sorted(train_labels)
    label_to_t = {
        label: labels_to_timesteps(sorted_train)[i]
        for i, label in enumerate(sorted_train)
    }

    geodesic_errors = []
    ambient_errors = []
    endpoint_rows = []

    # We know the true mapping is rigid rotation; seeds are shared across frames
    # because SphereRotDataModule uses the same seed set.  Use frame 0 as source.
    source_label = sorted_train[0]
    source = train_frames[source_label].detach().float().cpu().numpy()

    for target_label in sorted_train[1:]:
        traj = _run_ode_rollout(
            flow_net=flow_net,
            source=train_frames[source_label],
            t_source=label_to_t[source_label],
            t_target=label_to_t[target_label],
            n_steps=n_ode_steps,
            device=device,
        )
        traj_np = traj.detach().cpu().numpy()  # (n_steps, N, 3)

        # True rotated positions.
        theta = omega * (target_label - source_label) / max(sorted_train[-1] - sorted_train[0], 1)
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        true_target = source @ R.T

        # Path error: average geodesic and ambient distance along the trajectory.
        for step, pred_step in enumerate(traj_np):
            step_theta = theta * step / max(n_ode_steps - 1, 1)
            R_step = np.array(
                [
                    [np.cos(step_theta), -np.sin(step_theta), 0.0],
                    [np.sin(step_theta), np.cos(step_theta), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            true_step = source @ R_step.T
            geodesic_errors.append(float(_geodesic_distance_s2(pred_step, true_step).mean()))
            ambient_errors.append(float(np.linalg.norm(pred_step - true_step, axis=-1).mean()))

        # Endpoint distributional metrics.
        pred_target = traj_np[-1]
        endpoint_metrics = compute_distribution_metrics(
            torch.from_numpy(pred_target), torch.from_numpy(true_target)
        )
        endpoint_rows.append(
            {
                "target_label": target_label,
                **{k: float(v) for k, v in endpoint_metrics.items()},
            }
        )

    return {
        "mean_geodesic_error": float(np.mean(geodesic_errors)) if geodesic_errors else 0.0,
        "mean_ambient_error": float(np.mean(ambient_errors)) if ambient_errors else 0.0,
        "endpoint_metrics": endpoint_rows,
    }


def compute_metric_recovery(
    train_frames: dict,
    train_labels: list,
    alpha_learned: float,
    alpha_true: float,
    k_neighbors: int,
    sigma_local: float | None,
    geometry_window_segments: int,
    n_query: int,
    device: str,
) -> dict[str, float]:
    """Compare the learned metric field with the ground-truth sphere metric."""
    sorted_train = sorted(train_labels)
    train_idx = sorted_train
    anchors_list = [train_frames[l].detach().float() for l in sorted_train]
    min_n = min(a.shape[0] for a in anchors_list)
    train_anchors = torch.stack([a[:min_n] for a in anchors_list], dim=0).to(device)

    normal_bank = build_anchor_normal_bank(
        train_anchors=train_anchors,
        train_idx=train_idx,
        k_neighbors=k_neighbors,
        sigma=sigma_local,
    )
    segment_bandwidths = estimate_segment_geometric_bandwidths(
        train_idx=train_idx,
        normal_bank=normal_bank,
        k_neighbors=k_neighbors,
        geometry_window_segments=geometry_window_segments,
    )

    # Use the middle segment's bandwidths for evaluation.
    seg_params = segment_bandwidths[len(segment_bandwidths) // 2]

    # Query points: uniform samples on S^2 plus train points.
    rng = np.random.default_rng(123)
    query_uniform = rng.standard_normal((n_query, 3)).astype(np.float32)
    query_uniform = query_uniform / (np.linalg.norm(query_uniform, axis=1, keepdims=True) + 1e-8)
    query_points = np.concatenate([train_anchors.reshape(-1, 3).cpu().numpy(), query_uniform], axis=0)
    query_tensor = torch.from_numpy(query_points).to(device)
    t_query = torch.full((query_tensor.shape[0], 1), float(train_idx[0]), device=device)

    G_pred, C_N, _ = evaluate_metric_field(
        query_tensor,
        t_query,
        normal_bank,
        h_x=seg_params["metric_hx"],
        h_t=seg_params["metric_ht"],
        alpha=alpha_learned,
    )
    G_true = true_sphere_metric(query_tensor, alpha_true)

    # Frobenius recovery error.
    frob_error = float(
        torch.norm(G_pred - G_true, dim=(-2, -1)).mean()
        / torch.norm(G_true, dim=(-2, -1)).mean()
    )

    # Tangent block error: P_T G_pred P_T should be close to P_T.
    P_N_true = torch.einsum("ni,nj->nij", query_tensor, query_tensor)
    P_T_true = torch.eye(3, device=device).unsqueeze(0) - P_N_true
    tangent_block_error = float(
        torch.norm(
            torch.einsum("...ij,...jk,...kl->...il", P_T_true, G_pred, P_T_true) - P_T_true,
            dim=(-2, -1),
        ).mean()
    )

    # Normal eigenvalue error.
    normal_eig = torch.einsum("...i,...ij,...j->...", query_tensor, G_pred, query_tensor)
    normal_eig_error = float(
        torch.abs(normal_eig - (1.0 + alpha_true)).mean() / (1.0 + alpha_true)
    )

    # Condition number.
    eigvals = torch.linalg.eigvalsh(G_pred)
    condition_numbers = eigvals.max(dim=-1).values / (eigvals.min(dim=-1).values + 1e-12)

    return {
        "frobenius_relative_error": frob_error,
        "tangent_block_error": tangent_block_error,
        "normal_eigenvalue_error": normal_eig_error,
        "mean_condition_number": float(condition_numbers.mean()),
        "median_condition_number": float(condition_numbers.median()),
        "metric_hx": seg_params["metric_hx"],
        "metric_ht": seg_params["metric_ht"],
    }


def compute_velocity_ratio(
    train_frames: dict,
    train_labels: list,
    alpha_learned: float,
    omega: float,
    k_neighbors: int,
    sigma_local: float | None,
    geometry_window_segments: int,
    device: str,
) -> dict[str, float]:
    """Compute tangent-normal velocity ratio for the true velocity under G_pred."""
    sorted_train = sorted(train_labels)
    train_idx = sorted_train
    anchors_list = [train_frames[l].detach().float() for l in sorted_train]
    min_n = min(a.shape[0] for a in anchors_list)
    train_anchors = torch.stack([a[:min_n] for a in anchors_list], dim=0).to(device)

    normal_bank = build_anchor_normal_bank(
        train_anchors=train_anchors,
        train_idx=train_idx,
        k_neighbors=k_neighbors,
        sigma=sigma_local,
    )
    segment_bandwidths = estimate_segment_geometric_bandwidths(
        train_idx=train_idx,
        normal_bank=normal_bank,
        k_neighbors=k_neighbors,
        geometry_window_segments=geometry_window_segments,
    )
    seg_params = segment_bandwidths[len(segment_bandwidths) // 2]

    query = train_anchors.reshape(-1, 3)
    t_query = torch.full((query.shape[0], 1), float(train_idx[0]), device=device)
    G_pred, C_N, _ = evaluate_metric_field(
        query,
        t_query,
        normal_bank,
        h_x=seg_params["metric_hx"],
        h_t=seg_params["metric_ht"],
        alpha=alpha_learned,
    )

    v_true = true_sphere_velocity(query, omega)
    # Extract normal projector from C_N (which is a convex combination of P_Ns).
    P_N_pred = C_N
    P_T_pred = torch.eye(3, device=device).unsqueeze(0) - P_N_pred

    v_normal = torch.einsum("...ij,...j->...i", P_N_pred, v_true)
    v_tangent = torch.einsum("...ij,...j->...i", P_T_pred, v_true)
    norm_normal = torch.norm(v_normal, dim=-1)
    norm_tangent = torch.norm(v_tangent, dim=-1)
    ratio = (norm_normal / (norm_tangent + 1e-8)).mean()

    # Also report cosine similarity with true velocity (independent of metric).
    v_true_norm = torch.nn.functional.normalize(v_true, dim=-1, eps=1e-12)
    # Geodesic tangent direction is e_z × x / sin(theta); just compare to cross product.
    tangent_dir = torch.cross(
        torch.tensor([0.0, 0.0, 1.0], device=device, dtype=query.dtype).expand_as(query),
        query,
        dim=-1,
    )
    tangent_dir = torch.nn.functional.normalize(tangent_dir, dim=-1, eps=1e-12)
    cosine = (v_true_norm * tangent_dir).sum(dim=-1).mean()

    return {
        "tangent_normal_velocity_ratio": float(ratio),
        "true_velocity_tangent_cosine": float(cosine),
    }


def read_distribution_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics" / "distribution_metrics.csv"
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    dm = make_datamodule(cfg)
    dm.setup()

    train_frames = (
        dm.selected_train_frames
        if hasattr(dm, "selected_train_frames")
        else dm.train_frames
    )
    train_labels = sorted(dm.unique_train_labels)

    output: dict[str, Any] = {
        "run_dir": str(run_dir),
        "data_name": cfg.get("data_name"),
        "alpha_learned": float(cfg.get("metric_alpha", 8.0)),
        "alpha_true": float(cfg.get("sphere_alpha_true", 8.0)),
        "omega": float(cfg.get("sphere_omega", 0.5 * math.pi)),
    }

    # Path and endpoint errors from Stage-2 velocity field.
    ckpt_path = infer_checkpoint(run_dir, "stage2_flow")
    if ckpt_path is not None:
        log.info("Loading velocity checkpoint %s", ckpt_path)
        model = build_velocity_model(cfg, ckpt_path, args.device)
        output["path_error"] = compute_path_error(
            flow_net=model,
            train_frames=train_frames,
            train_labels=train_labels,
            omega=output["omega"],
            n_ode_steps=args.n_ode_steps,
            device=args.device,
        )
    else:
        log.warning("No Stage-2 checkpoint found; skipping path error.")

    # Metric recovery and velocity ratio from the learned normal bank.
    output["metric_recovery"] = compute_metric_recovery(
        train_frames=train_frames,
        train_labels=train_labels,
        alpha_learned=output["alpha_learned"],
        alpha_true=output["alpha_true"],
        k_neighbors=int(cfg.get("k_neighbors_local", 25)),
        sigma_local=cfg.get("sigma_local", None),
        geometry_window_segments=int(cfg.get("geometry_window_segments", 2)),
        n_query=args.n_query,
        device=args.device,
    )
    output["velocity_ratio"] = compute_velocity_ratio(
        train_frames=train_frames,
        train_labels=train_labels,
        alpha_learned=output["alpha_learned"],
        omega=output["omega"],
        k_neighbors=int(cfg.get("k_neighbors_local", 25)),
        sigma_local=cfg.get("sigma_local", None),
        geometry_window_segments=int(cfg.get("geometry_window_segments", 2)),
        device=args.device,
    )

    # Distributional metrics already computed by the runner.
    output["distribution_metrics"] = read_distribution_metrics(run_dir)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sphere_rot_metrics.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("SphereRot metrics saved to %s", output_path)


if __name__ == "__main__":
    main()
