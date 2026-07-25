#!/usr/bin/env python
"""Unified trajectory-correctness evaluator for PACE.

This script implements the four main output-level metrics from the rebuttal
experiment design:

    1. ADE  (Average Displacement Error)  — whole-trajectory correctness
    2. Branch / clone-compatible error    — does the endpoint land in the right basin?
    3. Oracle normal-motion fraction      — how much velocity leaves the true tangent space?
    4. W2                                 — held-out marginal reconstruction

These metrics can be computed for any deterministic trajectory method, not just
PACE, as long as the same source particles, time grid, oracle geometry and W2
evaluator are used.

Usage
-----
    python scripts/eval_trajectory_correctness.py \\
        results/sphere_rot/pace_approx \\
        --device cuda

The script reads ``resolved_config.yaml`` and the Stage-2 checkpoint, then writes
``trajectory_correctness.json`` under ``<run-dir>/metrics``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree as scipy_spatial_cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.pace.geometry import estimate_normal_projector
from src.methods.pace.stage2.eval_metrics import _run_ode_rollout, compute_distribution_metrics
from src.methods.pace.stage2.flow_matcher import labels_to_timesteps
from src.methods.pace.stage2.networks import VelocityNet

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute unified trajectory-correctness metrics for PACE."
    )
    parser.add_argument("run_dir", type=str, help="PACE run directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <run-dir>/metrics.",
    )
    parser.add_argument(
        "--n-ode-steps",
        type=int,
        default=101,
        help="ODE steps for rollout.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model inference.",
    )
    parser.add_argument(
        "--reference-k",
        type=int,
        default=25,
        help="k neighbors for reference local-PCA normal estimation on real data.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers: config / data / model
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def resolve_data_path(cfg: dict[str, Any]) -> Path:
    data_path = Path(str(cfg["data_path"]))
    if data_path.is_absolute():
        return data_path
    return REPO_ROOT / data_path


def make_datamodule(cfg: dict[str, Any]):
    """Build the same datamodule used by the iMFM runner."""
    data_name = str(cfg.get("data_name", "")).lower()
    args = SimpleNamespace(**{**cfg, "data_path": str(resolve_data_path(cfg))})
    if "eb" in data_name:
        from src.dataloaders.eb_data import EBDataModule
        return EBDataModule(args)
    if "sphere_rot" in data_name:
        from src.dataloaders.sphere_rot_data import SphereRotDataModule
        return SphereRotDataModule(args)
    if "curved_pitchfork" in data_name:
        from src.dataloaders.curved_pitchfork_data import CurvedPitchforkDataModule
        return CurvedPitchforkDataModule(args)
    if "toggle_switch" in data_name or "toggle" in data_name:
        from src.dataloaders.toggle_switch_data import ToggleSwitchDataModule
        return ToggleSwitchDataModule(args)
    if "larry" in data_name or "hematopoiesis" in data_name or "morris" in data_name or "celltag" in data_name:
        from src.dataloaders.larry_data import LARRYDataModule
        return LARRYDataModule(args)
    if "larray" in data_name or "spring" in data_name:
        from src.dataloaders.larray_data import LArrayDataModule
        return LArrayDataModule(args)
    if "mouse" in data_name or "hematopoiesis" in data_name:
        from src.dataloaders.mouse_hematopoiesis_data import MouseHematopoiesisDataModule
        return MouseHematopoiesisDataModule(args)
    if "oceans" in data_name:
        from src.dataloaders.oceans_data import OceansDataModule
        return OceansDataModule(args)
    if "s_curve" in data_name or "scurve" in data_name:
        from src.dataloaders.scurve_data import SCurveDataModule
        return SCurveDataModule(args)
    if "toy_bifurcation" in data_name or "two_branch" in data_name:
        from src.dataloaders.toy_bifurcation_data import ToyBifurcationDataModule
        return ToyBifurcationDataModule(args)
    if "ipsc" in data_name:
        from src.dataloaders.ipsc_data import IPSCDataModule
        return IPSCDataModule(args)
    if "multi" in data_name:
        from src.dataloaders.multi_data import MultiDataModule
        return MultiDataModule(args)
    if "cite" in data_name:
        from src.dataloaders.cite_data import CiteDataModule
        return CiteDataModule(args)
    if "wot" in data_name:
        from src.dataloaders.scnode_wot_data import ScNodeWOTDataModule
        return ScNodeWOTDataModule(args)
    if "petal" in data_name:
        from src.dataloaders.petal_data import PetalDataModule
        return PetalDataModule(args)
    raise ValueError(f"Unsupported data_name={data_name!r}")


def infer_stage2_checkpoint(run_dir: Path) -> Path | None:
    for subdir in ("stage2_flow", "flow"):
        ckpt_dir = run_dir / "checkpoints" / subdir
        if not ckpt_dir.exists():
            continue
        candidates = list(ckpt_dir.glob("*.ckpt"))
        if not candidates:
            continue
        for ckpt in candidates:
            if ckpt.name == "last.ckpt":
                return ckpt

        def sort_key(path: Path):
            match = re.search(r"epoch=(\d+)", path.name)
            epoch = int(match.group(1)) if match else -1
            version_match = re.search(r"-v(\d+)\.ckpt$", path.name)
            version = int(version_match.group(1)) if version_match else 0
            return epoch, version, path.stat().st_mtime

        return max(candidates, key=sort_key)
    return None


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


# ---------------------------------------------------------------------------
# Label / particle-ID loading
# ---------------------------------------------------------------------------


def load_per_label_array(
    cfg: dict[str, Any],
    dm,
    array_key: str,
) -> dict[Any, np.ndarray] | None:
    """Load a per-timepoint label/ID array respecting the datamodule subsampling."""
    data_path = resolve_data_path(cfg)
    data = np.load(data_path, allow_pickle=True)
    if array_key not in data or array_key == "":
        return None

    raw_array = data[array_key]
    raw_timepoints = data[str(cfg.get("timepoint_key", "timepoints"))]

    # Map time labels the same way the datamodule does.
    if hasattr(dm, "timepoint_label_map") and dm.timepoint_label_map:
        label_map = dm.timepoint_label_map
    else:
        unique = np.unique(raw_timepoints)
        if all(isinstance(l, (int, float, np.integer, np.floating)) for l in unique):
            label_map = {label: label for label in unique}
        else:
            sorted_labels = sorted(unique, key=lambda v: (0, float(re.search(r"(\d+(?:\.\d+)?)$", str(v)).group(1))) if re.search(r"(\d+(?:\.\d+)?)$", str(v)) else (1, str(v)))
            label_map = {label: idx for idx, label in enumerate(sorted_labels)}

    numeric = np.array([label_map[label] for label in raw_timepoints])

    selected_indices = getattr(dm, "selected_train_indices", {})
    test_indices = getattr(dm, "test_indices", {})

    out: dict[Any, np.ndarray] = {}
    for label in sorted(label_map.values()):
        mask = numeric == label
        idx = np.where(mask)[0]
        if label in selected_indices:
            idx = idx[selected_indices[label]]
        elif label in test_indices:
            idx = idx[test_indices[label]]
        out[label] = raw_array[idx]
    return out


def _valid_barcode_mask(clones: np.ndarray) -> np.ndarray:
    return np.array([str(c).startswith("clone_") for c in clones])


# ---------------------------------------------------------------------------
# Metric 1: ADE (Average Displacement Error)
# ---------------------------------------------------------------------------


def build_gt_trajectories(
    frames: dict[Any, np.ndarray],
    particle_id_frames: dict[Any, np.ndarray],
) -> dict[int, np.ndarray]:
    """Build dict particle_id -> (T, d) from aligned frames.

    This only works when the dataset preserves persistent particle IDs across
    timepoints (e.g., SphereRot seeds, Ocean simulator IDs).
    """
    labels = sorted(frames.keys())
    all_ids = np.unique(np.concatenate([particle_id_frames[l] for l in labels]))
    trajectories: dict[int, np.ndarray] = {}
    for pid in all_ids:
        pts = []
        for label in labels:
            idx = np.where(particle_id_frames[label] == pid)[0]
            if len(idx) == 0:
                break
            pts.append(frames[label][idx[0]])
        if len(pts) == len(labels):
            trajectories[int(pid)] = np.stack(pts)
    return trajectories


def build_predicted_trajectories(
    flow_net: torch.nn.Module,
    frames: dict[Any, np.ndarray],
    particle_id_frames: dict[Any, np.ndarray],
    labels: list,
    n_ode_steps: int,
    device: str,
) -> dict[int, np.ndarray]:
    """Roll from the earliest label to the latest and sample at all labels."""
    sorted_labels = sorted(labels)
    label_to_t = {label: labels_to_timesteps(sorted_labels)[i] for i, label in enumerate(sorted_labels)}
    source_label = sorted_labels[0]
    target_label = sorted_labels[-1]

    traj = _run_ode_rollout(
        flow_net=flow_net,
        source=torch.from_numpy(frames[source_label]).float(),
        t_source=label_to_t[source_label],
        t_target=label_to_t[target_label],
        n_steps=n_ode_steps,
        device=device,
    )
    traj_np = traj.detach().cpu().numpy()  # (n_steps, N_source, d)
    times = np.linspace(label_to_t[source_label], label_to_t[target_label], n_ode_steps)

    source_ids = particle_id_frames[source_label]
    pred: dict[int, np.ndarray] = {}
    for i, pid in enumerate(source_ids):
        pts = []
        for label in sorted_labels:
            step_idx = int(np.argmin(np.abs(times - label_to_t[label])))
            pts.append(traj_np[step_idx, i])
        pred[int(pid)] = np.stack(pts)
    return pred


def compute_ade(
    gt_trajectories: dict[int, np.ndarray],
    pred_trajectories: dict[int, np.ndarray],
    distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, float]:
    """Average Displacement Error over all particles and timepoints.

    ADE answers the most direct trajectory-correctness question:
    "Does the predicted path stay close to the ground-truth path at every step?"
    """
    common_ids = sorted(set(gt_trajectories.keys()) & set(pred_trajectories.keys()))
    if not common_ids:
        return {"ade": float("nan"), "fde": float("nan"), "n_particles": 0}

    per_particle_ade = []
    per_particle_fde = []
    for pid in common_ids:
        gt = gt_trajectories[pid]
        pred = pred_trajectories[pid]
        n = min(gt.shape[0], pred.shape[0])
        dists = distance_fn(gt[:n], pred[:n])
        per_particle_ade.append(float(dists.mean()))
        per_particle_fde.append(float(dists[-1]))

    return {
        "ade": float(np.mean(per_particle_ade)),
        "fde": float(np.mean(per_particle_fde)),
        "median_ade": float(np.median(per_particle_ade)),
        "n_particles": len(common_ids),
    }


# ---------------------------------------------------------------------------
# Metric 2: Branch / clone-compatible error
# ---------------------------------------------------------------------------


def compute_clone_compatible_endpoint_mass(
    source_positions: np.ndarray,
    source_clones: np.ndarray,
    pred_endpoints: np.ndarray,
    target_positions: np.ndarray,
    target_clones: np.ndarray,
) -> dict[str, float]:
    """Fraction of rolled source cells whose nearest target neighbor has the same clone.

    For datasets like LARRY/Morris, clone barcodes are the closest thing to
    lineage identity.  A high compatible mass means PACE pushes each source clone
    into the correct target clone basin.
    """
    valid = _valid_barcode_mask(source_clones)
    valid_target = _valid_barcode_mask(target_clones)
    if not valid.any() or not valid_target.any():
        return {"compatible_mass": float("nan"), "n_eval": 0}

    eval_idx = np.where(valid)[0]
    tgt_idx = np.where(valid_target)[0]
    tree = scipy_spatial_cKDTree(target_positions[tgt_idx])
    _, nn_idx = tree.query(pred_endpoints[eval_idx], k=1)
    hits = source_clones[eval_idx] == target_clones[tgt_idx[nn_idx]]
    return {
        "compatible_mass": float(hits.mean()),
        "n_eval": int(len(eval_idx)),
        "n_hits": int(hits.sum()),
        "n_target_barcoded": int(len(tgt_idx)),
    }


def compute_terminal_branch_consistency(
    source_labels: np.ndarray,
    pred_endpoints: np.ndarray,
    target_positions: np.ndarray,
    target_labels: np.ndarray,
) -> dict[str, float]:
    """Per-source-group terminal branch consistency.

    Groups source cells by their label (e.g. clone or early fate), rolls them to
    the terminal frame, and compares the majority predicted terminal label to the
    majority true terminal label of that same group.  This avoids requiring
    persistent cell IDs.
    """
    from collections import Counter

    if source_labels.size == 0 or target_labels.size == 0:
        return {"consistency": float("nan"), "n_groups": 0}

    tree = scipy_spatial_cKDTree(target_positions)
    _, nn_idx = tree.query(pred_endpoints, k=1)
    predicted_terminal = target_labels[nn_idx]

    correct = []
    sizes = []
    for group in np.unique(source_labels):
        mask = source_labels == group
        if mask.sum() == 0:
            continue
        true_terminal = Counter(target_labels).most_common(1)[0][0]
        pred_terminal = Counter(predicted_terminal[mask]).most_common(1)[0][0]
        correct.append(float(true_terminal == pred_terminal))
        sizes.append(int(mask.sum()))

    if not correct:
        return {"consistency": float("nan"), "n_groups": 0}
    return {
        "consistency": float(np.average(correct, weights=sizes)),
        "macro_consistency": float(np.mean(correct)),
        "n_groups": len(correct),
    }




def compute_clone_fate_consistency(
    source_clones: np.ndarray,
    source_fates: np.ndarray,
    pred_endpoints: np.ndarray,
    target_positions: np.ndarray,
    target_clones: np.ndarray,
    target_fates: np.ndarray,
    min_cells: int = 3,
) -> dict[str, float]:
    """Clone-conditioned terminal fate consistency (biological lineage metric).

    For each clone present at source and target, predict the terminal fate of
    its rolled cells by kNN-majority vote in the target frame, and compare to
    the clone's true majority fate at target.  Report size-weighted (micro)
    and unweighted (macro) accuracy.  Clones with fewer than ``min_cells``
    cells at source are skipped.
    """
    from collections import Counter

    source_clones = np.asarray(source_clones, dtype=object)
    target_fates = np.asarray(target_fates, dtype=object)
    if source_clones.size == 0 or target_fates.size == 0:
        return {"clone_fate_micro": float("nan"), "clone_fate_macro": float("nan"), "n_clones": 0}

    tree = scipy_spatial_cKDTree(target_positions)
    _, nn_idx = tree.query(pred_endpoints, k=1)
    pred_fates = target_fates[nn_idx]

    correct, sizes = [], []
    for clone in np.unique(source_clones):
        if str(clone).startswith("unassigned") or str(clone).lower() in ("nan", "none"):
            continue
        mask = source_clones == clone
        if mask.sum() < min_cells:
            continue
        # majority predicted fate among this clone's rolled cells
        pred_maj = Counter(pred_fates[mask].tolist()).most_common(1)[0][0]
        # the clone's true majority fate at the TARGET day (its lineage fate).
        tgt_c_mask = target_clones == clone
        if tgt_c_mask.sum() < min_cells:
            continue
        true_maj = Counter(target_fates[tgt_c_mask].tolist()).most_common(1)[0][0]
        correct.append(float(pred_maj == true_maj))
        sizes.append(int(mask.sum()))

    if not correct:
        return {"clone_fate_micro": float("nan"), "clone_fate_macro": float("nan"), "n_clones": 0}
    return {
        "clone_fate_micro": float(np.average(correct, weights=sizes)),
        "clone_fate_macro": float(np.mean(correct)),
        "n_clones": len(correct),
        "n_cells": int(sum(sizes)),
    }


# ---------------------------------------------------------------------------
# Metric 3: Oracle normal-motion fraction
# ---------------------------------------------------------------------------


def build_oracle_normal_projector(
    frames: dict[Any, np.ndarray],
    labels: list,
    k: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a reference normal projector from the union of train frames.

    For real data we do not have an analytic oracle, so we estimate a reference
    tangent/normal structure with local PCA on the empirical manifold.  All
    baselines then use the *same* reference projector, keeping the comparison fair.
    """
    sorted_labels = sorted(labels)
    union = np.concatenate([frames[l] for l in sorted_labels], axis=0)
    x = torch.from_numpy(union).float()
    P_N = estimate_normal_projector(x, k_neighbors=k)  # (N_union, d, d)
    P_N_np = P_N.cpu().numpy()
    tree = scipy_spatial_cKDTree(union)

    def projector(query: np.ndarray) -> np.ndarray:
        _, idx = tree.query(query, k=1)
        return P_N_np[idx]

    return projector


def compute_oracle_normal_fraction(
    flow_net: torch.nn.Module,
    frames: dict[Any, np.ndarray],
    labels: list,
    normal_projector_fn: Callable[[np.ndarray], np.ndarray],
    n_ode_steps: int,
    device: str,
) -> dict[str, float]:
    """Fraction of kinetic energy that lies in oracle normal directions.

    For each trajectory segment we query the learned velocity at the midpoint,
    project it onto the reference normal space, and accumulate normal vs total
    kinetic energy.  A low value means the model moves mostly along the true
    tangent geometry.
    """
    sorted_labels = sorted(labels)
    label_to_t = {label: labels_to_timesteps(sorted_labels)[i] for i, label in enumerate(sorted_labels)}

    total_energy = 0.0
    normal_energy = 0.0
    per_segment = []

    for l0, l1 in zip(sorted_labels[:-1], sorted_labels[1:]):
        source = torch.from_numpy(frames[l0]).float().to(device)
        t0 = label_to_t[l0]
        t1 = label_to_t[l1]
        traj = _run_ode_rollout(
            flow_net=flow_net,
            source=source,
            t_source=t0,
            t_target=t1,
            n_steps=n_ode_steps,
            device=device,
        )
        traj_np = traj.detach().cpu().numpy()

        # Midpoints and mid-times.
        x_mid = 0.5 * (traj_np[:-1] + traj_np[1:])  # (n_steps-1, N, d)
        times = np.linspace(t0, t1, n_ode_steps)
        dt = np.diff(times)[:, None]  # (n_steps-1, 1)
        t_mid = 0.5 * (times[:-1] + times[1:])

        # Query learned velocity at midpoints.
        n_steps, n_particles, dim = x_mid.shape
        x_flat = torch.from_numpy(x_mid.reshape(-1, dim)).float().to(device)
        t_flat = torch.from_numpy(np.repeat(t_mid, n_particles)).float().to(device)
        with torch.no_grad():
            v_flat = flow_net(t_flat, x_flat).cpu().numpy()
        v = v_flat.reshape(n_steps, n_particles, dim)

        # Reference normal projector at midpoints.
        x_mid_flat = x_mid.reshape(-1, dim)
        P_N = normal_projector_fn(x_mid_flat).reshape(n_steps, n_particles, dim, dim)

        v_norm_sq = np.sum(v * v, axis=-1)  # (n_steps, N)
        v_normal = np.einsum("snij,snj->sni", P_N, v)
        v_normal_norm_sq = np.sum(v_normal * v_normal, axis=-1)

        segment_total = float(np.sum(dt * v_norm_sq))
        segment_normal = float(np.sum(dt * v_normal_norm_sq))
        total_energy += segment_total
        normal_energy += segment_normal
        if segment_total > 0:
            per_segment.append(
                {
                    "source_label": l0,
                    "target_label": l1,
                    "normal_fraction": segment_normal / segment_total,
                }
            )

    global_fraction = normal_energy / (total_energy + 1e-12)
    return {
        "global_normal_fraction": float(global_fraction),
        "total_energy": float(total_energy),
        "normal_energy": float(normal_energy),
        "per_segment": per_segment,
    }


# ---------------------------------------------------------------------------
# Metric 4: W2
# ---------------------------------------------------------------------------


def compute_w2_terminal(
    flow_net: torch.nn.Module,
    frames: dict[Any, np.ndarray],
    labels: list,
    n_ode_steps: int,
    device: str,
) -> list[dict[str, float]]:
    """W2 between predicted and true distributions at every target label."""
    sorted_labels = sorted(labels)
    label_to_t = {label: labels_to_timesteps(sorted_labels)[i] for i, label in enumerate(sorted_labels)}
    source_label = sorted_labels[0]
    source = torch.from_numpy(frames[source_label]).float().to(device)

    rows = []
    for target_label in sorted_labels[1:]:
        traj = _run_ode_rollout(
            flow_net=flow_net,
            source=source,
            t_source=label_to_t[source_label],
            t_target=label_to_t[target_label],
            n_steps=n_ode_steps,
            device=device,
        )
        pred = traj[-1].detach().cpu()
        gt = torch.from_numpy(frames[target_label]).float().cpu()
        metrics = compute_distribution_metrics(pred, gt)
        rows.append({"target_label": target_label, **{k: float(v) for k, v in metrics.items()}})
    return rows


# ---------------------------------------------------------------------------
# PACE-only diagnostics
# ---------------------------------------------------------------------------


def compute_coupling_accuracy(run_dir: Path, particle_id_frames: dict[Any, np.ndarray]) -> dict[str, float] | None:
    """Compare Stage-1 matchings to the true particle map when IDs are available."""
    matchings_path = run_dir / "checkpoints" / "stage1_psi" / "stage1_matchings.npz"
    if not matchings_path.exists():
        return None

    matchings = np.load(matchings_path, allow_pickle=True)
    matching_keys = [k for k in matchings.files if k.startswith("matching_")]
    if not matching_keys:
        return None

    accs = []
    for key in matching_keys:
        parts = key.replace("matching_t", "").split("_to_t")
        if len(parts) != 2:
            continue
        l0 = float(parts[0]) if "." in parts[0] else int(parts[0])
        l1 = float(parts[1]) if "." in parts[1] else int(parts[1])
        if l0 not in particle_id_frames or l1 not in particle_id_frames:
            continue
        ids0 = particle_id_frames[l0]
        ids1 = particle_id_frames[l1]
        matching = matchings[key]
        if len(matching) != len(ids0):
            continue
        true_map = {i: np.where(ids1 == pid)[0] for i, pid in enumerate(ids0)}
        hits = 0
        for i, j in enumerate(matching):
            if j in true_map.get(i, []):
                hits += 1
        accs.append(hits / len(matching))

    if not accs:
        return None
    return {"mean_top1": float(np.mean(accs)), "n_segments": len(accs)}


# ---------------------------------------------------------------------------
# CurvedPitchfork: analytic-oracle evaluation
# ---------------------------------------------------------------------------


def load_cp_gt(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the raw CurvedPitchfork NPZ (dense GT trajectories + oracle geometry)."""
    data_path = resolve_data_path(cfg)
    data = np.load(data_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def cp_analytic_oracle(cp: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Analytic normal projector P_N = n n^T via nearest GT manifold point.

    The GT points densely sample the true 1-D curve(s), so a nearest-neighbour
    lookup of the saved analytic unit normal is an accurate oracle.  The same
    projector is used for every method, keeping the comparison fair.  When the
    dataset carries clean (pre-noise) positions, the oracle is anchored on
    those (the true manifold), never on the noisy observations.
    """
    use_clean = "clean_positions" in cp and float(cp.get("noise_sigma", 0.0) or 0.0) > 0.0
    pos = np.asarray(cp["clean_positions"] if use_clean else cp["positions"], dtype=np.float64)
    nrm = np.asarray(cp["normal_unit"], dtype=np.float64)
    d = pos.shape[-1]
    pos2 = pos.reshape(-1, d)
    P = np.einsum("ni,nj->nij", nrm.reshape(-1, d), nrm.reshape(-1, d))
    tree = scipy_spatial_cKDTree(pos2)

    def projector(query: np.ndarray) -> np.ndarray:
        _, idx = tree.query(np.asarray(query, dtype=np.float64), k=1)
        return P[idx]

    projector.tree = tree  # type: ignore[attr-defined]
    return projector


def eval_curved_pitchfork(
    flow_net: torch.nn.Module,
    cfg: dict[str, Any],
    frames: dict[Any, np.ndarray],
    labels: list,
    n_ode_steps: int,
    device: str,
    near_thresh: float = 1.0,
) -> dict[str, Any]:
    """All trajectory metrics from one fine ODE rollout over the full time range.

    A single rollout from the earliest snapshot to the latest yields:
      1. per-particle ADE/FDE (+ near-/far-from-branch stratification via |u|)
      2. committed-region branch error (nearest GT terminal branch curve)
      3. per-particle oracle normal-motion fraction (analytic projector)
      4. Spearman correlation between per-trajectory normal fraction and ADE
      5. off-manifold distance of the predicted paths
    """
    cp = load_cp_gt(cfg)
    sorted_labels = sorted(labels)
    label_to_t = {label: labels_to_timesteps(sorted_labels)[i] for i, label in enumerate(sorted_labels)}
    t0, t1 = label_to_t[sorted_labels[0]], label_to_t[sorted_labels[-1]]

    # If the dataset carries clean (pre-noise) positions, use them as the GT
    # reference for trajectories, branch curves, the oracle, and the rollout
    # source.  The model still trains on the noisy observed frames.
    noise_sigma = float(cp.get("noise_sigma", 0.0) or 0.0)
    use_clean_ref = "clean_positions" in cp and noise_sigma > 0.0
    clean = np.asarray(cp["clean_positions"], dtype=np.float32) if use_clean_ref else None

    if use_clean_ref:
        ref_frames = {label: clean[label] for label in sorted_labels}
    else:
        ref_frames = frames

    source = torch.from_numpy(ref_frames[sorted_labels[0]]).float()
    traj = _run_ode_rollout(
        flow_net=flow_net,
        source=source,
        t_source=t0,
        t_target=t1,
        n_steps=n_ode_steps,
        device=device,
    )
    traj_np = traj.detach().cpu().numpy()  # (S, N, d)
    times = np.linspace(t0, t1, traj_np.shape[0])

    # Sample the rollout at the snapshot times -> predicted trajectories.
    step_idx = [int(np.argmin(np.abs(times - label_to_t[label]))) for label in sorted_labels]
    pred_at_labels = traj_np[step_idx]  # (L, N, d)
    gt_at_labels = np.stack([ref_frames[label] for label in sorted_labels], axis=0)

    out: dict[str, Any] = {"eval_reference": "clean_positions" if use_clean_ref else "observed_positions"}

    # ------------------------------------------------------------------
    # 1. ADE / FDE with near-/far-from-branch stratification
    # ------------------------------------------------------------------
    u_traj = np.asarray(cp["u_traj"])  # (T_all, N); dense frames are index-aligned
    if u_traj.shape[0] > max(sorted_labels):
        u_at_labels = np.stack([u_traj[label] for label in sorted_labels], axis=0)
    else:
        u_at_labels = u_traj
    dists = np.linalg.norm(pred_at_labels - gt_at_labels, axis=-1)  # (L, N)
    per_particle_ade = dists.mean(axis=0)
    per_particle_fde = dists[-1]

    near_mask = np.abs(u_at_labels) < near_thresh
    out["ade"] = {
        "ade": float(per_particle_ade.mean()),
        "fde": float(per_particle_fde.mean()),
        "median_ade": float(np.median(per_particle_ade)),
        "p90_ade": float(np.percentile(per_particle_ade, 90)),
        "n_particles": int(per_particle_ade.shape[0]),
        "near_branch_thresh": float(near_thresh),
        "ade_near_branch": float(dists[near_mask].mean()) if near_mask.any() else float("nan"),
        "ade_far_branch": float(dists[~near_mask].mean()) if (~near_mask).any() else float("nan"),
        "n_near_pairs": int(near_mask.sum()),
        "n_far_pairs": int((~near_mask).sum()),
        "per_particle_ade": [float(v) for v in per_particle_ade],
    }

    # ------------------------------------------------------------------
    # 2. Committed-region branch error
    # ------------------------------------------------------------------
    branch_labels = np.asarray(cp["branch_labels"], dtype=object)  # (N,)
    committed = branch_labels != "trunk"
    gt_terminal = gt_at_labels[-1]
    if committed.sum() > 0:
        tree_terminal = scipy_spatial_cKDTree(gt_terminal[committed])
        _, nn_idx = tree_terminal.query(pred_at_labels[-1], k=1)
        terminal_branch_labels = branch_labels[committed]
        pred_branch = terminal_branch_labels[nn_idx]

        errors: dict[str, list[float]] = {}
        leakage: dict[str, float] = {}
        for b in ("left", "right"):
            mask_b = committed & (branch_labels == b)
            if mask_b.any():
                errors[b] = (pred_branch[mask_b] != b).astype(float).tolist()
        for b_src, b_dst in (("left", "right"), ("right", "left")):
            mask_b = committed & (branch_labels == b_src)
            if mask_b.any():
                leakage[f"{b_src}_to_{b_dst}"] = float((pred_branch[mask_b] == b_dst).mean())

        all_err = np.concatenate([np.asarray(v) for v in errors.values()]) if errors else np.array([float("nan")])
        sqrt_a1 = float(np.sqrt(max(float(cp["a1"]), 0.0))) if "a1" in cp else 1.0
        pred_uncommitted = np.abs(pred_at_labels[-1][:, 0]) < 0.5 * sqrt_a1
        out["branch_error"] = {
            "micro": float(all_err.mean()),
            "macro": float(np.mean([np.mean(v) for v in errors.values()])) if errors else float("nan"),
            "n_committed": int(committed.sum()),
            "leakage": leakage,
            "pred_uncommitted_frac_committed": float(pred_uncommitted[committed].mean()),
        }
    else:
        out["branch_error"] = {
            "micro": float("nan"),
            "macro": float("nan"),
            "n_committed": 0,
            "leakage": {},
            "pred_uncommitted_frac_committed": float("nan"),
        }

    # ------------------------------------------------------------------
    # 3. Per-particle oracle normal-motion fraction (analytic projector)
    # ------------------------------------------------------------------
    projector = cp_analytic_oracle(cp)
    x_mid = 0.5 * (traj_np[:-1] + traj_np[1:])  # (S-1, N, d)
    S1, N, d = x_mid.shape
    dt = np.diff(times)[:, None]  # (S-1, 1)
    t_mid = 0.5 * (times[:-1] + times[1:])

    x_flat = torch.from_numpy(x_mid.reshape(-1, d)).float().to(device)
    t_flat = torch.from_numpy(np.repeat(t_mid, N)).float().to(device)
    with torch.no_grad():
        v_flat = flow_net(t_flat, x_flat).cpu().numpy()
    v = v_flat.reshape(S1, N, d)

    P_N = projector(x_mid.reshape(-1, d)).reshape(S1, N, d, d)
    v_norm_sq = np.sum(v * v, axis=-1)
    v_normal = np.einsum("snij,snj->sni", P_N, v)
    v_normal_sq = np.sum(v_normal * v_normal, axis=-1)

    e_total = np.sum(dt * v_norm_sq, axis=0)  # (N,)
    e_normal = np.sum(dt * v_normal_sq, axis=0)
    eta = e_normal / (e_total + 1e-12)

    # Off-manifold distance: nearest-GT distance at midpoints.
    off_dist, _ = projector.tree.query(x_mid.reshape(-1, d), k=1)  # type: ignore[attr-defined]
    per_particle_off = off_dist.reshape(S1, N).mean(axis=0)

    norm_block: dict[str, Any] = {
        "global_normal_fraction": float(e_normal.sum() / (e_total.sum() + 1e-12)),
        "median_per_traj": float(np.median(eta)),
        "mean_per_traj": float(eta.mean()),
        "p90_per_traj": float(np.percentile(eta, 90)),
        "off_manifold_dist_mean": float(per_particle_off.mean()),
        "off_manifold_dist_p90": float(np.percentile(per_particle_off, 90)),
        "total_energy": float(e_total.sum()),
        "normal_energy": float(e_normal.sum()),
        "per_particle_eta": [float(v) for v in eta],
    }

    # ------------------------------------------------------------------
    # 4. Spearman(per-trajectory normal fraction, per-trajectory ADE)
    # ------------------------------------------------------------------
    try:
        from scipy.stats import spearmanr

        rho, pval = spearmanr(eta, per_particle_ade)
        norm_block["spearman_eta_ade"] = float(rho)
        norm_block["spearman_eta_ade_pvalue"] = float(pval)
        norm_block["spearman_n_trajectories"] = int(eta.shape[0])
    except Exception as exc:  # pragma: no cover
        log.warning("Spearman computation failed: %s", exc)
    out["oracle_normal_fraction"] = norm_block

    # ------------------------------------------------------------------
    # 5. Metadata
    # ------------------------------------------------------------------
    out["curved_pitchfork_meta"] = {
        "t_bif": float(cp["t_bif"]) if "t_bif" in cp else None,
        "a0": float(cp["a0"]) if "a0" in cp else None,
        "a1": float(cp["a1"]) if "a1" in cp else None,
        "beta": float(cp["beta"]) if "beta" in cp else None,
        "noise_sigma": float(cp["noise_sigma"]) if "noise_sigma" in cp else None,
        "n_ode_steps": int(n_ode_steps),
    }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    dm = make_datamodule(cfg)
    dm.setup()

    ckpt_path = infer_stage2_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No Stage-2 checkpoint found in {run_dir}")
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

    is_curved_pitchfork = "curved_pitchfork" in str(cfg.get("data_name", "")).lower()

    # ------------------------------------------------------------------
    # CurvedPitchfork: dedicated analytic-oracle evaluation
    # ------------------------------------------------------------------
    if is_curved_pitchfork:
        log.info("Running CurvedPitchfork analytic-oracle evaluation...")
        output.update(
            eval_curved_pitchfork(
                flow_net=model,
                cfg=cfg,
                frames=frames,
                labels=labels,
                n_ode_steps=args.n_ode_steps,
                device=args.device,
            )
        )
        # PACE-only coupling accuracy (Stage-1 matchings vs true particle map).
        ids = getattr(dm, "particle_ids", None)
        if ids is not None:
            pid_frames = {label: np.asarray(ids) for label in labels}
            coupling_acc = compute_coupling_accuracy(run_dir, pid_frames)
            if coupling_acc is not None:
                output["coupling_accuracy"] = coupling_acc
        log.info("Computing terminal W2...")
        output["w2"] = compute_w2_terminal(model, frames, labels, args.n_ode_steps, args.device)
        # For noisy datasets additionally report W2 against the clean GT cloud.
        cp_gt = load_cp_gt(cfg)
        if "clean_positions" in cp_gt and float(cp_gt.get("noise_sigma", 0.0) or 0.0) > 0.0:
            clean = np.asarray(cp_gt["clean_positions"], dtype=np.float32)
            clean_frames = {label: clean[label] for label in labels}
            output["w2_clean"] = compute_w2_terminal(
                model, clean_frames, labels, args.n_ode_steps, args.device
            )

        output_dir = Path(args.output_dir) if args.output_dir else run_dir / "metrics"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "trajectory_correctness.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        log.info("Trajectory correctness metrics saved to %s", output_path)
        return

    # ------------------------------------------------------------------
    # 1. ADE (needs persistent particle IDs)
    # ------------------------------------------------------------------
    particle_id_key = str(cfg.get("particle_id_key", "particle_ids"))
    particle_id_frames = None
    if hasattr(dm, "particle_ids") and dm.particle_ids is not None:
        # SphereRot exposes particle IDs directly.
        ids = dm.particle_ids
        particle_id_frames = {label: ids for label in labels}
    else:
        particle_id_frames = load_per_label_array(cfg, dm, particle_id_key)

    if particle_id_frames is not None:
        log.info("Computing ADE...")
        distance_fn = lambda a, b: np.linalg.norm(a - b, axis=-1)
        if "sphere_rot" in str(cfg.get("data_name", "")):
            distance_fn = lambda a, b: _geodesic_distance_s2(a, b)

        gt_traj = build_gt_trajectories(frames, particle_id_frames)
        pred_traj = build_predicted_trajectories(
            model, frames, particle_id_frames, labels, args.n_ode_steps, args.device
        )
        output["ade"] = compute_ade(gt_traj, pred_traj, distance_fn)

        coupling_acc = compute_coupling_accuracy(run_dir, particle_id_frames)
        if coupling_acc is not None:
            output["coupling_accuracy"] = coupling_acc
    else:
        log.info("No persistent particle IDs; skipping ADE and coupling accuracy.")

    # ------------------------------------------------------------------
    # 2. Branch / clone-compatible error
    # ------------------------------------------------------------------
    clone_id_key = str(cfg.get("clone_id_key", "clone_ids"))
    clone_frames = load_per_label_array(cfg, dm, clone_id_key)
    fate_label_key = str(cfg.get("fate_label_key", "fate_labels"))
    fate_frames = load_per_label_array(cfg, dm, fate_label_key)

    # Actually compute pred_endpoints cleanly:
    sorted_labels = labels
    label_to_t = {label: labels_to_timesteps(sorted_labels)[i] for i, label in enumerate(sorted_labels)}
    traj = _run_ode_rollout(
        flow_net=model,
        source=torch.from_numpy(frames[sorted_labels[0]]).float(),
        t_source=label_to_t[sorted_labels[0]],
        t_target=label_to_t[sorted_labels[-1]],
        n_steps=args.n_ode_steps,
        device=args.device,
    )
    pred_endpoints = traj[-1].detach().cpu().numpy()

    if clone_frames is not None:
        test_labs = sorted(dm.unique_test_labels) if hasattr(dm, "unique_test_labels") else []
        train_labs = sorted(dm.unique_train_labels) if hasattr(dm, "unique_train_labels") else []
        clone_results = {}
        for tl in test_labs:
            prev = max((tr for tr in train_labs if tr < tl), default=None)
            nxt = min((tr for tr in train_labs if tr > tl), default=None)
            if prev is None or nxt is None or prev not in clone_frames or tl not in clone_frames:
                continue
            ratio = (tl - prev) / (nxt - prev) if nxt != prev else 0.5
            seg_traj = _run_ode_rollout(
                flow_net=model,
                source=torch.from_numpy(frames[prev]).float(),
                t_source=label_to_t[prev],
                t_target=label_to_t[nxt],
                n_steps=args.n_ode_steps,
                device=args.device,
            )
            q_idx = min(int(round(ratio * (seg_traj.shape[0] - 1))), seg_traj.shape[0] - 1)
            seg_pred = seg_traj[q_idx].detach().cpu().numpy()
            clone_results[str(tl)] = compute_clone_compatible_endpoint_mass(
                source_positions=frames[prev],
                source_clones=clone_frames[prev],
                pred_endpoints=seg_pred,
                target_positions=frames[tl],
                target_clones=clone_frames[tl],
            )
        if clone_results:
            output["clone_compatible_mass"] = clone_results

    # Clone-conditioned terminal fate consistency on FULL frames (the metric
    # needs clone structure that subsampled frames destroy).
    if clone_frames is not None and fate_frames is not None:
        fate_results = {}
        raw = np.load(resolve_data_path(cfg), allow_pickle=True)
        pos_key = str(cfg.get("position_key", "positions"))
        tp_key = str(cfg.get("timepoint_key", "timepoints"))
        pos_all = np.asarray(raw[pos_key], dtype=np.float32)
        tp_all = raw[tp_key]
        cl_all = raw[str(cfg.get("clone_id_key", "clone_ids"))]
        ft_all = raw[str(cfg.get("fate_label_key", "fate_labels"))]
        # Map raw time labels to the datamodule's numeric labels.
        if hasattr(dm, "timepoint_label_map") and dm.timepoint_label_map:
            lab_map = dm.timepoint_label_map
        else:
            uniq = sorted(np.unique(tp_all), key=lambda v: float(v) if np.issubdtype(np.asarray(v).dtype, np.number) else str(v))
            lab_map = {v: i for i, v in enumerate(uniq)}

        def _frame_raw(label):
            mask = np.array([lab_map.get(v, lab_map.get(str(v))) for v in tp_all]) == label
            if not mask.any():
                # numeric timepoints: direct compare
                mask = tp_all == label
            return mask

        if getattr(dm, "scaler", None) is not None:
            pos_eval = dm.scaler.transform(pos_all).astype(np.float32)
        else:
            pos_eval = pos_all

        for tl in test_labs:
            prev = max((tr for tr in train_labs if tr < tl), default=None)
            nxt = min((tr for tr in train_labs if tr > tl), default=None)
            if prev is None or nxt is None:
                continue
            src_m, tgt_m = _frame_raw(prev), _frame_raw(tl)
            if not src_m.any() or not tgt_m.any():
                continue
            ratio = (tl - prev) / (nxt - prev) if nxt != prev else 0.5
            seg_traj = _run_ode_rollout(
                flow_net=model,
                source=torch.from_numpy(pos_eval[src_m]).float(),
                t_source=label_to_t[prev],
                t_target=label_to_t[nxt],
                n_steps=args.n_ode_steps,
                device=args.device,
            )
            q_idx = min(int(round(ratio * (seg_traj.shape[0] - 1))), seg_traj.shape[0] - 1)
            seg_pred = seg_traj[q_idx].detach().cpu().numpy()
            res = compute_clone_fate_consistency(
                source_clones=cl_all[src_m],
                source_fates=ft_all[src_m],
                pred_endpoints=seg_pred,
                target_positions=pos_eval[tgt_m],
                target_clones=cl_all[tgt_m],
                target_fates=ft_all[tgt_m],
            )
            if res.get("n_clones", 0) < 10:
                # Too few clones pass the >=3 filter (e.g. LARRY day2); fall back
                # to singletons and flag the protocol honestly.
                res1 = compute_clone_fate_consistency(
                    source_clones=cl_all[src_m],
                    source_fates=ft_all[src_m],
                    pred_endpoints=seg_pred,
                    target_positions=pos_eval[tgt_m],
                    target_clones=cl_all[tgt_m],
                    target_fates=ft_all[tgt_m],
                    min_cells=1,
                )
                res1["min_cells_used"] = 1
                if res1.get("n_clones", 0) > res.get("n_clones", 0):
                    res = res1
            else:
                res["min_cells_used"] = 3
            fate_results[str(tl)] = res
        if fate_results:
            output["clone_fate_consistency"] = fate_results

    if fate_frames is not None:
        log.info("Computing terminal branch consistency...")
        output["terminal_branch_consistency"] = compute_terminal_branch_consistency(
            source_labels=fate_frames[sorted_labels[0]],
            pred_endpoints=pred_endpoints,
            target_positions=frames[sorted_labels[-1]],
            target_labels=fate_frames[sorted_labels[-1]],
        )

    # ------------------------------------------------------------------
    # 3. Oracle normal-motion fraction
    # ------------------------------------------------------------------
    log.info("Computing oracle normal-motion fraction...")
    if "sphere_rot" in str(cfg.get("data_name", "")):
        alpha_true = float(cfg.get("sphere_alpha_true", 8.0))

        def oracle_normal_projector(query: np.ndarray) -> np.ndarray:
            x = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-12)
            return np.einsum("ni,nj->nij", x, x)
    else:
        oracle_normal_projector = build_oracle_normal_projector(
            {l: frames[l] for l in dm.unique_train_labels},
            list(dm.unique_train_labels),
            args.reference_k,
        )

    output["oracle_normal_fraction"] = compute_oracle_normal_fraction(
        flow_net=model,
        frames=frames,
        labels=labels,
        normal_projector_fn=oracle_normal_projector,
        n_ode_steps=args.n_ode_steps,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 4. W2
    # ------------------------------------------------------------------
    log.info("Computing terminal W2...")
    output["w2"] = compute_w2_terminal(model, frames, labels, args.n_ode_steps, args.device)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectory_correctness.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log.info("Trajectory correctness metrics saved to %s", output_path)


def _geodesic_distance_s2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    cos_angle = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.arccos(np.abs(cos_angle))


if __name__ == "__main__":
    main()
