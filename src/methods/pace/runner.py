"""PACE two-stage training runner.

Orchestrates:
    1. Build DataModule
    2. Stage 1: Train psi network with alternating Hungarian matching
    3. Stage 2: Train velocity field using frozen psi
    4. Test: ODE rollout on held-out frames
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from src.methods.pace.stage2.networks import VelocityNet, GeoPathMLP
from src.methods.pace.stage2.ema import EMA
from src.methods.pace.stage2.flow_train import FlowNetTrain
from src.methods.pace.stage2.data_wrapper import PaceDataModuleWrapper
from src.methods.pace.stage2.flow_matcher import labels_to_timesteps
from src.methods.pace.stage2.matching import build_ot_sampler
from src.methods.pace.stage2.eval_metrics import (
    build_full_trajectory_rollout_predictions,
    compute_velocity_alignment_metrics,
    evaluate_full_trajectory_rollout_metrics,
    evaluate_stage2_rollout_metrics,
    write_distribution_metrics_table,
    write_velocity_metrics_table,
    compute_distribution_metrics,
)

from src.methods.pace.flow_matcher import PACEFlowMatcher
from src.methods.pace.psi_train import PACEPsiTrain

log = logging.getLogger(__name__)


# ======================================================================
# Seed
# ======================================================================

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# DataModule factory
# ======================================================================

def _build_datamodule(
    cfg: DictConfig,
    test_labels_override: list | None = None,
) -> PaceDataModuleWrapper:
    data_name = cfg.get("data_name", "")
    plain = OmegaConf.to_container(cfg, resolve=True)
    args = SimpleNamespace(**plain)

    if test_labels_override is not None:
        args.test_timepoint_labels = test_labels_override

    if "eb" in data_name:
        from src.dataloaders.eb_data import EBDataModule
        base = EBDataModule(args)
    else:
        raise ValueError(
            f"Unknown data_name: {data_name}. PACE distribution only ships with eb_phate."
        )

    return PaceDataModuleWrapper(base)


# ======================================================================
# Build Stage 1 training anchors from the datamodule
# ======================================================================

def _build_train_anchors(
    datamodule: PaceDataModuleWrapper,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """Stack per-timepoint training frames into [T, N, dim] anchor tensor.

    All snapshots are trimmed to the minimum particle count across timepoints.
    """
    train_labels = sorted(datamodule.unique_train_labels)
    frames = datamodule.selected_train_frames if hasattr(datamodule, "selected_train_frames") else datamodule.train_frames

    min_n = min(frames[l].shape[0] for l in train_labels)
    anchors = torch.stack([frames[l][:min_n].float() for l in train_labels], dim=0)
    return anchors.to(device), train_labels


def _build_sequential_rollout_predictions(
    rollout_fn,
    flow_net: torch.nn.Module,
    train_frames: dict[int, torch.Tensor],
    train_labels: list[int],
    label_to_t: dict[int, float],
    n_steps: int,
    device: str,
) -> dict[int, np.ndarray]:
    """Roll forward from the earliest train frame and keep per-label predictions.

    Each segment starts from the previous segment's predicted endpoint rather than
    reusing the empirical anchor at the next label. This gives a cleaner view of
    what the learned dynamics produce when rolled out end-to-end.
    """
    if not train_labels:
        return {}

    first_label = train_labels[0]
    current = train_frames[first_label].float().cpu()
    pred_frames = {first_label: current.numpy()}

    for l0, l1 in zip(train_labels[:-1], train_labels[1:]):
        traj = rollout_fn(
            flow_net=flow_net,
            source=current,
            t_source=label_to_t[l0],
            t_target=label_to_t[l1],
            n_steps=n_steps,
            device=device,
        )
        current = traj[-1].detach().cpu()
        pred_frames[l1] = current.numpy()

    return pred_frames


# ======================================================================
# Main runner
# ======================================================================

def run_pace(cfg: DictConfig) -> dict:
    """Run the full PACE two-stage training pipeline."""
    seed = cfg.get("seed", 42)
    _set_seed(seed)

    project_root = Path(cfg.get("working_dir", ".")).resolve()
    method_name = cfg.get("ablation_name", None) or cfg.get("method", {}).get("name", "pace")
    data_name_out = cfg.get("data_name", "unknown")
    dim = cfg.get("dim", None)
    if dim is not None:
        data_name_out = f"{data_name_out}_dim{dim}"
    test_labels = cfg.get("test_timepoint_labels", [])
    if test_labels:
        test_str = "-".join(str(t) for t in sorted(test_labels))
        data_name_out = f"{data_name_out}_test{test_str}"
    output_dir = str(project_root / "results" / data_name_out / method_name)
    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "distribution_metrics.csv")

    log.info(f"Output directory: {output_dir}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    accelerator = cfg.get("accelerator", "auto")
    enable_progress_bar = cfg.get("enable_progress_bar", True)
    dim = cfg.dim

    results: dict = {}
    distribution_metric_rows: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    datamodule = _build_datamodule(cfg)
    datamodule.setup()

    log.info(
        f"Data: {cfg.data_name}, dim={dim}, "
        f"train_labels={sorted(datamodule.unique_train_labels)}, "
        f"test_labels={sorted(datamodule.unique_test_labels)}"
    )

    # ==================================================================
    # STAGE 1:  PACE Psi Training (alternating matching + psi optimisation)
    # ==================================================================
    stage1_ckpt = cfg.get("load_stage1_ckpt", None)

    # Build the psi network (same architecture as pace runner TimeConditionedPsiNet)
    hidden_geopath = list(cfg.get("hidden_dims_geopath", [64, 64]))
    act_geopath = cfg.get("activation_geopath", "gelu")
    time_geopath = cfg.get("time_geopath", True)

    psi_net = GeoPathMLP(
        input_dim=dim,
        hidden_dims=hidden_geopath,
        time_geopath=time_geopath,
        activation=act_geopath,
        batch_norm=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and accelerator != "cpu" else "cpu")

    if stage1_ckpt is None:
        log.info("=" * 60)
        log.info("STAGE 1: PACE Psi network training (alternating matching)")
        log.info("=" * 60)

        train_anchors, train_idx = _build_train_anchors(datamodule, device)
        rematch_candidate_topk = cfg.get("rematch_candidate_topk", None)

        psi_model = PACEPsiTrain(
            psi_net=psi_net,
            train_anchors=train_anchors,
            train_idx=train_idx,
            lambda_metric=float(cfg.get("lambda_metric", 1.0)),
            lambda_global=float(cfg.get("lambda_global", 3.0)),
            lambda_ortho=float(cfg.get("lambda_ortho", 0.0)),
            k_neighbors_local=int(cfg.get("k_neighbors_local", 25)),
            sigma_local=cfg.get("sigma_local", None),
            geometry_window_segments=int(cfg.get("geometry_window_segments", 2)),
            metric_alpha=float(cfg.get("metric_alpha", 8.0)),
            matching_method=cfg.get("matching_method", "hungarian"),
            matching_postprocess=cfg.get("matching_postprocess", "argmax"),
            sinkhorn_reg=float(cfg.get("sinkhorn_reg", 0.05)),
            uot_reg=float(cfg.get("uot_reg", 0.05)),
            uot_reg_m=float(cfg.get("uot_reg_m", 1.0)),
            alpha_dist_init=float(cfg.get("alpha_dist_init", 1.0)),
            gamma_normal_init=float(cfg.get("gamma_normal_init", 0.35)),
            rematch_every=int(cfg.get("rematch_every", 10)),
            T_steps=int(cfg.get("T_steps", 20)),
            t_probe_steps=int(cfg.get("t_probe_steps", 5)),
            rematch_batch_i=int(cfg.get("rematch_batch_i", 32)),
            rematch_batch_j=int(cfg.get("rematch_batch_j", 64)),
            rematch_candidate_topk=(
                None if rematch_candidate_topk is None else int(rematch_candidate_topk)
            ),
            rematch_candidate_min_points=int(cfg.get("rematch_candidate_min_points", 2000)),
            psi_batch_size=cfg.get("stage1_batch_size", None),
            lr=float(cfg.get("geopath_lr", 0.03)),
            optimizer_name=cfg.get("geopath_optimizer", "adam"),
            metric_kernel_mode=str(cfg.get("metric_kernel_mode", "both")),
        )

        total_epochs = int(cfg.get("total_epochs_stage1", 60))

        stage1_dir = os.path.join(output_dir, "checkpoints", "stage1_psi")
        os.makedirs(stage1_dir, exist_ok=True)
        stage1_callbacks = [
            ModelCheckpoint(
                dirpath=stage1_dir,
                every_n_epochs=max(1, total_epochs // 5),
                save_top_k=-1,
                filename="psi-{epoch:03d}",
            )
        ]

        trainer_s1 = pl.Trainer(
            max_epochs=total_epochs,
            callbacks=stage1_callbacks,
            accelerator=accelerator,
            enable_progress_bar=enable_progress_bar,
            num_sanity_val_steps=0,
            default_root_dir=output_dir,
            logger=_build_logger(output_dir, "stage1"),
        )

        trainer_s1.fit(psi_model)

        # Save final checkpoint
        stage1_ckpt = os.path.join(stage1_dir, "last.ckpt")
        trainer_s1.save_checkpoint(stage1_ckpt)
        results["stage1_ckpt"] = stage1_ckpt
        log.info(f"Stage 1 checkpoint: {stage1_ckpt}")

        # Extract trained psi_net
        psi_net = psi_model.psi_net
    else:
        log.info(f"Loading Stage 1 checkpoint: {stage1_ckpt}")
        results["stage1_ckpt"] = stage1_ckpt

        train_anchors, train_idx = _build_train_anchors(datamodule, device)
        psi_model = PACEPsiTrain.load_from_checkpoint(
            stage1_ckpt,
            psi_net=psi_net,
            train_anchors=train_anchors,
            train_idx=train_idx,
        )
        psi_net = psi_model.psi_net

    # Freeze psi for Stage 2
    psi_net.eval()
    for p in psi_net.parameters():
        p.requires_grad = False

    stage1_matchings = [
        m.detach().cpu().long()
        for m in getattr(psi_model, "matchings", [])
        if m is not None
    ]
    if cfg.get("use_stage1_matching", False) and not stage1_matchings:
        log.info(
            "No cached Stage 1 matchings found; rebuilding initial matchings for "
            "Stage 1 metrics/visualization."
        )
        from src.methods.pace.matching import build_initial_matchings

        rebuilt_matchings = build_initial_matchings(
            psi_model.train_anchors.to(device),
            k_neighbors=int(cfg.get("k_neighbors_local", 25)),
            sigma=cfg.get("sigma_local", None),
            alpha_dist=float(cfg.get("alpha_dist_init", 1.0)),
            gamma_normal=float(cfg.get("gamma_normal_init", 0.35)),
        )
        stage1_matchings = [m.detach().cpu().long() for m in rebuilt_matchings]

    # ------------------------------------------------------------------
    # Stage 1 distribution metrics (interpolant midpoint evaluation)
    # ------------------------------------------------------------------
    if datamodule.unique_test_labels:
        log.info("Computing Stage 1 interpolant midpoint metrics...")
        train_labels = sorted(datamodule.unique_train_labels)
        test_labels = sorted(datamodule.unique_test_labels)
        all_labels = sorted(set(train_labels) | set(test_labels))
        all_timesteps = labels_to_timesteps(all_labels)
        label_to_t = {l: all_timesteps[i] for i, l in enumerate(all_labels)}

        train_frames = datamodule.selected_train_frames if hasattr(datamodule, "selected_train_frames") else datamodule.train_frames
        test_frames_dict = datamodule.test_frames

        for test_label in test_labels:
            if test_label not in test_frames_dict:
                continue
            idx = all_labels.index(test_label)
            prev_label = all_labels[idx - 1] if idx > 0 else None
            next_label = all_labels[idx + 1] if idx < len(all_labels) - 1 else None
            if prev_label is None or next_label is None:
                continue
            if prev_label not in train_frames or next_label not in train_frames:
                continue

            f0 = train_frames[prev_label].float().cpu()
            f1 = train_frames[next_label].float().cpu()
            gt = test_frames_dict[test_label].float().cpu()
            seg_idx = train_labels.index(prev_label)
            if seg_idx < len(stage1_matchings):
                match_idx = stage1_matchings[seg_idx].clamp_max(f1.shape[0] - 1)
                n = min(f0.shape[0], match_idx.shape[0])
                x0, x1 = f0[:n], f1[match_idx[:n]]
            else:
                n = min(f0.shape[0], f1.shape[0])
                x0, x1 = f0[:n], f1[:n]

            # PACE interpolant midpoint: s = ratio
            t_min_val = label_to_t[prev_label]
            t_max_val = label_to_t[next_label]
            t_mid_val = label_to_t[test_label]
            s = (t_mid_val - t_min_val) / (t_max_val - t_min_val)

            with torch.no_grad():
                s_batch = torch.full((n, 1), s)
                psi_out = psi_net(x0, x1, s_batch)
                pred = (1 - s) * x0 + s * x1 + s * (1 - s) * psi_out

            m = compute_distribution_metrics(pred, gt)
            row = {
                "stage": "stage1_pace",
                "test_label": int(test_label),
                "mmd": m["mmd"],
                "w1": m["w1"],
                "w2": m["w2"],
                "gw": m["gw"],
            }
            distribution_metric_rows.append(row)
            log.info(
                f"Stage 1 | test_label={test_label} | "
                f"MMD={m['mmd']:.6f} W1={m['w1']:.6f} W2={m['w2']:.6f} GW={m['gw']:.6f}"
            )

        write_distribution_metrics_table(distribution_metric_rows, metrics_path)

    # ------------------------------------------------------------------
    # Stage 1 Visualization (2-D only)
    # ------------------------------------------------------------------
    if dim <= 2:
      log.info("Generating Stage 1 visualizations...")
      try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.plot.stage1_visualization import (
            plot_interpolation_paths,
            plot_midpoint_comparison,
            plot_psi_magnitude,
            plot_trajectory_overview,
        )

        plot_dir = os.path.join(output_dir, "plots", "stage1")
        os.makedirs(plot_dir, exist_ok=True)

        train_labels = sorted(datamodule.unique_train_labels)
        test_labels = sorted(datamodule.unique_test_labels)
        train_frames_raw = (
            datamodule.selected_train_frames
            if hasattr(datamodule, "selected_train_frames")
            else datamodule.train_frames
        )
        test_frames_raw = datamodule.test_frames

        all_labels = sorted(set(train_labels) | set(test_labels))
        all_timesteps = labels_to_timesteps(all_labels)
        label_to_t = {label: all_timesteps[i] for i, label in enumerate(all_labels)}

        n_pairs = 20
        n_interp_steps = 50
        vis_device = "cpu"
        psi_net.eval()

        # --- Interpolation paths (linear vs geodesic) ---
        linear_paths_list, geo_paths_list, bg_frames_list = [], [], []
        for seg_i, (l0, l1) in enumerate(zip(train_labels[:-1], train_labels[1:])):
            f0 = train_frames_raw[l0].float().to(vis_device)
            f1 = train_frames_raw[l1].float().to(vis_device)
            n = min(n_pairs, f0.shape[0], f1.shape[0])
            idx0 = torch.randperm(f0.shape[0])[:n]
            if seg_i < len(stage1_matchings) and stage1_matchings[seg_i].numel() >= f0.shape[0]:
                idx1 = stage1_matchings[seg_i][idx0].to(dtype=torch.long)
            else:
                idx1 = torch.randperm(f1.shape[0])[:n]
            x0, x1 = f0[idx0], f1[idx1]
            ss = torch.linspace(0.0, 1.0, n_interp_steps)
            lin_pts, geo_pts = [], []
            with torch.no_grad():
                for s_val in ss:
                    lin_pt = (1 - s_val) * x0 + s_val * x1
                    lin_pts.append(lin_pt.cpu().numpy())
                    s_batch = torch.full((n, 1), s_val.item(), device=vis_device)
                    psi_out = psi_net(x0, x1, s_batch)
                    geo_pt = lin_pt + s_val * (1 - s_val) * psi_out
                    geo_pts.append(geo_pt.cpu().numpy())
            linear_paths_list.append(np.array(lin_pts))
            geo_paths_list.append(np.array(geo_pts))
            bg_frames_list.append((f0.cpu().numpy(), f1.cpu().numpy()))

        fig1 = plot_interpolation_paths(
            linear_paths_list, geo_paths_list, bg_frames_list, train_labels,
        )
        fig1.savefig(
            os.path.join(plot_dir, "interpolation_paths.png"),
            dpi=150, bbox_inches="tight",
        )

        # --- Midpoint comparison (GT vs linear vs geodesic) ---
        test_gt_np: dict = {}
        lin_pred_np: dict = {}
        geo_pred_np: dict = {}
        metrics_per_label: dict = {}

        if test_labels:
            for test_label in test_labels:
                idx = all_labels.index(test_label)
                prev_label = all_labels[idx - 1] if idx > 0 else None
                next_label = all_labels[idx + 1] if idx < len(all_labels) - 1 else None
                if prev_label is None or next_label is None:
                    continue
                if prev_label not in train_frames_raw or next_label not in train_frames_raw:
                    continue
                if test_label not in test_frames_raw:
                    continue
                f0 = train_frames_raw[prev_label].float().to(vis_device)
                f1 = train_frames_raw[next_label].float().to(vis_device)
                gt = test_frames_raw[test_label].float().cpu()
                t_min = label_to_t[prev_label]
                t_max = label_to_t[next_label]
                t_mid = label_to_t[test_label]
                s = (t_mid - t_min) / (t_max - t_min)
                n = min(f0.shape[0], f1.shape[0])
                x0, x1 = f0[:n], f1[:n]
                with torch.no_grad():
                    lin_mid = ((1 - s) * x0 + s * x1).cpu()
                    s_batch = torch.full((n, 1), s, device=vis_device)
                    psi_out = psi_net(x0, x1, s_batch)
                    geo_mid = (lin_mid + (s * (1 - s) * psi_out).cpu())
                test_gt_np[test_label] = gt.numpy()
                lin_pred_np[test_label] = lin_mid.numpy()
                geo_pred_np[test_label] = geo_mid.numpy()
                lin_m = compute_distribution_metrics(lin_mid, gt)
                geo_m = compute_distribution_metrics(geo_mid, gt)
                metrics_per_label[test_label] = {"linear": lin_m, "geodesic": geo_m}

            fig2 = plot_midpoint_comparison(
                test_gt_np, lin_pred_np, geo_pred_np, test_labels, metrics_per_label,
            )
            fig2.savefig(
                os.path.join(plot_dir, "midpoint_comparison.png"),
                dpi=150, bbox_inches="tight",
            )

        # --- Psi correction magnitude ---
        midpoints_list, psi_norms_list, transition_labels_vis = [], [], []
        for l0, l1 in zip(train_labels[:-1], train_labels[1:]):
            f0 = train_frames_raw[l0].float().to(vis_device)
            f1 = train_frames_raw[l1].float().to(vis_device)
            n = min(f0.shape[0], f1.shape[0])
            x0, x1 = f0[:n], f1[:n]
            with torch.no_grad():
                s_half = torch.full((n, 1), 0.5, device=vis_device)
                psi_out = psi_net(x0, x1, s_half)
                psi_norm = psi_out.norm(dim=-1).cpu().numpy()
            midpoint = (0.5 * (x0 + x1)).cpu().numpy()
            midpoints_list.append(midpoint)
            psi_norms_list.append(psi_norm)
            transition_labels_vis.append((l0, l1))

        fig3 = plot_psi_magnitude(
            midpoints_list, psi_norms_list, transition_labels_vis,
        )
        fig3.savefig(
            os.path.join(plot_dir, "psi_magnitude.png"),
            dpi=150, bbox_inches="tight",
        )

        # --- Trajectory overview ---
        train_frames_np = {l: train_frames_raw[l].cpu().numpy() for l in train_labels}
        n_traj = 30
        traj_segs = []
        for l0, l1 in zip(train_labels[:-1], train_labels[1:]):
            f0 = train_frames_raw[l0].float().to(vis_device)
            f1 = train_frames_raw[l1].float().to(vis_device)
            seg_idx = train_labels.index(l0)
            if seg_idx < len(stage1_matchings):
                match_idx = stage1_matchings[seg_idx].to(vis_device).clamp_max(f1.shape[0] - 1)
                nt = min(n_traj, f0.shape[0], match_idx.shape[0])
                idx0 = torch.randperm(min(f0.shape[0], match_idx.shape[0]), device=vis_device)[:nt]
                x0, x1 = f0[idx0], f1[match_idx[idx0]]
            else:
                nt = min(n_traj, f0.shape[0], f1.shape[0])
                idx0 = torch.randperm(f0.shape[0], device=vis_device)[:nt]
                idx1 = torch.randperm(f1.shape[0], device=vis_device)[:nt]
                x0, x1 = f0[idx0], f1[idx1]
            ss = torch.linspace(0.0, 1.0, 60)
            pts = []
            with torch.no_grad():
                for s_val in ss:
                    lin_pt = (1 - s_val) * x0 + s_val * x1
                    s_batch = torch.full((nt, 1), s_val.item(), device=vis_device)
                    psi_out = psi_net(x0, x1, s_batch)
                    geo_pt = lin_pt + s_val * (1 - s_val) * psi_out
                    pts.append(geo_pt.cpu().numpy())
            traj_segs.append({"l0": l0, "l1": l1, "traj": np.array(pts)})

        test_pred_s1_np: dict = {}
        for test_label in test_labels:
            idx = all_labels.index(test_label)
            prev_label = all_labels[idx - 1] if idx > 0 else None
            next_label = all_labels[idx + 1] if idx < len(all_labels) - 1 else None
            if prev_label is None or next_label is None:
                continue
            if prev_label not in train_frames_raw or next_label not in train_frames_raw:
                continue
            f0 = train_frames_raw[prev_label].float().to(vis_device)
            f1 = train_frames_raw[next_label].float().to(vis_device)
            t_min = label_to_t[prev_label]
            t_max = label_to_t[next_label]
            t_mid = label_to_t[test_label]
            s = (t_mid - t_min) / (t_max - t_min)
            seg_idx = train_labels.index(prev_label)
            if seg_idx < len(stage1_matchings):
                match_idx = stage1_matchings[seg_idx].to(vis_device).clamp_max(f1.shape[0] - 1)
                nt = min(f0.shape[0], match_idx.shape[0])
                x0, x1 = f0[:nt], f1[match_idx[:nt]]
            else:
                nt = min(f0.shape[0], f1.shape[0])
                x0, x1 = f0[:nt], f1[:nt]
            with torch.no_grad():
                s_batch = torch.full((nt, 1), s, device=vis_device)
                psi_out = psi_net(x0, x1, s_batch)
                geo_mid = ((1 - s) * x0 + s * x1 + s * (1 - s) * psi_out).cpu().numpy()
            test_pred_s1_np[test_label] = geo_mid

        fig4 = plot_trajectory_overview(
            train_frames_np, train_labels,
            test_gt_np,
            test_pred_s1_np, test_labels,
            traj_segs,
        )
        fig4.savefig(
            os.path.join(plot_dir, "trajectory_overview.png"),
            dpi=150, bbox_inches="tight",
        )

        plt.close("all")
        log.info(f"Stage 1 plots saved to {plot_dir}")
      except Exception as e:
        log.warning(f"Stage 1 visualization failed: {e}", exc_info=True)
    else:
      log.info(f"Skipping Stage 1 visualization (dim={dim} > 2).")

    # ==================================================================
    # STAGE 2:  Flow (Velocity Field) Training
    # ==================================================================
    log.info("=" * 60)
    log.info("STAGE 2: Training Flow (velocity field) network")
    log.info("=" * 60)

    # Build velocity network
    hidden_flow = list(cfg.get("hidden_dims_flow", [64, 64, 64]))
    act_flow = cfg.get("activation_flow", "selu")
    flow_net = VelocityNet(
        dim=dim, hidden_dims=hidden_flow, activation=act_flow, batch_norm=False,
    )

    ema_decay = cfg.get("ema_decay", None)
    if ema_decay is not None:
        flow_net = EMA(model=flow_net, decay=ema_decay)

    # Build PACE flow matcher with frozen psi
    flow_matcher = PACEFlowMatcher(
        psi_net=psi_net,
        sigma=float(cfg.get("sigma", 0.0)),
    )

    # Stage 2 pairing: either reuse PACE Stage 1 matchings or use the
    # original Stage 2 mini-batch OT matching hook.
    ot_sampler = None
    if cfg.get("use_stage1_matching", False):
        if stage1_matchings:
            from src.methods.pace.matching_sampler import PrecomputedMatchingSampler
            ot_sampler = PrecomputedMatchingSampler(
                train_anchors=psi_model.train_anchors.detach().cpu(),
                matchings=[m.detach().cpu() for m in stage1_matchings],
            )
            log.info("Stage 2 will use pre-computed matchings from Stage 1")
        else:
            log.info("use_stage1_matching=True but no cached matchings; rebuilding from initial matching ...")
            from src.methods.pace.matching_sampler import PrecomputedMatchingSampler
            from src.methods.pace.matching import build_initial_matchings
            _anchors = psi_model.train_anchors.to(device)
            rebuilt_matchings = build_initial_matchings(
                _anchors,
                k_neighbors=int(cfg.get("k_neighbors_local", 25)),
                sigma=cfg.get("sigma_local", None),
                alpha_dist=float(cfg.get("alpha_dist_init", 1.0)),
                gamma_normal=float(cfg.get("gamma_normal_init", 0.35)),
            )
            ot_sampler = PrecomputedMatchingSampler(
                train_anchors=_anchors.detach().cpu(),
                matchings=[m.detach().cpu() for m in rebuilt_matchings],
            )
            log.info("Stage 2 will use rebuilt initial matchings")
    else:
        ot_sampler = build_ot_sampler(cfg)
        if ot_sampler is None:
            log.info("Stage 2 will use raw same-index mini-batch pairing")
        else:
            log.info(
                "Stage 2 will use OT mini-batch matching (method=%s)",
                cfg.get("optimal_transport_method", "None"),
            )

    # Rebuild datamodule for Stage 2 (train-only, no test labels)
    datamodule_s2 = _build_datamodule(cfg, test_labels_override=[])

    has_validation = _has_validation(cfg)

    flow_model = FlowNetTrain(
        flow_matcher=flow_matcher,
        flow_net=flow_net,
        ot_sampler=ot_sampler,
        lr=float(cfg.get("flow_lr", 1e-3)),
        weight_decay=float(cfg.get("flow_weight_decay", 1e-5)),
        optimizer_name=cfg.get("flow_optimizer", "adamw"),
        has_validation=has_validation,
    )

    stage2_dir = os.path.join(output_dir, "checkpoints", "stage2_flow")
    os.makedirs(stage2_dir, exist_ok=True)
    stage2_callbacks = _build_stage2_callbacks(cfg, stage2_dir, has_validation)

    trainer_s2 = pl.Trainer(
        max_epochs=int(cfg.get("epochs", 100)),
        callbacks=stage2_callbacks,
        accelerator=accelerator,
        enable_progress_bar=enable_progress_bar,
        num_sanity_val_steps=0,
        default_root_dir=output_dir,
        logger=_build_logger(output_dir, "stage2"),
    )

    trainer_s2.fit(flow_model, datamodule=datamodule_s2)

    # Save stage 2 checkpoint
    ckpt_callback = [c for c in stage2_callbacks if isinstance(c, ModelCheckpoint)]
    if ckpt_callback and ckpt_callback[0].best_model_path:
        results["stage2_ckpt"] = ckpt_callback[0].best_model_path
    else:
        s2_ckpt = os.path.join(stage2_dir, "last.ckpt")
        trainer_s2.save_checkpoint(s2_ckpt)
        results["stage2_ckpt"] = s2_ckpt

    log.info(f"Stage 2 checkpoint: {results['stage2_ckpt']}")

    # ------------------------------------------------------------------
    # Stage 2 distribution metrics (ODE rollout on held-out frames)
    # ------------------------------------------------------------------
    if cfg.get("test", True) and datamodule.unique_test_labels:
        log.info("Computing Stage 2 distribution metrics (ODE rollout)...")
        train_frames_for_eval = (
            datamodule.selected_train_frames
            if hasattr(datamodule, "selected_train_frames")
            else datamodule.train_frames
        )

        stage2_metric_rows = evaluate_stage2_rollout_metrics(
            flow_net=flow_net,
            train_frames=train_frames_for_eval,
            test_frames=datamodule.test_frames,
            train_labels=sorted(datamodule.unique_train_labels),
            test_labels=sorted(datamodule.unique_test_labels),
            device="cpu",
        )
        distribution_metric_rows.extend(stage2_metric_rows)
        write_distribution_metrics_table(distribution_metric_rows, metrics_path)
        results["distribution_metrics"] = metrics_path
        log.info(f"Stage 2 distribution metrics appended to {metrics_path}")

    # ------------------------------------------------------------------
    # Table-4-style velocity alignment metrics (optional)
    # ------------------------------------------------------------------
    if cfg.get("eval_velocity_metrics", False):
        if hasattr(datamodule, "test_velocity_frames"):
            velocity_rows = compute_velocity_alignment_metrics(
                flow_net=flow_net,
                frames=datamodule.test_frames,
                velocity_frames=datamodule.test_velocity_frames,
                eval_labels=sorted(datamodule.unique_test_labels),
                train_labels=sorted(datamodule.unique_train_labels),
                batch_size=int(cfg.get("velocity_eval_batch_size", 2048)),
                device="cpu",
                stage="stage2",
                predictor="velocity_field",
            )
            stage2_w2_by_label = {
                row["test_label"]: row.get("w2")
                for row in distribution_metric_rows
                if row.get("stage") == "stage2" and row.get("predictor") == "ode_rollout"
            }
            for row in velocity_rows:
                row["w2"] = stage2_w2_by_label.get(row["test_label"], "")

            velocity_metrics_path = os.path.join(metrics_dir, "table4_metrics.csv")
            write_velocity_metrics_table(velocity_rows, velocity_metrics_path)
            results["velocity_metrics"] = velocity_metrics_path
            log.info(f"Table-4-style metrics saved to {velocity_metrics_path}")
        else:
            log.info("eval_velocity_metrics=True but datamodule has no velocity frames; skipping.")

    # ------------------------------------------------------------------
    # Full-trajectory distribution metrics (shared-source rollout)
    # ------------------------------------------------------------------
    if cfg.get("eval_full_trajectory_rollout", False):
        source_label = cfg.get("full_trajectory_source_label", None)
        fulltraj_labels = cfg.get("full_trajectory_eval_labels", None)
        train_frames_for_fulltraj = (
            datamodule.selected_train_frames
            if hasattr(datamodule, "selected_train_frames")
            else datamodule.train_frames
        )
        available_labels = sorted(
            label for label in datamodule.unique_train_labels if label in train_frames_for_fulltraj
        )
        if source_label is None and available_labels:
            source_label = available_labels[0]
        target_labels = list(available_labels if fulltraj_labels is None else fulltraj_labels)

        if source_label in train_frames_for_fulltraj and target_labels:
            log.info(
                "Computing full-trajectory rollout metrics from source_label=%s over labels=%s",
                source_label,
                target_labels,
            )
            fulltraj_metric_rows = evaluate_full_trajectory_rollout_metrics(
                flow_net=flow_net,
                frames=train_frames_for_fulltraj,
                labels=available_labels,
                source_label=source_label,
                target_labels=target_labels,
                n_ode_steps=101,
                device="cpu",
                predictor=cfg.get("full_trajectory_predictor_name", "ode_rollout_from_t0"),
                stage=cfg.get("full_trajectory_stage_name", "stage2_fulltraj"),
            )
            fulltraj_metrics_path = os.path.join(metrics_dir, "full_trajectory_metrics.csv")
            write_distribution_metrics_table(fulltraj_metric_rows, fulltraj_metrics_path)
            results["full_trajectory_metrics"] = fulltraj_metrics_path
            log.info(f"Full-trajectory metrics saved to {fulltraj_metrics_path}")

    # ------------------------------------------------------------------
    # Stage 2 Visualization (ODE rollout)
    # ------------------------------------------------------------------
    if dim <= 2:
      log.info("Generating Stage 2 visualizations...")
      try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.plot.stage2_visualization import (
            plot_full_trajectory_predictions,
            plot_ode_trajectory_overview,
            plot_rollout_prediction_snapshots,
            plot_test_predictions,
        )
        from src.methods.pace.stage2.eval_metrics import (
            _run_ode_rollout,
            _find_train_bracket,
        )

        plot_dir_s2 = os.path.join(output_dir, "plots", "stage2")
        os.makedirs(plot_dir_s2, exist_ok=True)

        s2_train_labels = sorted(datamodule.unique_train_labels)
        s2_test_labels = sorted(datamodule.unique_test_labels)
        s2_train_frames = (
            datamodule.selected_train_frames
            if hasattr(datamodule, "selected_train_frames")
            else datamodule.train_frames
        )

        s2_train_timesteps = labels_to_timesteps(s2_train_labels)
        s2_label_to_t = {l: s2_train_timesteps[i] for i, l in enumerate(s2_train_labels)}

        n_ode_steps = 101
        vis_device_s2 = "cpu"
        flow_net.eval()

        s2_train_np = {l: s2_train_frames[l].cpu().numpy() for l in s2_train_labels}
        s2_test_gt_np: dict = {}
        s2_test_pred_np: dict = {}
        s2_metrics_per_label: dict = {}
        ode_traj_segs = []

        # ODE trajectories between consecutive train frames
        for l0, l1 in zip(s2_train_labels[:-1], s2_train_labels[1:]):
            traj = _run_ode_rollout(
                flow_net=flow_net,
                source=s2_train_frames[l0],
                t_source=s2_label_to_t[l0],
                t_target=s2_label_to_t[l1],
                n_steps=n_ode_steps,
                device=vis_device_s2,
            )
            ode_traj_segs.append({
                "l0": l0, "l1": l1,
                "traj": traj.cpu().numpy(),
            })

        sequential_pred_np = _build_sequential_rollout_predictions(
            rollout_fn=_run_ode_rollout,
            flow_net=flow_net,
            train_frames=s2_train_frames,
            train_labels=s2_train_labels,
            label_to_t=s2_label_to_t,
            n_steps=n_ode_steps,
            device=vis_device_s2,
        )

        # Test predictions via bracket-based ODE rollout
        for test_label in s2_test_labels:
            if test_label not in datamodule.test_frames:
                continue
            prev_label, next_label = _find_train_bracket(test_label, s2_train_labels)
            if prev_label is None or next_label is None:
                continue

            t_prev = s2_label_to_t[prev_label]
            t_next = s2_label_to_t[next_label]
            ratio = (test_label - prev_label) / (next_label - prev_label)

            traj = _run_ode_rollout(
                flow_net=flow_net,
                source=s2_train_frames[prev_label],
                t_source=t_prev,
                t_target=t_next,
                n_steps=n_ode_steps,
                device=vis_device_s2,
            )
            query_idx = int(round(ratio * (n_ode_steps - 1)))
            pred = traj[query_idx].cpu()
            gt = datamodule.test_frames[test_label].float().cpu()

            s2_test_gt_np[test_label] = gt.numpy()
            s2_test_pred_np[test_label] = pred.numpy()
            m = compute_distribution_metrics(pred, gt)
            s2_metrics_per_label[test_label] = m

        fig_overview = plot_ode_trajectory_overview(
            s2_train_np, s2_train_labels,
            s2_test_gt_np, s2_test_pred_np, s2_test_labels,
            ode_traj_segs,
        )
        fig_overview.savefig(
            os.path.join(plot_dir_s2, "ode_trajectory_overview.png"),
            dpi=150, bbox_inches="tight",
        )

        fig_rollout_preds = plot_rollout_prediction_snapshots(
            pred_frames=sequential_pred_np,
            labels=s2_train_labels,
            predictor_label="ODE",
        )
        fig_rollout_preds.savefig(
            os.path.join(plot_dir_s2, "ode_rollout_predictions.png"),
            dpi=150, bbox_inches="tight",
        )

        if s2_test_labels:
            fig_preds = plot_test_predictions(
                s2_test_gt_np, s2_test_pred_np, s2_test_labels, s2_metrics_per_label,
            )
            fig_preds.savefig(
                os.path.join(plot_dir_s2, "test_predictions.png"),
                dpi=150, bbox_inches="tight",
            )

        if cfg.get("eval_full_trajectory_rollout", False):
            source_label = cfg.get("full_trajectory_source_label", None)
            fulltraj_labels = cfg.get("full_trajectory_eval_labels", None)
            available_labels = sorted(
                label for label in s2_train_labels if label in s2_train_frames
            )
            if source_label is None and available_labels:
                source_label = available_labels[0]
            target_labels = list(available_labels if fulltraj_labels is None else fulltraj_labels)

            if source_label in s2_train_frames and target_labels:
                fulltraj_pred_tensors = build_full_trajectory_rollout_predictions(
                    flow_net=flow_net,
                    frames=s2_train_frames,
                    labels=available_labels,
                    source_label=source_label,
                    target_labels=target_labels,
                    n_ode_steps=n_ode_steps,
                    device=vis_device_s2,
                )
                fulltraj_gt_np = {
                    label: s2_train_frames[label].float().cpu().numpy()
                    for label in target_labels
                    if label in s2_train_frames
                }
                fulltraj_pred_np = {
                    label: tensor.numpy()
                    for label, tensor in fulltraj_pred_tensors.items()
                }
                fulltraj_metrics = {
                    row["test_label"]: {
                        "mmd": row["mmd"],
                        "w1": row["w1"],
                        "w2": row["w2"],
                        "gw": row["gw"],
                    }
                    for row in evaluate_full_trajectory_rollout_metrics(
                        flow_net=flow_net,
                        frames=s2_train_frames,
                        labels=available_labels,
                        source_label=source_label,
                        target_labels=target_labels,
                        n_ode_steps=n_ode_steps,
                        device=vis_device_s2,
                        predictor=cfg.get("full_trajectory_predictor_name", "ode_rollout_from_t0"),
                        stage=cfg.get("full_trajectory_stage_name", "stage2_fulltraj"),
                    )
                }

                fig_fulltraj = plot_full_trajectory_predictions(
                    gt_frames=fulltraj_gt_np,
                    pred_frames=fulltraj_pred_np,
                    labels=target_labels,
                    metrics_per_label=fulltraj_metrics,
                    predictor_label="ODE",
                    source_label=source_label,
                )
                fig_fulltraj.savefig(
                    os.path.join(plot_dir_s2, "full_trajectory_predictions.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
        plt.close("all")
        log.info(f"Stage 2 plots saved to {plot_dir_s2}")
      except Exception as e:
        log.warning(f"Stage 2 visualization failed: {e}", exc_info=True)
    else:
      log.info(f"Skipping Stage 2 visualization (dim={dim} > 2).")

    return results


# ======================================================================
# Helpers
# ======================================================================

def _has_validation(cfg: DictConfig) -> bool:
    ratios = list(cfg.get("split_ratios", [1, 0]))
    return len(ratios) >= 2 and float(ratios[0]) < 1.0


def _build_stage2_callbacks(
    cfg: DictConfig, dirpath: str, has_validation: bool,
) -> list:
    from pytorch_lightning.callbacks import EarlyStopping

    callbacks: list = []
    if has_validation:
        callbacks.append(
            EarlyStopping(
                monitor="FlowNet/val_loss",
                patience=int(cfg.get("patience", 25)),
                mode="min",
            )
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=dirpath,
                monitor="FlowNet/val_loss",
                mode="min",
                save_top_k=1,
                filename="best-flow-{epoch:03d}-{step}",
            )
        )
    else:
        callbacks.append(
            ModelCheckpoint(
                dirpath=dirpath,
                every_n_epochs=max(1, int(cfg.get("epochs", 100)) // 5),
                save_top_k=-1,
                filename="flow-{epoch:03d}",
            )
        )
    return callbacks


def _build_logger(output_dir: str, stage: str):
    from pytorch_lightning.loggers import CSVLogger
    return CSVLogger(save_dir=output_dir, name=f"logs_{stage}")
