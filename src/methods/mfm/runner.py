"""MFM two-stage training runner.

Orchestrates the full pipeline:
    1. Build the benchmark DataModule
    2. Stage 1: Train GeoPath (psi) network  ->  checkpoint + visualization
    3. Stage 2: Train Flow (velocity) network using frozen psi
    4. Test: ODE rollout on held-out frames

All configuration is driven by Hydra / OmegaConf.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.methods.mfm.networks import GeoPathMLP, VelocityNet
from src.methods.mfm.ema import EMA
from src.methods.mfm.flow_matcher import labels_to_timesteps
from src.methods.mfm.eval_metrics import (
    _get_module_device,
    build_full_trajectory_rollout_predictions,
    evaluate_full_trajectory_rollout_metrics,
    evaluate_stage1_midpoint_metrics,
    evaluate_stage2_rollout_metrics,
    write_distribution_metrics_table,
)
from src.methods.mfm.flow_matcher import MetricFlowMatcher
from src.methods.mfm.geo_metrics import DataManifoldMetric
from src.methods.mfm.geopath_train import GeoPathNetTrain
from src.methods.mfm.flow_train import FlowNetTrain
from src.methods.mfm.data_wrapper import MFMDataModuleWrapper

log = logging.getLogger(__name__)


# ======================================================================
# Seed
# ======================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ======================================================================
# DataModule factory
# ======================================================================

def build_datamodule(
    cfg: DictConfig,
    test_labels_override: list | None = None,
) -> MFMDataModuleWrapper:
    """Build an MFM-wrapped datamodule from config.

    Parameters
    ----------
    test_labels_override : list | None
        If provided, overrides ``test_timepoint_labels`` in the config.
        Pass ``[]`` to create a train-only datamodule (no test frames).
    """
    data_name = cfg.get("data_name", "")
    args = _cfg_to_namespace(cfg)

    if test_labels_override is not None:
        args.test_timepoint_labels = test_labels_override

    if "oceans" in data_name:
        from src.dataloaders.oceans_data import OceansDataModule
        base = OceansDataModule(args)
    elif "toy_bifurcation" in data_name.lower() or "two_branch" in data_name.lower():
        from src.dataloaders.toy_bifurcation_data import ToyBifurcationDataModule
        base = ToyBifurcationDataModule(args)
    elif "eb" in data_name:
        from src.dataloaders.eb_data import EBDataModule
        base = EBDataModule(args)
    elif "wot" in data_name.lower():
        from src.dataloaders.scnode_wot_data import ScNodeWOTDataModule
        base = ScNodeWOTDataModule(args)
    elif "cite" in data_name:
        from src.dataloaders.cite_data import CiteDataModule
        base = CiteDataModule(args)
    elif "multi" in data_name:
        from src.dataloaders.multi_data import MultiDataModule
        base = MultiDataModule(args)
    elif "ipsc" in data_name:
        from src.dataloaders.ipsc_data import IPSCDataModule
        base = IPSCDataModule(args)
    else:
        raise ValueError(f"Unknown data_name: {data_name}")

    return MFMDataModuleWrapper(base)


# ======================================================================
# Main runner
# ======================================================================

def run_mfm(cfg: DictConfig) -> dict:
    """Run the full MFM two-stage training pipeline.

    Returns a dict with paths to checkpoints and any test metrics.
    """
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # All artefacts go under results/<method_name>/
    project_root = Path(cfg.get("working_dir", ".")).resolve()
    method_name = cfg.get("method", {}).get("name", "mfm")
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
    metrics_path = os.path.join(metrics_dir, "distribution_metrics.csv")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    datamodule = build_datamodule(cfg)
    datamodule.setup()
    dim = cfg.dim
    has_validation = _has_validation(cfg)
    num_timesteps = datamodule.num_timesteps

    log.info(
        f"Data: {cfg.data_name}, dim={dim}, "
        f"train_labels={datamodule.unique_train_labels}, "
        f"test_labels={datamodule.unique_test_labels}, "
        f"has_validation={has_validation}"
    )

    # ------------------------------------------------------------------
    # 2. Networks
    # ------------------------------------------------------------------
    hidden_geopath = list(cfg.get("hidden_dims_geopath", [64, 64, 64]))
    hidden_flow = list(cfg.get("hidden_dims_flow", [64, 64, 64]))
    act_geopath = cfg.get("activation_geopath", "selu")
    act_flow = cfg.get("activation_flow", "selu")
    time_geopath = cfg.get("time_geopath", True)

    geopath_net = GeoPathMLP(
        input_dim=dim,
        hidden_dims=hidden_geopath,
        time_geopath=time_geopath,
        activation=act_geopath,
        batch_norm=False,
    )
    flow_net = VelocityNet(
        dim=dim,
        hidden_dims=hidden_flow,
        activation=act_flow,
        batch_norm=False,
    )

    ema_decay = cfg.get("ema_decay", None)
    if ema_decay is not None:
        geopath_net = EMA(model=geopath_net, decay=ema_decay)
        flow_net = EMA(model=flow_net, decay=ema_decay)

    # OT sampler
    ot_sampler = _build_ot_sampler(cfg)

    # ------------------------------------------------------------------
    # 3. Flow matcher
    # ------------------------------------------------------------------
    flow_matcher = MetricFlowMatcher(
        geopath_net=geopath_net,
        sigma=cfg.get("sigma", 0.1),
        alpha=1.0,  # MFM always uses geodesic correction
    )

    # ------------------------------------------------------------------
    # 4. Accelerator / Trainer common args
    # ------------------------------------------------------------------
    accelerator = cfg.get("accelerator", "auto")
    enable_progress_bar = cfg.get("enable_progress_bar", True)

    results: dict = {}
    distribution_metric_rows: list[dict] = []

    # ==================================================================
    # STAGE 1:  GeoPath (Psi) Training
    # ==================================================================
    stage1_ckpt = cfg.get("load_stage1_ckpt", None)

    if stage1_ckpt is None:
        log.info("=" * 60)
        log.info("STAGE 1: Training GeoPath (psi) network")
        log.info("=" * 60)

        gammas = list(cfg.get("gammas", [0.1]))
        rho = float(cfg.get("rho", 1e-4))
        alpha_metric = float(cfg.get("alpha_metric", 0.3))
        velocity_metric = cfg.get("velocity_metric", "land")

        data_manifold_metric = DataManifoldMetric(
            velocity_metric=velocity_metric,
            gammas=gammas,
            rho=rho,
            alpha_metric=alpha_metric,
        )

        geopath_model = GeoPathNetTrain(
            flow_matcher=flow_matcher,
            geopath_net=geopath_net,
            data_manifold_metric=data_manifold_metric,
            num_timesteps=num_timesteps,
            ot_sampler=ot_sampler,
            lr=float(cfg.get("geopath_lr", 1e-4)),
            weight_decay=float(cfg.get("geopath_weight_decay", 1e-5)),
            optimizer_name=cfg.get("geopath_optimizer", "adam"),
            has_validation=has_validation,
        )

        # Callbacks
        stage1_dir = os.path.join(output_dir, "checkpoints", "stage1_geopath")
        stage1_callbacks = _build_stage1_callbacks(cfg, stage1_dir, has_validation)

        trainer_s1 = pl.Trainer(
            max_epochs=int(cfg.get("epochs_geopath", 200)),
            callbacks=stage1_callbacks,
            accelerator=accelerator,
            enable_progress_bar=enable_progress_bar,
            num_sanity_val_steps=0,
            default_root_dir=output_dir,
            logger=_build_logger(cfg, "stage1", output_dir),
        )

        trainer_s1.fit(geopath_model, datamodule=datamodule)

        # Load best checkpoint
        ckpt_callback = [c for c in stage1_callbacks if isinstance(c, ModelCheckpoint)]
        if ckpt_callback and ckpt_callback[0].best_model_path:
            stage1_ckpt = ckpt_callback[0].best_model_path
        else:
            # No best checkpoint (no validation) — save manually
            stage1_ckpt = os.path.join(stage1_dir, "last.ckpt")
            trainer_s1.save_checkpoint(stage1_ckpt)

        results["stage1_ckpt"] = stage1_ckpt
        log.info(f"Stage 1 checkpoint: {stage1_ckpt}")
    else:
        log.info(f"Loading Stage 1 checkpoint: {stage1_ckpt}")
        results["stage1_ckpt"] = stage1_ckpt

    # Reload geopath weights from checkpoint
    loaded = GeoPathNetTrain.load_from_checkpoint(
        stage1_ckpt,
        flow_matcher=flow_matcher,
        geopath_net=geopath_net,
        data_manifold_metric=DataManifoldMetric(),  # dummy, just for loading
        num_timesteps=num_timesteps,
    )
    flow_matcher.geopath_net = loaded.geopath_net
    geopath_net = loaded.geopath_net

    # ------------------------------------------------------------------
    # Stage 1 distribution metrics (separate from Lightning metrics.csv)
    # ------------------------------------------------------------------
    if datamodule.unique_test_labels:
        stage1_metric_rows = evaluate_stage1_midpoint_metrics(
            geopath_net=geopath_net,
            flow_matcher=flow_matcher,
            train_frames=datamodule.selected_train_frames,
            test_frames=datamodule.test_frames,
            train_labels=sorted(datamodule.unique_train_labels),
            test_labels=sorted(datamodule.unique_test_labels),
            device="cpu",
        )
        distribution_metric_rows.extend(stage1_metric_rows)
        write_distribution_metrics_table(distribution_metric_rows, metrics_path)
        results["distribution_metrics"] = metrics_path
        log.info(f"Stage 1 distribution metrics saved to {metrics_path}")

    # ------------------------------------------------------------------
    # Stage 1 Visualization (2-D only)
    # ------------------------------------------------------------------
    if dim <= 2:
      log.info("Generating Stage 1 visualizations...")
      try:
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
        train_frames_raw = datamodule.selected_train_frames
        test_frames_raw = datamodule.test_frames

        all_labels = sorted(set(train_labels) | set(test_labels))
        all_timesteps = labels_to_timesteps(all_labels)
        label_to_t = {label: all_timesteps[i] for i, label in enumerate(all_labels)}

        n_pairs = 20
        n_interp_steps = 50
        device = _get_module_device(geopath_net, fallback="cpu")
        geopath_net.eval()

        # --- Pre-compute interpolation paths ---
        linear_paths_list, geo_paths_list, bg_frames_list = [], [], []
        for row, (l0, l1) in enumerate(zip(train_labels[:-1], train_labels[1:])):
            f0 = train_frames_raw[l0].to(device)
            f1 = train_frames_raw[l1].to(device)
            n = min(n_pairs, f0.shape[0], f1.shape[0])
            idx0 = torch.randperm(f0.shape[0])[:n]
            idx1 = torch.randperm(f1.shape[0])[:n]
            x0, x1 = f0[idx0], f1[idx1]
            t_min = label_to_t[l0]
            t_max = label_to_t[l1]
            ts = torch.linspace(t_min, t_max, n_interp_steps).to(device)
            lin_pts, geo_pts = [], []
            with torch.no_grad():
                for t_val in ts:
                    s = (t_val - t_min) / (t_max - t_min)
                    lin_pt = (1 - s) * x0 + s * x1
                    lin_pts.append(lin_pt.cpu().numpy())
                    t_batch = t_val.expand(n)
                    gamma_val = flow_matcher.gamma(t_batch, t_min, t_max).unsqueeze(-1)
                    psi = geopath_net(x0, x1, t_batch.unsqueeze(-1))
                    geo_pt = lin_pt + gamma_val * psi
                    geo_pts.append(geo_pt.cpu().numpy())
            linear_paths_list.append(np.array(lin_pts))
            geo_paths_list.append(np.array(geo_pts))
            bg_frames_list.append((f0.cpu().numpy(), f1.cpu().numpy()))

        fig1 = plot_interpolation_paths(linear_paths_list, geo_paths_list, bg_frames_list, train_labels)
        fig1.savefig(os.path.join(plot_dir, "interpolation_paths.png"), dpi=150, bbox_inches="tight")

        # --- Pre-compute midpoint comparison ---
        if test_labels:
            from src.methods.mfm.eval_metrics import compute_distribution_metrics
            test_gt_np, lin_pred_np, geo_pred_np = {}, {}, {}
            metrics_per_label = {}
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
                f0 = train_frames_raw[prev_label].to(device)
                f1 = train_frames_raw[next_label].to(device)
                gt = test_frames_raw[test_label].float().cpu()
                t_min = label_to_t[prev_label]
                t_max = label_to_t[next_label]
                t_mid = label_to_t[test_label]
                s = (t_mid - t_min) / (t_max - t_min)
                n = min(f0.shape[0], f1.shape[0])
                x0, x1 = f0[:n], f1[:n]
                with torch.no_grad():
                    lin_mid = ((1 - s) * x0 + s * x1).cpu()
                    t_batch = torch.full((n,), t_mid, device=device)
                    gamma_val = flow_matcher.gamma(t_batch, t_min, t_max).unsqueeze(-1)
                    psi = geopath_net(x0, x1, t_batch.unsqueeze(-1))
                    geo_mid = (lin_mid + (gamma_val * psi).cpu())
                test_gt_np[test_label] = gt.numpy()
                lin_pred_np[test_label] = lin_mid.numpy()
                geo_pred_np[test_label] = geo_mid.numpy()
                lin_m = compute_distribution_metrics(lin_mid, gt)
                geo_m = compute_distribution_metrics(geo_mid, gt)
                metrics_per_label[test_label] = {"linear": lin_m, "geodesic": geo_m}

            fig2 = plot_midpoint_comparison(test_gt_np, lin_pred_np, geo_pred_np, test_labels, metrics_per_label)
            fig2.savefig(os.path.join(plot_dir, "midpoint_comparison.png"), dpi=150, bbox_inches="tight")

        # --- Pre-compute psi magnitude ---
        midpoints_list, psi_norms_list, transition_labels = [], [], []
        for l0, l1 in zip(train_labels[:-1], train_labels[1:]):
            f0 = train_frames_raw[l0].to(device)
            f1 = train_frames_raw[l1].to(device)
            n = min(f0.shape[0], f1.shape[0])
            x0, x1 = f0[:n], f1[:n]
            with torch.no_grad():
                t_half = torch.full((n, 1), 0.5, device=device)
                psi = geopath_net(x0, x1, t_half)
                psi_norm = psi.norm(dim=-1).cpu().numpy()
            midpoint = (0.5 * (x0 + x1)).cpu().numpy()
            midpoints_list.append(midpoint)
            psi_norms_list.append(psi_norm)
            transition_labels.append((l0, l1))

        fig3 = plot_psi_magnitude(midpoints_list, psi_norms_list, transition_labels)
        fig3.savefig(os.path.join(plot_dir, "psi_magnitude.png"), dpi=150, bbox_inches="tight")

        # --- Pre-compute trajectory overview ---
        if test_labels:
            train_frames_np = {l: train_frames_raw[l].cpu().numpy() for l in train_labels}
            # Re-use geo trajectories from interpolation paths
            n_traj = 30
            traj_segs = []
            for seg_i, (l0, l1) in enumerate(zip(train_labels[:-1], train_labels[1:])):
                f0 = train_frames_raw[l0].to(device)
                f1 = train_frames_raw[l1].to(device)
                n = min(n_traj, f0.shape[0], f1.shape[0])
                idx0 = torch.randperm(f0.shape[0])[:n]
                idx1 = torch.randperm(f1.shape[0])[:n]
                x0, x1 = f0[idx0], f1[idx1]
                t_min = label_to_t[l0]
                t_max = label_to_t[l1]
                ts = torch.linspace(t_min, t_max, 60).to(device)
                pts = []
                with torch.no_grad():
                    for t_val in ts:
                        s = (t_val - t_min) / (t_max - t_min)
                        lin_pt = (1 - s) * x0 + s * x1
                        t_batch = t_val.expand(n)
                        gamma_val = flow_matcher.gamma(t_batch, t_min, t_max).unsqueeze(-1)
                        psi = geopath_net(x0, x1, t_batch.unsqueeze(-1))
                        geo_pt = lin_pt + gamma_val * psi
                        pts.append(geo_pt.cpu().numpy())
                traj_segs.append({"l0": l0, "l1": l1, "traj": np.array(pts)})

            # Geodesic predicted midpoints for test labels
            test_pred_np = {}
            for test_label in test_labels:
                idx = all_labels.index(test_label)
                prev_label = all_labels[idx - 1] if idx > 0 else None
                next_label = all_labels[idx + 1] if idx < len(all_labels) - 1 else None
                if prev_label is None or next_label is None:
                    continue
                if prev_label not in train_frames_raw or next_label not in train_frames_raw:
                    continue
                f0 = train_frames_raw[prev_label].to(device)
                f1 = train_frames_raw[next_label].to(device)
                t_min = label_to_t[prev_label]
                t_max = label_to_t[next_label]
                t_mid = label_to_t[test_label]
                s = (t_mid - t_min) / (t_max - t_min)
                n = min(f0.shape[0], f1.shape[0])
                x0, x1 = f0[:n], f1[:n]
                with torch.no_grad():
                    lin_mid = (1 - s) * x0 + s * x1
                    t_batch = torch.full((n,), t_mid, device=device)
                    gamma_val = flow_matcher.gamma(t_batch, t_min, t_max).unsqueeze(-1)
                    psi = geopath_net(x0, x1, t_batch.unsqueeze(-1))
                    geo_mid = (lin_mid + gamma_val * psi).cpu().numpy()
                test_pred_np[test_label] = geo_mid

            fig4 = plot_trajectory_overview(
                train_frames_np, train_labels, test_gt_np, test_pred_np, test_labels, traj_segs,
            )
            fig4.savefig(os.path.join(plot_dir, "trajectory_overview.png"), dpi=150, bbox_inches="tight")

        plt.close("all")
        log.info(f"Stage 1 plots saved to {plot_dir}")
      except Exception as e:
        log.warning(f"Stage 1 visualization failed: {e}")
    else:
      log.info(f"Skipping Stage 1 visualization (dim={dim} > 2).")

    # ==================================================================
    # STAGE 2:  Flow (Velocity Field) Training
    # ==================================================================
    log.info("=" * 60)
    log.info("STAGE 2: Training Flow (velocity field) network")
    log.info("=" * 60)

    # Freeze geopath network
    for p in geopath_net.parameters():
        p.requires_grad = False

    # Rebuild datamodule for stage 2 — train-only (no test labels),
    # so the training data is identical to Stage 1.
    datamodule_s2 = build_datamodule(cfg, test_labels_override=[])

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
    stage2_callbacks = _build_stage2_callbacks(cfg, stage2_dir, has_validation)

    trainer_s2 = pl.Trainer(
        max_epochs=int(cfg.get("epochs", 100)),
        callbacks=stage2_callbacks,
        accelerator=accelerator,
        enable_progress_bar=enable_progress_bar,
        num_sanity_val_steps=0,
        default_root_dir=output_dir,
        logger=_build_logger(cfg, "stage2", output_dir),
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
    # Stage 2 distribution metrics (separate from Lightning metrics.csv)
    # Use the original datamodule (with test labels) for evaluation.
    # ------------------------------------------------------------------
    if cfg.get("test", True) and datamodule.unique_test_labels:
        log.info("Computing Stage 2 distribution metrics (ODE rollout)...")
        if hasattr(datamodule, "selected_train_frames"):
            stage2_train_frames = datamodule.selected_train_frames
        else:
            stage2_train_frames = datamodule.train_frames

        stage2_metric_rows = evaluate_stage2_rollout_metrics(
            flow_net=flow_net,
            train_frames=stage2_train_frames,
            test_frames=datamodule.test_frames,
            train_labels=sorted(datamodule.unique_train_labels),
            test_labels=sorted(datamodule.unique_test_labels),
            device="cpu",
        )
        distribution_metric_rows.extend(stage2_metric_rows)
        write_distribution_metrics_table(distribution_metric_rows, metrics_path)
        results["distribution_metrics"] = metrics_path
        log.info(f"Stage 2 distribution metrics appended to {metrics_path}")

    if cfg.get("eval_full_trajectory_rollout", False):
        train_frames_for_fulltraj = (
            datamodule.selected_train_frames
            if hasattr(datamodule, "selected_train_frames")
            else datamodule.train_frames
        )
        available_labels = sorted(
            label for label in datamodule.unique_train_labels if label in train_frames_for_fulltraj
        )
        source_label = cfg.get("full_trajectory_source_label", None)
        if source_label is None and available_labels:
            source_label = available_labels[0]
        fulltraj_labels = cfg.get("full_trajectory_eval_labels", None)
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
    # Stage 2 Visualization (2-D only)
    # ------------------------------------------------------------------
    if dim <= 2:
      log.info("Generating Stage 2 visualizations...")
      try:
        import matplotlib.pyplot as plt
        from src.plot.stage2_visualization import (
            plot_full_trajectory_predictions,
            plot_ode_trajectory_overview,
            plot_test_predictions,
        )
        from src.methods.mfm.eval_metrics import (
            _run_ode_rollout,
            _find_train_bracket,
            compute_distribution_metrics,
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

        # Training-time mapping (must match flow_train.py)
        s2_train_timesteps = labels_to_timesteps(s2_train_labels)
        s2_label_to_t = {l: s2_train_timesteps[i] for i, l in enumerate(s2_train_labels)}

        n_ode_steps = 101
        device_s2 = _get_module_device(flow_net, fallback="cpu")
        flow_net.eval()

        # Pre-compute ODE trajectories and test predictions
        s2_train_np = {l: s2_train_frames[l].cpu().numpy() for l in s2_train_labels}
        s2_test_gt_np = {}
        s2_test_pred_np = {}
        s2_metrics_per_label = {}
        ode_traj_segs = []

        # ODE trajectories between consecutive train frames
        for l0, l1 in zip(s2_train_labels[:-1], s2_train_labels[1:]):
            traj = _run_ode_rollout(
                flow_net=flow_net,
                source=s2_train_frames[l0],
                t_source=s2_label_to_t[l0],
                t_target=s2_label_to_t[l1],
                n_steps=n_ode_steps,
                device=device_s2,
            )
            ode_traj_segs.append({
                "l0": l0, "l1": l1,
                "traj": traj.cpu().numpy(),  # (steps, n, 2)
            })

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
                device=device_s2,
            )
            query_idx = int(round(ratio * (n_ode_steps - 1)))
            pred = traj[query_idx].cpu()
            gt = datamodule.test_frames[test_label].float().cpu()

            s2_test_gt_np[test_label] = gt.numpy()
            s2_test_pred_np[test_label] = pred.numpy()
            m = compute_distribution_metrics(pred, gt)
            s2_metrics_per_label[test_label] = m

        # Draw
        fig_overview = plot_ode_trajectory_overview(
            s2_train_np, s2_train_labels,
            s2_test_gt_np, s2_test_pred_np, s2_test_labels,
            ode_traj_segs,
        )
        fig_overview.savefig(
            os.path.join(plot_dir_s2, "ode_trajectory_overview.png"),
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
            available_labels = sorted(
                label for label in s2_train_labels if label in s2_train_frames
            )
            source_label = cfg.get("full_trajectory_source_label", None)
            if source_label is None and available_labels:
                source_label = available_labels[0]
            fulltraj_labels = cfg.get("full_trajectory_eval_labels", None)
            target_labels = list(available_labels if fulltraj_labels is None else fulltraj_labels)

            if source_label in s2_train_frames and target_labels:
                fulltraj_pred_tensors = build_full_trajectory_rollout_predictions(
                    flow_net=flow_net,
                    frames=s2_train_frames,
                    labels=available_labels,
                    source_label=source_label,
                    target_labels=target_labels,
                    n_ode_steps=n_ode_steps,
                    device=device_s2,
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
                        device=device_s2,
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
        log.warning(f"Stage 2 visualization failed: {e}")
    else:
      log.info(f"Skipping Stage 2 visualization (dim={dim} > 2).")

    return results


# ======================================================================
# Helpers
# ======================================================================

def _cfg_to_namespace(cfg: DictConfig):
    """Convert OmegaConf to a simple namespace for backward-compatible DataModules."""
    from types import SimpleNamespace
    plain = OmegaConf.to_container(cfg, resolve=True)
    return SimpleNamespace(**plain)


def _has_validation(cfg: DictConfig) -> bool:
    ratios = list(cfg.get("split_ratios", [1, 0]))
    return len(ratios) >= 2 and float(ratios[0]) < 1.0


def _build_ot_sampler(cfg: DictConfig):
    method = cfg.get("optimal_transport_method", "None")
    if method is None or str(method) == "None":
        return None
    from torchcfm.optimal_transport import OTPlanSampler
    return OTPlanSampler(method=method)


def _build_stage1_callbacks(
    cfg: DictConfig, dirpath: str, has_validation: bool,
) -> list:
    os.makedirs(dirpath, exist_ok=True)
    callbacks: list = []

    if has_validation:
        callbacks.append(
            EarlyStopping(
                monitor="GeoPathNet/val_loss",
                patience=int(cfg.get("patience_geopath", 25)),
                mode="min",
            )
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=dirpath,
                monitor="GeoPathNet/val_loss",
                mode="min",
                save_top_k=1,
                filename="best-geopath-{epoch:03d}-{step}",
            )
        )
    else:
        # No validation: save checkpoint every N epochs or at the end
        callbacks.append(
            ModelCheckpoint(
                dirpath=dirpath,
                every_n_epochs=max(1, int(cfg.get("epochs_geopath", 200)) // 5),
                save_top_k=-1,
                filename="geopath-{epoch:03d}",
            )
        )
    return callbacks


def _build_stage2_callbacks(
    cfg: DictConfig, dirpath: str, has_validation: bool,
) -> list:
    os.makedirs(dirpath, exist_ok=True)
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


def _build_logger(cfg: DictConfig, stage: str, output_dir: str):
    """Build a CSVLogger (always available) for the given stage."""
    from pytorch_lightning.loggers import CSVLogger
    return CSVLogger(save_dir=output_dir, name=f"logs_{stage}")
