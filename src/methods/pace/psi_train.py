"""Stage 1 — PACE Psi network training with alternating Hungarian matching.

Wraps the alternating optimisation loop from ``pace runner`` into a
PyTorch Lightning module:

1. Build initial Hungarian matchings (distance + normal penalty).
2. Train psi for ``rematch_every`` epochs with fixed matchings.
3. Re-solve matchings using metric-weighted path action cost.
4. Repeat until ``total_epochs`` is reached.

The psi network learns the geodesic correction in the PACE interpolant:

    x_t = (1-s)*x0 + s*x1 + s*(1-s)*psi(x0, x1, s)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.methods.pace.geometry import (
    build_anchor_normal_bank,
    compute_metric_energy_for_segments,
    estimate_local_normals_kernel,
    estimate_normal_projector,
    estimate_segment_geometric_bandwidths,
    evaluate_metric_field,
)
from src.methods.pace.interpolant import compute_interpolant_and_velocity
from src.methods.pace.matching import (
    build_initial_matchings,
    build_refined_matchings_from_metric_cost,
)

log = logging.getLogger(__name__)


class PACEPsiTrain(pl.LightningModule):
    """Stage 1 Lightning module for PACE psi network training."""

    def __init__(
        self,
        psi_net: nn.Module,
        train_anchors: torch.Tensor,
        train_idx: list[int],
        train_point_ids: torch.Tensor | None = None,
        # Loss weights
        lambda_metric: float = 1.0,
        lambda_global: float = 3.0,
        lambda_ortho: float = 0.0,
        # Geometry
        k_neighbors_local: int = 25,
        sigma_local: float | None = None,
        geometry_window_segments: int = 2,
        metric_alpha: float = 8.0,
        # Matching
        matching_method: str = "hungarian",
        matching_postprocess: str = "argmax",
        sinkhorn_reg: float = 0.05,
        uot_reg: float = 0.05,
        uot_reg_m: float = 1.0,
        alpha_dist_init: float = 1.0,
        gamma_normal_init: float = 0.35,
        rematch_every: int = 10,
        # Trajectory sampling
        T_steps: int = 20,
        t_probe_steps: int = 5,
        rematch_batch_i: int = 32,
        rematch_batch_j: int = 64,
        rematch_candidate_topk: int | None = None,
        rematch_candidate_min_points: int = 2000,
        # Mini-batch
        psi_batch_size: int | None = None,
        # Optimiser
        lr: float = 0.03,
        optimizer_name: str = "adam",
        # Ablation knobs
        metric_kernel_mode: str = "both",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["psi_net", "train_anchors", "train_point_ids"])
        self.psi_net = psi_net
        self.automatic_optimization = True

        # Register anchors as buffers so they move with the module
        self.register_buffer("train_anchors", train_anchors)
        if train_point_ids is not None:
            self.register_buffer("train_point_ids", train_point_ids)
        else:
            self.train_point_ids = None

        self.train_idx = list(train_idx)
        self.num_segments = train_anchors.shape[0] - 1
        self.N = train_anchors.shape[1]
        self.dim = train_anchors.shape[2]

        # Loss weights
        self.lambda_metric = lambda_metric
        self.lambda_global = lambda_global
        self.lambda_ortho = lambda_ortho

        # Geometry params
        self.k_neighbors_local = k_neighbors_local
        self.sigma_local = sigma_local
        self.geometry_window_segments = geometry_window_segments
        self.metric_alpha = metric_alpha

        # Matching params
        self.matching_method = matching_method
        self.matching_postprocess = matching_postprocess
        self.sinkhorn_reg = sinkhorn_reg
        self.uot_reg = uot_reg
        self.uot_reg_m = uot_reg_m
        self.alpha_dist_init = alpha_dist_init
        self.gamma_normal_init = gamma_normal_init
        self.rematch_every = rematch_every

        # Trajectory sampling
        self.T_steps = T_steps
        self.t_probe_steps = t_probe_steps
        self.rematch_batch_i = rematch_batch_i
        self.rematch_batch_j = rematch_batch_j
        self.rematch_candidate_topk = rematch_candidate_topk
        self.rematch_candidate_min_points = rematch_candidate_min_points

        # Mini-batch
        # If psi_batch_size is None or >= N, use full-batch (original behaviour)
        if psi_batch_size is not None and psi_batch_size >= self.N:
            psi_batch_size = None
        self.psi_batch_size = psi_batch_size
        self._use_minibatch = psi_batch_size is not None

        # Optimiser
        self.lr = lr
        self.optimizer_name = optimizer_name

        # Ablation knobs
        self.metric_kernel_mode = metric_kernel_mode

        # State (initialised in on_train_start)
        self.matchings: list[torch.Tensor] = []
        self.normal_bank: dict | None = None
        self.segment_bandwidths: list[dict] = []

        # Pre-build segment labels for the cross-segment mask (full-batch only)
        if not self._use_minibatch:
            seg_labels = [
                torch.full((self.N * T_steps, 1), float(k))
                for k in range(self.num_segments)
            ]
            seg_all = torch.cat(seg_labels, dim=0)
            self.register_buffer("mask_diff_seg", (seg_all != seg_all.T).float())
        else:
            self.mask_diff_seg = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_train_start(self) -> None:
        device = self.device
        anchors = self.train_anchors.to(device)

        # Build geometry
        self.normal_bank = build_anchor_normal_bank(
            anchors, self.train_idx,
            k_neighbors=self.k_neighbors_local,
            sigma=self.sigma_local,
        )
        self.segment_bandwidths = estimate_segment_geometric_bandwidths(
            self.train_idx, self.normal_bank,
            k_neighbors=self.k_neighbors_local,
            geometry_window_segments=self.geometry_window_segments,
        )
        for k, sp in enumerate(self.segment_bandwidths):
            log.info(
                f"segment {k} | metric_hx={sp['metric_hx']:.6g} "
                f"metric_ht={sp['metric_ht']:.6g} "
                f"sigma_x={sp['sigma_x']:.6g} sigma_t={sp['sigma_t']:.6g}"
            )

        # Build initial matchings
        self.matchings = build_initial_matchings(
            anchors,
            k_neighbors=self.k_neighbors_local,
            sigma=self.sigma_local,
            alpha_dist=self.alpha_dist_init,
            gamma_normal=self.gamma_normal_init,
            matching_method=self.matching_method,
            matching_postprocess=self.matching_postprocess,
            sinkhorn_reg=self.sinkhorn_reg,
            uot_reg=self.uot_reg,
            uot_reg_m=self.uot_reg_m,
        )
        log.info(f"Initial matching built (method={self.matching_method})")

    def on_train_epoch_end(self) -> None:
        epoch = self.current_epoch + 1  # 0-indexed → 1-indexed
        # Ablation: rematch_every <= 0 disables periodic re-matching entirely
        # (also avoids a modulo-by-zero error).
        if self.rematch_every is None or self.rematch_every <= 0:
            return
        if epoch % self.rematch_every != 0:
            return
        # Don't rematch on the very last epoch (waste of compute)
        if epoch >= self.trainer.max_epochs:
            return

        log.info(f"[Epoch {epoch}] Re-solving metric-path Hungarian matchings …")
        device = self.device
        t_probe = torch.linspace(0, 1, self.t_probe_steps, device=device).view(1, self.t_probe_steps, 1)
        was_training = self.psi_net.training
        requires_grad_flags = [param.requires_grad for param in self.psi_net.parameters()]
        for param in self.psi_net.parameters():
            param.requires_grad_(False)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        try:
            self.matchings = build_refined_matchings_from_metric_cost(
                self.train_anchors.to(device),
                self.train_idx,
                self.psi_net,
                t_probe_local=t_probe,
                normal_bank=self.normal_bank,
                segment_bandwidths=self.segment_bandwidths,
                alpha_g=self.metric_alpha,
                matching_method=self.matching_method,
                matching_postprocess=self.matching_postprocess,
                sinkhorn_reg=self.sinkhorn_reg,
                uot_reg=self.uot_reg,
                uot_reg_m=self.uot_reg_m,
                batch_i=self.rematch_batch_i,
                batch_j=self.rematch_batch_j,
                candidate_topk=self.rematch_candidate_topk,
                candidate_min_points=self.rematch_candidate_min_points,
                kernel_mode=self.metric_kernel_mode,
            )
        finally:
            for param, requires_grad in zip(self.psi_net.parameters(), requires_grad_flags):
                param.requires_grad_(requires_grad)
            self.psi_net.train(was_training)
        log.info(f"Matching re-solved (method={self.matching_method})")

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _compute_trajectory_and_velocity(
        self, start: torch.Tensor, end: torch.Tensor, t_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute PACE trajectories and velocities via autograd."""
        n, dim = start.shape
        t_count = t_base.shape[1]

        s_ext = start.unsqueeze(1).expand(n, t_count, dim)
        e_ext = end.unsqueeze(1).expand(n, t_count, dim)
        t_in = t_base.expand(n, t_count, 1).clone()

        # Flatten to 2D for GeoPathMLP (expects [B, dim])
        s_flat = s_ext.reshape(n * t_count, dim)
        e_flat = e_ext.reshape(n * t_count, dim)
        t_flat = t_in.reshape(n * t_count, 1)

        # Keep the legacy reverse-mode path in 2-D and switch to an equivalent
        # forward-mode JVP formulation only for higher-dimensional data.
        if dim <= 2:
            t_in = t_in.requires_grad_(True)
            t_flat = t_in.reshape(n * t_count, 1)
            psi_flat = self.psi_net(s_flat, e_flat, t_flat)
            psi = psi_flat.reshape(n, t_count, dim)

            xt = (1 - t_in) * s_ext + t_in * e_ext + t_in * (1 - t_in) * psi

            v_components = []
            for d in range(dim):
                vd = torch.autograd.grad(
                    outputs=xt[..., d].sum(),
                    inputs=t_in,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                v_components.append(vd)
            vt = torch.cat(v_components, dim=-1)
        else:
            xt_flat, vt_flat = compute_interpolant_and_velocity(
                self.psi_net,
                s_flat,
                e_flat,
                t_flat,
            )
            xt = xt_flat.reshape(n, t_count, dim)
            vt = vt_flat.reshape(n, t_count, dim)
        return xt, vt

    def _forward_all_segments(
        self, t_base: torch.Tensor, point_idx: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        anchors = self.train_anchors
        batch_N = point_idx.shape[0] if point_idx is not None else self.N
        X_segments, V_segments, T_global_list = [], [], []

        for k in range(self.num_segments):
            if point_idx is not None:
                x_src = anchors[k][point_idx]
                x_tgt = anchors[k + 1][self.matchings[k][point_idx]]
            else:
                x_src = anchors[k]
                x_tgt = anchors[k + 1][self.matchings[k]]
            xk, vk = self._compute_trajectory_and_velocity(x_src, x_tgt, t_base)
            X_segments.append(xk)
            V_segments.append(vk)

            t_start = float(self.train_idx[k])
            t_end = float(self.train_idx[k + 1])
            t_global = t_start + (t_end - t_start) * t_base.expand(batch_N, self.T_steps, 1)
            T_global_list.append(t_global)

        return X_segments, V_segments, T_global_list

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _ortho_loss(
        self, X_segments: list[torch.Tensor], V_segments: list[torch.Tensor],
    ) -> torch.Tensor:
        """Penalise velocity components in the normal space of the local manifold.

        In 2-D the normal space is 1-D (a single vector), but in high-D
        the normal space is (D - d_eff)-dimensional.  We use the full
        normal-space projector P_N = I - P_T so the loss generalises
        correctly to any ambient dimension.
        """
        dim = self.dim
        loss_list = []
        for k in range(self.num_segments):
            xk, vk = X_segments[k], V_segments[k]
            _, t_count, _ = xk.shape
            for t in range(t_count):
                v_t = vk[:, t, :]  # (N, D) — gradients flow through here
                if dim <= 2:
                    # Fast path: single normal vector is sufficient
                    with torch.no_grad():
                        normals_t, _, _ = estimate_local_normals_kernel(
                            xk[:, t, :].detach(),
                            k_neighbors=self.k_neighbors_local,
                            sigma=self.sigma_local,
                        )
                    dot_t = torch.sum(v_t * normals_t, dim=1)
                    loss_list.append(torch.mean(dot_t ** 2))
                else:
                    # High-dim: use full normal-space projector
                    with torch.no_grad():
                        P_N = estimate_normal_projector(
                            xk[:, t, :].detach(),
                            k_neighbors=self.k_neighbors_local,
                            sigma=self.sigma_local,
                        )  # (N, D, D)
                    v_normal = torch.bmm(P_N, v_t.unsqueeze(-1)).squeeze(-1)  # (N, D)
                    loss_list.append(torch.mean(torch.sum(v_normal ** 2, dim=1)))
        return torch.stack(loss_list).mean()

    def _global_alignment_loss(
        self,
        X_segments: list[torch.Tensor],
        V_segments: list[torch.Tensor],
    ) -> torch.Tensor:
        dim = self.dim
        X_all = torch.cat([xk.reshape(-1, dim) for xk in X_segments], dim=0)
        V_all = torch.cat([vk.reshape(-1, dim) for vk in V_segments], dim=0)

        # Per-segment sigma_x / sigma_t from bandwidths
        sigma_x_all = torch.cat([
            torch.full(
                (xk.shape[0] * xk.shape[1],),
                float(self.segment_bandwidths[k]["sigma_x"]),
                device=X_all.device,
            )
            for k, xk in enumerate(X_segments)
        ], dim=0)
        sigma_t_all = torch.cat([
            torch.full(
                (xk.shape[0] * xk.shape[1],),
                float(self.segment_bandwidths[k]["sigma_t"]),
                device=X_all.device,
            )
            for k, xk in enumerate(X_segments)
        ], dim=0)

        # Build time array for cross-segment distance, reusing T_global_list
        # We rebuild it quickly here since we need the actual global times
        batch_N = X_segments[0].shape[0]
        T_parts = []
        for k in range(self.num_segments):
            t_start = float(self.train_idx[k])
            t_end = float(self.train_idx[k + 1])
            t_local = torch.linspace(0, 1, self.T_steps, device=X_all.device)
            t_global_k = t_start + (t_end - t_start) * t_local
            T_parts.append(
                t_global_k.unsqueeze(0).expand(batch_N, self.T_steps).reshape(-1, 1)
            )
        T_all = torch.cat(T_parts, dim=0)

        dist_sq = torch.cdist(X_all, X_all, p=2) ** 2
        time_sq = torch.cdist(T_all, T_all, p=2) ** 2

        sigma_x_pair_sq = sigma_x_all[:, None] * sigma_x_all[None, :] + 1e-8
        sigma_t_pair_sq = sigma_t_all[:, None] * sigma_t_all[None, :] + 1e-8

        # Build cross-segment mask (use pre-computed buffer or build dynamically)
        if self.mask_diff_seg is not None and self.mask_diff_seg.shape[0] == X_all.shape[0]:
            mask = self.mask_diff_seg.to(X_all.device)
        else:
            seg_labels = [
                torch.full((batch_N * self.T_steps,), float(k), device=X_all.device)
                for k in range(self.num_segments)
            ]
            seg_all = torch.cat(seg_labels, dim=0)
            mask = (seg_all[:, None] != seg_all[None, :]).float()

        W_total = (
            torch.exp(-dist_sq / sigma_x_pair_sq)
            * torch.exp(-time_sq / sigma_t_pair_sq)
            * mask
        )

        V_norm = V_all / (torch.norm(V_all, dim=1, keepdim=True) + 1e-8)
        C_matrix = torch.matmul(V_norm, V_norm.T)
        loss_global = torch.sum(W_total * (1.0 - C_matrix)) / (torch.sum(W_total) + 1e-8)
        return loss_global

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        t_base = torch.linspace(0, 1, self.T_steps, device=self.device).view(1, self.T_steps, 1)

        # Extract point indices from batch when using mini-batch mode
        point_idx = batch[0].long() if self._use_minibatch else None

        X_segments, V_segments, T_global_list = self._forward_all_segments(t_base, point_idx=point_idx)

        # Metric energy loss
        loss_metric = compute_metric_energy_for_segments(
            X_segments, V_segments, T_global_list,
            self.normal_bank, self.segment_bandwidths,
            alpha_g=self.metric_alpha,
            kernel_mode=self.metric_kernel_mode,
        )

        # Global alignment loss
        loss_global = self._global_alignment_loss(X_segments, V_segments)

        # Orthogonality loss
        if self.lambda_ortho > 0:
            loss_ortho = self._ortho_loss(X_segments, V_segments)
        else:
            loss_ortho = torch.tensor(0.0, device=self.device)

        loss = (
            self.lambda_metric * loss_metric
            + self.lambda_global * loss_global
            + self.lambda_ortho * loss_ortho
        )

        self.log("PACE_Psi/train_loss", loss, prog_bar=True, logger=True)
        self.log("PACE_Psi/loss_metric", loss_metric, prog_bar=False, logger=True)
        self.log("PACE_Psi/loss_global", loss_global, prog_bar=False, logger=True)
        self.log("PACE_Psi/loss_ortho", loss_ortho, prog_bar=False, logger=True)
        return loss

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.psi_net.parameters(), lr=self.lr)
        elif self.optimizer_name == "adamw":
            return torch.optim.AdamW(self.psi_net.parameters(), lr=self.lr)
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    # ------------------------------------------------------------------
    # Dataloader (dummy — the real data is in train_anchors buffer)
    # ------------------------------------------------------------------
    def train_dataloader(self):
        """Return a dataloader over point indices, or a dummy single-batch loader.

        When ``psi_batch_size`` is set (and < N), the loader yields shuffled
        index batches so each epoch covers all N points in mini-batches.
        Otherwise a single dummy batch triggers one ``training_step`` per epoch
        (original full-batch behaviour).
        """
        if self._use_minibatch:
            indices = torch.arange(self.N)
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(indices),
                batch_size=self.psi_batch_size,
                shuffle=True,
            )
        dummy = torch.zeros(1)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(dummy),
            batch_size=1,
        )
