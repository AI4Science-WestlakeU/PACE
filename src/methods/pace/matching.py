"""Matching utilities for PACE.

Provides initial (distance + normal-penalty) matching and refined
metric-path-cost matching using the learned psi network.

Supported matching solvers:
    - ``hungarian``: exact linear sum assignment (scipy)
    - ``emd``:       Earth Mover's Distance / network simplex (POT)
    - ``sinkhorn``:  entropic-regularised OT (POT)
    - ``uot``:       unbalanced Sinkhorn OT (POT)

For soft solvers (emd, sinkhorn, uot) the transport plan is converted to
a hard 1-to-1 matching via ``argmax`` (deterministic) or ``sample``
(stochastic weighted sampling).
"""

from __future__ import annotations

import logging

import numpy as np
import ot as pot
import torch
from scipy.optimize import linear_sum_assignment

from src.methods.pace.geometry import (
    estimate_local_normals_kernel,
    evaluate_metric_field,
)
from src.methods.pace.interpolant import compute_interpolant_and_velocity

log = logging.getLogger(__name__)

_VALID_METHODS = {"hungarian", "emd", "sinkhorn", "uot"}
_VALID_POSTPROCESS = {"argmax", "sample"}


def _uniform_marginals_like(cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build uniform source/target marginals on the same device as ``cost_matrix``."""
    n, m = cost_matrix.shape
    dtype = cost_matrix.dtype if torch.is_floating_point(cost_matrix) else torch.float32
    device = cost_matrix.device
    a = torch.full((n,), 1.0 / max(n, 1), device=device, dtype=dtype)
    b = torch.full((m,), 1.0 / max(m, 1), device=device, dtype=dtype)
    return a, b


# ------------------------------------------------------------------
# Core solvers
# ------------------------------------------------------------------

@torch.no_grad()
def hungarian_from_cost(cost_matrix: torch.Tensor) -> torch.Tensor:
    cost_np = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    match_idx = np.zeros(cost_np.shape[0], dtype=np.int64)
    match_idx[row_ind] = col_ind
    return torch.tensor(match_idx, dtype=torch.long, device=cost_matrix.device)


@torch.no_grad()
def emd_from_cost(cost_matrix: torch.Tensor) -> np.ndarray:
    """Exact OT via POT's network simplex. Returns N×N transport plan."""
    cost_np = cost_matrix.detach().cpu().double().numpy()
    n, m = cost_np.shape
    a = pot.unif(n)
    b = pot.unif(m)
    return pot.emd(a, b, cost_np, numItermax=500_000)


@torch.no_grad()
def sinkhorn_from_cost(
    cost_matrix: torch.Tensor, reg: float = 0.05,
) -> torch.Tensor:
    """Entropic-regularised OT via Sinkhorn on the current torch device."""
    cost_t = cost_matrix.detach()
    # Normalize cost matrix to avoid numerical overflow in exp(-C/reg)
    cost_max = cost_t.max()
    if cost_max > 0:
        cost_t = cost_t / cost_max
    a, b = _uniform_marginals_like(cost_t)
    plan = pot.sinkhorn(a, b, cost_t, reg=reg, numItermax=5000,
                        method='sinkhorn_log')
    if isinstance(plan, np.ndarray):
        plan = torch.as_tensor(plan, device=cost_t.device, dtype=cost_t.dtype)
    return plan


@torch.no_grad()
def uot_from_cost(
    cost_matrix: torch.Tensor,
    reg: float = 0.05,
    reg_m: float = 1.0,
) -> torch.Tensor:
    """Unbalanced Sinkhorn OT on the current torch device."""
    cost_t = cost_matrix.detach()
    # Normalize cost matrix to avoid numerical overflow
    cost_max = cost_t.max()
    if cost_max > 0:
        cost_t = cost_t / cost_max
    a, b = _uniform_marginals_like(cost_t)
    plan = pot.unbalanced.sinkhorn_unbalanced(
        a, b, cost_t, reg=reg, reg_m=reg_m, numItermax=5000,
        method='sinkhorn_log',
    )
    if isinstance(plan, np.ndarray):
        plan = torch.as_tensor(plan, device=cost_t.device, dtype=cost_t.dtype)
    return plan


# ------------------------------------------------------------------
# Plan → hard matching converters
# ------------------------------------------------------------------

def plan_to_argmax(plan: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Per-row argmax of the transport plan → 1-D index tensor."""
    if isinstance(plan, torch.Tensor):
        return plan.argmax(dim=1).to(device=device, dtype=torch.long)
    match_idx = plan.argmax(axis=1).astype(np.int64)
    return torch.tensor(match_idx, dtype=torch.long, device=device)


def plan_to_sample(plan: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Weighted random sample per row of the transport plan → 1-D index tensor."""
    if isinstance(plan, torch.Tensor):
        plan = plan.detach().cpu().numpy()

    n, m = plan.shape
    match_idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        row = plan[i]
        total = row.sum()
        if total > 0:
            probs = row / total
        else:
            probs = np.ones(m) / m
        match_idx[i] = np.random.choice(m, p=probs)
    return torch.tensor(match_idx, dtype=torch.long, device=device)


# ------------------------------------------------------------------
# Unified dispatcher
# ------------------------------------------------------------------

@torch.no_grad()
def solve_matching(
    cost_matrix: torch.Tensor,
    method: str = "hungarian",
    postprocess: str = "argmax",
    *,
    sinkhorn_reg: float = 0.05,
    uot_reg: float = 0.05,
    uot_reg_m: float = 1.0,
) -> torch.Tensor:
    """Solve a matching from a cost matrix using the selected algorithm.

    Returns a 1-D ``match_idx`` tensor of shape ``[N]`` where
    ``match_idx[i]`` is the index of the target point matched to source
    point *i*.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown matching method '{method}'. Choose from {_VALID_METHODS}."
        )
    if postprocess not in _VALID_POSTPROCESS:
        raise ValueError(
            f"Unknown postprocess '{postprocess}'. Choose from {_VALID_POSTPROCESS}."
        )
    device = cost_matrix.device

    if method == "hungarian":
        return hungarian_from_cost(cost_matrix)

    # Compute soft transport plan
    if method == "emd":
        plan = emd_from_cost(cost_matrix)
    elif method == "sinkhorn":
        plan = sinkhorn_from_cost(cost_matrix, reg=sinkhorn_reg)
    elif method == "uot":
        plan = uot_from_cost(cost_matrix, reg=uot_reg, reg_m=uot_reg_m)
    else:
        raise ValueError(method)  # unreachable

    # Convert soft plan to hard matching
    if postprocess == "argmax":
        return plan_to_argmax(plan, device)
    else:
        return plan_to_sample(plan, device)


# ------------------------------------------------------------------
# Initial matching: distance + normal projection penalty
# ------------------------------------------------------------------

@torch.no_grad()
def build_initial_cost_matrix(
    x_src: torch.Tensor,
    x_tgt: torch.Tensor,
    k_neighbors: int = 25,
    sigma: float | None = None,
    alpha_dist: float = 1.0,
    gamma_normal: float = 0.35,
    eps: float = 1e-8,
) -> torch.Tensor:
    normals_src, _, _ = estimate_local_normals_kernel(
        x_src, k_neighbors=k_neighbors, sigma=sigma,
    )
    d = x_tgt.unsqueeze(0) - x_src.unsqueeze(1)
    dist_sq = torch.sum(d ** 2, dim=-1)

    d_unit = d / (torch.norm(d, dim=-1, keepdim=True) + eps)
    normal_proj = torch.sum(d_unit * normals_src.unsqueeze(1), dim=-1)
    normal_penalty = normal_proj ** 2

    dist_scale = torch.median(torch.sqrt(dist_sq + eps)).clamp_min(eps)
    dist_term = dist_sq / (dist_scale ** 2 + eps)
    return alpha_dist * dist_term + gamma_normal * normal_penalty


@torch.no_grad()
def build_initial_matchings(
    train_anchors: torch.Tensor,
    k_neighbors: int = 25,
    sigma: float | None = None,
    alpha_dist: float = 1.0,
    gamma_normal: float = 0.35,
    matching_method: str = "hungarian",
    matching_postprocess: str = "argmax",
    sinkhorn_reg: float = 0.05,
    uot_reg: float = 0.05,
    uot_reg_m: float = 1.0,
) -> list[torch.Tensor]:
    log.info(f"Building initial matchings with method={matching_method}, postprocess={matching_postprocess}")
    matchings = []
    for k in range(train_anchors.shape[0] - 1):
        cost = build_initial_cost_matrix(
            train_anchors[k],
            train_anchors[k + 1],
            k_neighbors=k_neighbors,
            sigma=sigma,
            alpha_dist=alpha_dist,
            gamma_normal=gamma_normal,
        )
        matchings.append(solve_matching(
            cost,
            method=matching_method,
            postprocess=matching_postprocess,
            sinkhorn_reg=sinkhorn_reg,
            uot_reg=uot_reg,
            uot_reg_m=uot_reg_m,
        ))
    return matchings


# ------------------------------------------------------------------
# Refined matching: metric-weighted path-action cost
# ------------------------------------------------------------------

@torch.no_grad()
def compute_pairwise_metric_path_cost(
    x_src: torch.Tensor,
    x_tgt: torch.Tensor,
    model: torch.nn.Module,
    t_probe_local: torch.Tensor,
    time_start: float,
    time_end: float,
    normal_bank: dict,
    h_x: float,
    h_t: float,
    alpha_g: float,
    batch_i: int = 32,
    batch_j: int = 64,
    candidate_topk: int | None = None,
    candidate_min_points: int = 2000,
    kernel_mode: str = "both",
) -> torch.Tensor:
    model.eval()
    dim = x_src.shape[-1]
    n = x_src.shape[0]
    m = x_tgt.shape[0]
    m_probe = t_probe_local.shape[1]
    if candidate_topk is not None and candidate_topk <= 0:
        raise ValueError("candidate_topk must be positive when provided.")
    if candidate_min_points <= 0:
        raise ValueError("candidate_min_points must be positive.")

    use_candidate_pruning = (
        candidate_topk is not None
        and int(candidate_topk) < m
        and max(n, m) >= int(candidate_min_points)
    )
    if use_candidate_pruning:
        topk = min(int(candidate_topk), m)
        # Keep a dense fallback cost for the solver, but only spend the
        # expensive metric-path evaluation budget on the nearest candidates.
        out_cost = torch.cdist(x_src, x_tgt, p=2) ** 2
        candidate_cols = out_cost.topk(k=topk, largest=False, dim=1).indices

        for i0 in range(0, n, batch_i):
            i1 = min(i0 + batch_i, n)
            row_candidates = candidate_cols[i0:i1]
            unique_cols, inverse = torch.unique(
                row_candidates.reshape(-1),
                sorted=True,
                return_inverse=True,
            )
            refined_block = compute_pairwise_metric_path_cost(
                x_src[i0:i1],
                x_tgt[unique_cols],
                model,
                t_probe_local=t_probe_local,
                time_start=time_start,
                time_end=time_end,
                normal_bank=normal_bank,
                h_x=h_x,
                h_t=h_t,
                alpha_g=alpha_g,
                batch_i=max(1, i1 - i0),
                batch_j=batch_j,
                candidate_topk=None,
                candidate_min_points=candidate_min_points,
                kernel_mode=kernel_mode,
            )

            bi = i1 - i0
            candidate_pos = inverse.reshape(bi, topk)
            row_idx_local = torch.arange(bi, device=x_src.device).unsqueeze(1).expand_as(candidate_pos)
            refined_selected = refined_block[row_idx_local, candidate_pos]
            row_idx_global = torch.arange(i0, i1, device=x_src.device).unsqueeze(1).expand_as(row_candidates)
            out_cost[row_idx_global, row_candidates] = refined_selected
        return out_cost

    out_cost = torch.zeros(n, m, device=x_src.device)

    for i0 in range(0, n, batch_i):
        i1 = min(i0 + batch_i, n)
        xs = x_src[i0:i1]
        bi = xs.shape[0]

        for j0 in range(0, m, batch_j):
            j1 = min(j0 + batch_j, m)
            xt = x_tgt[j0:j1]
            bj = xt.shape[0]

            s_pair = xs[:, None, :].expand(bi, bj, dim).reshape(bi * bj, dim)
            e_pair = xt[None, :, :].expand(bi, bj, dim).reshape(bi * bj, dim)
            tt = t_probe_local.expand(bi * bj, m_probe, 1).clone()

            s_ext = s_pair.unsqueeze(1).expand(bi * bj, m_probe, dim)
            e_ext = e_pair.unsqueeze(1).expand(bi * bj, m_probe, dim)

            # Flatten to 2D for GeoPathMLP
            total = bi * bj * m_probe
            s_flat = s_ext.reshape(total, dim)
            e_flat = e_ext.reshape(total, dim)
            t_flat = tt.reshape(total, 1)

            # Preserve the existing reverse-mode path in 2-D, but keep the
            # high-dimensional branch fully in no-grad mode during rematching.
            if dim <= 2:
                with torch.enable_grad():
                    tt = tt.requires_grad_(True)
                    t_flat = tt.reshape(total, 1)
                    psi = model(
                        s_flat,
                        e_flat,
                        t_flat,
                    ).reshape(bi * bj, m_probe, dim)

                    x_curve = (1 - tt) * s_ext + tt * e_ext + tt * (1 - tt) * psi

                    v_components = []
                    for d in range(dim):
                        vd = torch.autograd.grad(
                            outputs=x_curve[..., d].sum(),
                            inputs=tt,
                            create_graph=False,
                            retain_graph=(d < dim - 1),
                        )[0]
                        v_components.append(vd)
                    v_curve = torch.cat(v_components, dim=-1)
                    x_flat = x_curve.reshape(-1, dim)
                    v_flat = v_curve.reshape(-1, dim)
            else:
                x_flat, v_flat = compute_interpolant_and_velocity(
                    model,
                    s_flat,
                    e_flat,
                    t_flat,
                )

            t_global = (time_start + (time_end - time_start) * tt).reshape(-1, 1)

            G_flat, _, _ = evaluate_metric_field(
                x_flat, t_global, normal_bank,
                h_x=h_x, h_t=h_t, alpha=alpha_g,
                kernel_mode=kernel_mode,
            )
            quad = torch.einsum("bi,bij,bj->b", v_flat, G_flat, v_flat)
            action = quad.view(bi * bj, m_probe).mean(dim=1)
            out_cost[i0:i1, j0:j1] = action.reshape(bi, bj)

    return out_cost


@torch.no_grad()
def build_refined_matchings_from_metric_cost(
    train_anchors: torch.Tensor,
    train_idx: list[int],
    model: torch.nn.Module,
    t_probe_local: torch.Tensor,
    normal_bank: dict,
    segment_bandwidths: list[dict],
    alpha_g: float,
    matching_method: str = "hungarian",
    matching_postprocess: str = "argmax",
    sinkhorn_reg: float = 0.05,
    uot_reg: float = 0.05,
    uot_reg_m: float = 1.0,
    batch_i: int = 32,
    batch_j: int = 64,
    candidate_topk: int | None = None,
    candidate_min_points: int = 2000,
    kernel_mode: str = "both",
) -> list[torch.Tensor]:
    matchings = []
    for k in range(train_anchors.shape[0] - 1):
        seg_params = segment_bandwidths[k]
        segment_n = max(train_anchors[k].shape[0], train_anchors[k + 1].shape[0])
        if (
            candidate_topk is not None
            and int(candidate_topk) < train_anchors[k + 1].shape[0]
            and segment_n >= int(candidate_min_points)
        ):
            log.info(
                "segment %d rematch: refine top-%d/%d candidates per source (N=%d)",
                k,
                int(candidate_topk),
                train_anchors[k + 1].shape[0],
                segment_n,
            )
        cost = compute_pairwise_metric_path_cost(
            train_anchors[k],
            train_anchors[k + 1],
            model,
            t_probe_local=t_probe_local,
            time_start=float(train_idx[k]),
            time_end=float(train_idx[k + 1]),
            normal_bank=normal_bank,
            h_x=seg_params["metric_hx"],
            h_t=seg_params["metric_ht"],
            alpha_g=alpha_g,
            batch_i=batch_i,
            batch_j=batch_j,
            candidate_topk=candidate_topk,
            candidate_min_points=candidate_min_points,
            kernel_mode=kernel_mode,
        )
        matchings.append(solve_matching(
            cost,
            method=matching_method,
            postprocess=matching_postprocess,
            sinkhorn_reg=sinkhorn_reg,
            uot_reg=uot_reg,
            uot_reg_m=uot_reg_m,
        ))
    return matchings
