"""Manifold geometry estimation for PACE.

Provides kernel-PCA normal/tangent estimation, normal bank construction,
adaptive bandwidth estimation, and Riemannian metric field evaluation.
"""

from __future__ import annotations

import numpy as np
import torch


# ------------------------------------------------------------------
# Local kernel-PCA normal estimation
# ------------------------------------------------------------------

def _estimate_local_kernel_pca(
    x_t: torch.Tensor,
    k_neighbors: int = 25,
    sigma: float | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local kernel-PCA eigendecompositions for each point."""
    n, dim = x_t.shape
    dist = torch.cdist(x_t, x_t, p=2)

    k_use = min(k_neighbors, n)
    knn_dist, knn_idx = torch.topk(dist, k=k_use, largest=False, dim=1)
    neighbors = x_t[knn_idx]

    if sigma is None:
        sigma_local = torch.median(knn_dist, dim=1).values.clamp_min(eps)
    else:
        sigma_local = torch.full((n,), float(sigma), device=x_t.device)

    w = torch.exp(-(knn_dist ** 2) / (2.0 * sigma_local[:, None] ** 2 + eps))
    w = w / (w.sum(dim=1, keepdim=True) + eps)

    mu = torch.sum(neighbors * w.unsqueeze(-1), dim=1, keepdim=True)
    xc = neighbors - mu
    cov = torch.einsum("nki,nkj,nk->nij", xc, xc, w)
    cov = cov + eps * torch.eye(dim, device=x_t.device).unsqueeze(0)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    return eigvals, eigvecs


def _build_tangent_and_normal_projectors(
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    tangent_ratio: float = 0.95,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build adaptive tangent/normal projectors from kernel-PCA spectra."""
    dim = eigvals.shape[-1]
    eigvals_desc = eigvals.flip(dims=[-1])
    eigvecs_desc = eigvecs.flip(dims=[-1])

    cumvar = torch.cumsum(eigvals_desc, dim=-1)
    total_var = cumvar[:, -1:].clamp_min(eps)
    ratio = cumvar / total_var
    above = (ratio >= tangent_ratio).float()
    d_eff = above.argmax(dim=-1) + 1
    d_eff = d_eff.clamp(min=1, max=dim)
    d_max = int(d_eff.max().item())

    j_idx = torch.arange(d_max, device=eigvals.device).unsqueeze(0)
    mask_top = (j_idx < d_eff.unsqueeze(1)).float()
    v_top = eigvecs_desc[:, :, :d_max]
    v_masked = v_top * mask_top.unsqueeze(1)
    p_t = torch.bmm(v_masked, v_top.transpose(1, 2))
    identity = torch.eye(dim, device=eigvals.device).unsqueeze(0)
    p_n = identity - p_t
    return p_t, p_n, d_eff

def estimate_local_normals_kernel(
    x_t: torch.Tensor,
    k_neighbors: int = 25,
    sigma: float | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate local normals and tangents via kernel-weighted PCA.

    ``normals`` is the minimum-variance direction.
    ``tangents`` is the principal tangent direction (maximum-variance eigenvector).
    """
    eigvals, eigvecs = _estimate_local_kernel_pca(
        x_t,
        k_neighbors=k_neighbors,
        sigma=sigma,
        eps=eps,
    )
    normals = eigvecs[:, :, 0]
    tangents = eigvecs[:, :, -1]

    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + eps)
    tangents = tangents / (torch.norm(tangents, dim=1, keepdim=True) + eps)
    return normals, tangents, eigvals


@torch.no_grad()
def estimate_normal_projector(
    x_t: torch.Tensor,
    k_neighbors: int = 25,
    sigma: float | None = None,
    tangent_ratio: float = 0.95,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate the normal-space projection matrix P_N = I - P_T.

    In high-dimensional spaces (dim >> 2) the local manifold tangent space
    may have effective dimensionality d_eff > 1.  This function selects the
    top eigenvectors that explain ``tangent_ratio`` of the local variance,
    builds the tangent-space projector P_T from them, and returns P_N = I - P_T.

    Parameters
    ----------
    x_t : (N, D) point cloud at a single time step.
    k_neighbors : number of neighbours for local PCA.
    sigma : bandwidth for kernel weighting (None = adaptive median).
    tangent_ratio : cumulative variance fraction that defines tangent space.
    eps : numerical safety constant.

    Returns
    -------
    P_N : (N, D, D)  per-point normal-space projection matrix.
    """
    eigvals, eigvecs = _estimate_local_kernel_pca(
        x_t,
        k_neighbors=k_neighbors,
        sigma=sigma,
        eps=eps,
    )
    _, P_N, _ = _build_tangent_and_normal_projectors(
        eigvals,
        eigvecs,
        tangent_ratio=tangent_ratio,
        eps=eps,
    )
    return P_N


# ------------------------------------------------------------------
# Normal bank from training anchors
# ------------------------------------------------------------------

@torch.no_grad()
def build_anchor_normal_bank(
    train_anchors: torch.Tensor,
    train_idx: list[int],
    k_neighbors: int = 25,
    sigma: float | None = None,
    tangent_ratio: float = 0.95,
) -> dict:
    """Build a bank of normals/tangents/projectors from training snapshots."""
    all_x, all_t, all_normals, all_tangents, all_pn, all_pt = [], [], [], [], [], []
    x_list, t_list, normal_list, tangent_list, pn_list, pt_list = [], [], [], [], [], []

    for k in range(train_anchors.shape[0]):
        xk = train_anchors[k]
        eigvals_k, eigvecs_k = _estimate_local_kernel_pca(
            xk,
            k_neighbors=k_neighbors,
            sigma=sigma,
        )
        normals_k = eigvecs_k[:, :, 0]
        tangents_k = eigvecs_k[:, :, -1]
        normals_k = normals_k / (torch.norm(normals_k, dim=1, keepdim=True) + 1e-8)
        tangents_k = tangents_k / (torch.norm(tangents_k, dim=1, keepdim=True) + 1e-8)
        if xk.shape[-1] <= 2:
            pk = torch.einsum("ni,nj->nij", normals_k, normals_k)
            ptk = torch.einsum("ni,nj->nij", tangents_k, tangents_k)
        else:
            ptk, pk, _ = _build_tangent_and_normal_projectors(
                eigvals_k,
                eigvecs_k,
                tangent_ratio=tangent_ratio,
                eps=1e-8,
            )
        tk = torch.full((xk.shape[0], 1), float(train_idx[k]), device=xk.device)

        all_x.append(xk)
        all_t.append(tk)
        all_normals.append(normals_k)
        all_tangents.append(tangents_k)
        all_pn.append(pk)
        all_pt.append(ptk)
        x_list.append(xk)
        t_list.append(tk)
        normal_list.append(normals_k)
        tangent_list.append(tangents_k)
        pn_list.append(pk)
        pt_list.append(ptk)

    return {
        "x": torch.cat(all_x, dim=0),
        "t": torch.cat(all_t, dim=0),
        "normal": torch.cat(all_normals, dim=0),
        "tangent": torch.cat(all_tangents, dim=0),
        "P_N": torch.cat(all_pn, dim=0),
        "P_T": torch.cat(all_pt, dim=0),
        "x_list": x_list,
        "t_list": t_list,
        "normal_list": normal_list,
        "tangent_list": tangent_list,
        "P_N_list": pn_list,
        "P_T_list": pt_list,
    }


# ------------------------------------------------------------------
# Metric field evaluation
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_metric_field(
    x_query: torch.Tensor,
    t_query: torch.Tensor,
    normal_bank: dict,
    h_x: float,
    h_t: float,
    alpha: float = 8.0,
    eps: float = 1e-8,
    kernel_mode: str = "both",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate G = I + alpha * C_N via kernel-smoothed normal projectors.

    ``kernel_mode`` controls which factors enter the kernel logits:
        - ``"both"`` (default): spatial + temporal kernel (original behaviour).
        - ``"spatial"``: only the spatial term ``-dist_sq / h_x**2``.
        - ``"temporal"``: only the temporal term ``-time_sq / h_t**2``.
    """
    bank_x = normal_bank["x"]
    bank_t = normal_bank["t"]
    bank_pn = normal_bank["P_N"]
    dim = x_query.shape[-1]

    dist_sq = torch.cdist(x_query, bank_x, p=2) ** 2
    time_sq = torch.cdist(t_query, bank_t, p=2) ** 2

    if kernel_mode == "both":
        logits = -dist_sq / (h_x ** 2 + eps) - time_sq / (h_t ** 2 + eps)
    elif kernel_mode == "spatial":
        logits = -dist_sq / (h_x ** 2 + eps)
    elif kernel_mode == "temporal":
        logits = -time_sq / (h_t ** 2 + eps)
    else:
        raise ValueError(
            f"Unknown kernel_mode={kernel_mode!r}; expected 'both', 'spatial', or 'temporal'."
        )
    logits = logits - logits.max(dim=1, keepdim=True).values
    weights = torch.exp(logits)
    weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
    C_N = torch.einsum("mk,kab->mab", weights, bank_pn)
    I = torch.eye(dim, device=x_query.device).unsqueeze(0).expand(x_query.shape[0], dim, dim)
    G = I + alpha * C_N
    return G, C_N, weights


def compute_metric_energy_for_segments(
    X_segments: list[torch.Tensor],
    V_segments: list[torch.Tensor],
    T_global_list: list[torch.Tensor],
    normal_bank: dict,
    segment_bandwidths: list[dict],
    alpha_g: float,
    kernel_mode: str = "both",
) -> torch.Tensor:
    """Mean Riemannian kinetic energy across all segments."""
    dim = X_segments[0].shape[-1]
    energy_list = []
    for seg_k, (xk, vk, tg) in enumerate(zip(X_segments, V_segments, T_global_list)):
        x_flat = xk.reshape(-1, dim)
        v_flat = vk.reshape(-1, dim)
        t_flat = tg.reshape(-1, 1)
        seg_params = segment_bandwidths[seg_k]

        G_flat, _, _ = evaluate_metric_field(
            x_flat, t_flat, normal_bank,
            h_x=seg_params["metric_hx"],
            h_t=seg_params["metric_ht"],
            alpha=alpha_g,
            kernel_mode=kernel_mode,
        )
        quad = torch.einsum("bi,bij,bj->b", v_flat, G_flat, v_flat)
        energy_list.append(quad.mean())
    return torch.stack(energy_list).mean()


# ------------------------------------------------------------------
# Geometric bandwidth estimation helpers
# ------------------------------------------------------------------

@torch.no_grad()
def _frobenius_norm_2x2(mats: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.sum(mats * mats, dim=(-2, -1)) + eps)


@torch.no_grad()
def _median_positive(scale_values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    positive = scale_values[torch.isfinite(scale_values) & (scale_values > eps)]
    if positive.numel() == 0:
        return torch.tensor(1.0, device=scale_values.device, dtype=scale_values.dtype)
    return torch.median(positive).clamp_min(eps)


@torch.no_grad()
def _estimate_snapshot_spatial_geometry(
    xk: torch.Tensor,
    tangents_k: torch.Tensor,
    pn_k: torch.Tensor,
    pt_k: torch.Tensor,
    k_neighbors: int = 25,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = xk.shape[0]
    if n <= 1:
        ones = torch.ones((n,), device=xk.device, dtype=xk.dtype).clamp_min(eps)
        return ones, ones, ones

    k_use = min(k_neighbors + 1, n)
    dist_same = torch.cdist(xk, xk, p=2)
    _, knn_idx = torch.topk(dist_same, k=k_use, largest=False, dim=1)
    neighbor_idx = knn_idx[:, 1:]

    disp = xk[neighbor_idx] - xk[:, None, :]
    if xk.shape[-1] <= 2:
        tangential_step = torch.abs(
            torch.sum(disp * tangents_k[:, None, :], dim=-1)
        ).clamp_min(eps)
    else:
        tangent_disp = torch.einsum("nij,nkj->nki", pt_k, disp)
        tangential_step = torch.norm(tangent_disp, dim=-1).clamp_min(eps)
    local_spacing = torch.median(tangential_step, dim=1).values.clamp_min(eps)

    pn_diff = _frobenius_norm_2x2(pn_k[neighbor_idx] - pn_k[:, None, :, :], eps=eps)
    pt_diff = _frobenius_norm_2x2(pt_k[neighbor_idx] - pt_k[:, None, :, :], eps=eps)

    pn_rate_x = torch.median(pn_diff / tangential_step, dim=1).values.clamp_min(eps)
    pt_rate_x = torch.median(pt_diff / tangential_step, dim=1).values.clamp_min(eps)
    return local_spacing, pn_rate_x, pt_rate_x


@torch.no_grad()
def _estimate_cross_snapshot_temporal_rates(
    x_src: torch.Tensor,
    pn_src: torch.Tensor,
    pt_src: torch.Tensor,
    x_tgt: torch.Tensor,
    pn_tgt: torch.Tensor,
    pt_tgt: torch.Tensor,
    delta_t: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.cdist(x_src, x_tgt, p=2)
    nn_src_to_tgt = dist.argmin(dim=1)
    nn_tgt_to_src = dist.argmin(dim=0)

    pn_rate_src = _frobenius_norm_2x2(pn_tgt[nn_src_to_tgt] - pn_src, eps=eps) / delta_t
    pt_rate_src = _frobenius_norm_2x2(pt_tgt[nn_src_to_tgt] - pt_src, eps=eps) / delta_t
    pn_rate_tgt = _frobenius_norm_2x2(pn_src[nn_tgt_to_src] - pn_tgt, eps=eps) / delta_t
    pt_rate_tgt = _frobenius_norm_2x2(pt_src[nn_tgt_to_src] - pt_tgt, eps=eps) / delta_t

    pn_rate_t = torch.cat([pn_rate_src, pn_rate_tgt], dim=0).clamp_min(eps)
    pt_rate_t = torch.cat([pt_rate_src, pt_rate_tgt], dim=0).clamp_min(eps)
    return pn_rate_t, pt_rate_t


@torch.no_grad()
def estimate_segment_geometric_bandwidths(
    train_idx: list[int],
    normal_bank: dict,
    k_neighbors: int = 25,
    geometry_window_segments: int = 1,
    eps: float = 1e-8,
) -> list[dict]:
    """Per-segment bandwidth estimation for the metric and alignment kernels."""
    segment_bandwidths = []
    num_times = len(train_idx)
    num_segments = num_times - 1

    for k in range(num_segments):
        anchor_start = max(0, k - geometry_window_segments)
        anchor_end = min(num_times - 1, k + 1 + geometry_window_segments)

        spatial_spacings = []
        normal_spatial_rates = []
        tangent_spatial_rates = []
        normal_temporal_rates = []
        tangent_temporal_rates = []
        delta_t_values = []

        for a in range(anchor_start, anchor_end + 1):
            xa = normal_bank["x_list"][a]
            tangents_a = normal_bank["tangent_list"][a]
            pn_a = normal_bank["P_N_list"][a]
            pt_a = normal_bank["P_T_list"][a]

            spacing_a, pn_rate_x_a, pt_rate_x_a = _estimate_snapshot_spatial_geometry(
                xa, tangents_a, pn_a, pt_a, k_neighbors=k_neighbors, eps=eps,
            )
            spatial_spacings.append(spacing_a)
            normal_spatial_rates.append(pn_rate_x_a)
            tangent_spatial_rates.append(pt_rate_x_a)

        for a in range(anchor_start, anchor_end):
            b = a + 1
            xa = normal_bank["x_list"][a]
            xb = normal_bank["x_list"][b]
            pn_a = normal_bank["P_N_list"][a]
            pn_b = normal_bank["P_N_list"][b]
            pt_a = normal_bank["P_T_list"][a]
            pt_b = normal_bank["P_T_list"][b]
            delta_t_ab = abs(float(train_idx[b]) - float(train_idx[a]))
            delta_t_ab = max(delta_t_ab, float(eps))

            normal_rate_ab, tangent_rate_ab = _estimate_cross_snapshot_temporal_rates(
                xa, pn_a, pt_a, xb, pn_b, pt_b, delta_t=delta_t_ab, eps=eps,
            )
            normal_temporal_rates.append(normal_rate_ab)
            tangent_temporal_rates.append(tangent_rate_ab)
            delta_t_values.append(delta_t_ab)

        spatial_spacing = _median_positive(torch.cat(spatial_spacings, dim=0), eps=eps)
        normal_spatial_rate = _median_positive(torch.cat(normal_spatial_rates, dim=0), eps=eps)
        tangent_spatial_rate = _median_positive(torch.cat(tangent_spatial_rates, dim=0), eps=eps)
        normal_temporal_rate = _median_positive(torch.cat(normal_temporal_rates, dim=0), eps=eps)
        tangent_temporal_rate = _median_positive(torch.cat(tangent_temporal_rates, dim=0), eps=eps)
        delta_t = max(float(np.median(delta_t_values)), float(eps))

        metric_hx = max(float(spatial_spacing.item()), float((1.0 / normal_spatial_rate).item()))
        sigma_x = max(float(spatial_spacing.item()), float((1.0 / tangent_spatial_rate).item()))

        normal_speed = max(
            float((normal_temporal_rate / normal_spatial_rate.clamp_min(eps)).item()),
            float(eps),
        )
        tangent_speed = max(
            float((tangent_temporal_rate / tangent_spatial_rate.clamp_min(eps)).item()),
            float(eps),
        )
        metric_ht = metric_hx / normal_speed
        sigma_t = sigma_x / tangent_speed

        segment_bandwidths.append({
            "metric_hx": metric_hx,
            "metric_ht": metric_ht,
            "sigma_x": sigma_x,
            "sigma_t": sigma_t,
            "window_anchor_start": int(anchor_start),
            "window_anchor_end": int(anchor_end),
            "normal_speed": normal_speed,
            "tangent_speed": tangent_speed,
            "spatial_spacing": float(spatial_spacing.item()),
            "time_spacing": float(delta_t),
            "normal_spatial_rate": float(normal_spatial_rate.item()),
            "normal_temporal_rate": float(normal_temporal_rate.item()),
            "tangent_spatial_rate": float(tangent_spatial_rate.item()),
            "tangent_temporal_rate": float(tangent_temporal_rate.item()),
        })

    return segment_bandwidths
