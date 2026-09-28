"""Geometry-aware metrics for Metric Flow Matching.

Provides:
- land_metric_tensor: fast kernel-based diagonal metric (no learning)
- DataManifoldMetric: orchestrates metric computation for the Riemannian velocity loss
"""

from __future__ import annotations

from typing import List, Optional

import torch


# ---------------------------------------------------------------------------
# LAND metric  (Local Adaptive Normalizing metric — Diagonal)
# ---------------------------------------------------------------------------

def _weighting_function(
    x: torch.Tensor, samples: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Gaussian kernel weights between query points and reference samples.

    Args:
        x: (B, D) query points
        samples: (N, D) reference samples
        gamma: kernel bandwidth

    Returns:
        (B, N) weight matrix
    """
    pairwise_sq_dist = ((x[:, None, :] - samples[None, :, :]) ** 2).sum(-1)
    return torch.exp(-pairwise_sq_dist / (2 * gamma**2))


def land_metric_tensor(
    x: torch.Tensor,
    samples: torch.Tensor,
    gamma: float,
    rho: float,
) -> torch.Tensor:
    """Compute diagonal LAND metric tensor at query points.

    M_dd^{-1}(x) = 1 / (sum_n w_n (s_n_d - x_d)^2 + rho)

    Args:
        x: (B, D) query points on the interpolant
        samples: (N, D) data samples (from both source and target frames)
        gamma: kernel bandwidth
        rho: regularization preventing division by zero

    Returns:
        (B, D) inverse diagonal metric entries
    """
    weights = _weighting_function(x, samples, gamma)          # (B, N)
    differences = samples[None, :, :] - x[:, None, :]         # (B, N, D)
    sq_diff = differences ** 2                                 # (B, N, D)
    M_dd_diag = torch.einsum("bn,bnd->bd", weights, sq_diff) + rho  # (B, D)
    return 1.0 / M_dd_diag                                    # (B, D)


# ---------------------------------------------------------------------------
# DataManifoldMetric — wraps metric computation for the velocity loss
# ---------------------------------------------------------------------------

class DataManifoldMetric:
    """Orchestrates metric computation for the Riemannian velocity loss.

    Currently supports:
    - "land": fast kernel-based LAND metric (no learning required)
    - "rbf": placeholder for learned RBF metric (not yet implemented)

    Args:
        velocity_metric: "land" or "rbf"
        gammas: list of kernel bandwidths (one per transition or a single shared value)
        rho: regularization
        alpha_metric: exponent applied to the raw metric tensor
    """

    def __init__(
        self,
        velocity_metric: str = "land",
        gammas: list[float] | None = None,
        rho: float = 1e-4,
        alpha_metric: float = 0.3,
    ):
        self.metric_type = velocity_metric
        self.gammas = gammas if gammas is not None else [0.1]
        self.rho = rho
        self.alpha_metric = alpha_metric

        if self.metric_type not in ("land", "rbf"):
            raise ValueError(f"Unknown velocity_metric: {self.metric_type!r}")
        if self.metric_type == "rbf":
            raise NotImplementedError(
                "RBF metric learning is not yet implemented. Use 'land'."
            )

    def _gamma_for_transition(self, timestep: int) -> float:
        if len(self.gammas) == 1:
            return self.gammas[0]
        return self.gammas[min(timestep, len(self.gammas) - 1)]

    def calculate_metric(
        self,
        x_t: torch.Tensor,
        samples: torch.Tensor,
        current_timestep: int,
    ) -> torch.Tensor:
        """Compute the diagonal metric tensor at points x_t.

        Returns:
            (B, D) metric tensor (inverse diagonal) raised to alpha_metric power.
        """
        gamma = self._gamma_for_transition(current_timestep)
        M_dd = land_metric_tensor(x_t, samples, gamma, self.rho)
        return M_dd ** self.alpha_metric

    def calculate_velocity(
        self,
        x_t: torch.Tensor,
        u_t: torch.Tensor,
        samples: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """Compute Riemannian speed:  ||u_t||_M  =  sqrt( sum_d  u_d^2 * M_dd ).

        Args:
            x_t: (B, D) interpolated points
            u_t: (B, D) conditional velocity
            samples: (N, D) reference data samples
            timestep: index of the current time transition

        Returns:
            (B,) Riemannian speed per sample
        """
        if u_t.dim() > 2:
            u_t = u_t.reshape(u_t.shape[0], -1)
            x_t = x_t.reshape(x_t.shape[0], -1)
        M_dd = self.calculate_metric(x_t, samples, timestep)
        return torch.sqrt(((u_t ** 2) * M_dd).sum(dim=-1))
