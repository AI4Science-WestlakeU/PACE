"""Utilities for evaluating the PACE interpolant and its local-time velocity."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import jvp


def compute_psi_and_dpsi_ds(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    s_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``psi(x0, x1, s)`` and ``dpsi/ds`` for flattened inputs."""

    def _forward(ss: torch.Tensor) -> torch.Tensor:
        return model(x0, x1, ss)

    psi, dpsi = jvp(_forward, (s_raw,), (torch.ones_like(s_raw),))
    return psi, dpsi


def compute_interpolant_and_velocity(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    s_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``x_s`` and ``dx_s/ds`` for the PACE interpolant.

    The interpolant is

        x_s = (1 - s) * x0 + s * x1 + s * (1 - s) * psi(x0, x1, s)

    and its local-time velocity is

        dx_s/ds = (x1 - x0) + (1 - 2s) * psi + s * (1 - s) * dpsi/ds
    """

    psi, dpsi = compute_psi_and_dpsi_ds(model, x0, x1, s_raw)
    gamma = s_raw * (1.0 - s_raw)
    xt = (1.0 - s_raw) * x0 + s_raw * x1 + gamma * psi
    vt = (x1 - x0) + (1.0 - 2.0 * s_raw) * psi + gamma * dpsi
    return xt, vt
