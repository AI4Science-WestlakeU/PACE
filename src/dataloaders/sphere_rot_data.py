"""Synthetic SphereRot datamodule for controlled ground-truth evaluation.

Generates points on S^2 that rotate rigidly about the z-axis.  The ground-truth
metric is ``G_true(x) = I + alpha * x x^T`` so the normal direction has cost
``1 + alpha``.  This lets us test whether PACE's data-driven local-PCA metric
recovers the exact normal/tangent structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.dataloaders.balanced_timepoint_data import BalancedTimepointDataModule


class SphereRotDataModule(BalancedTimepointDataModule):
    """DataModule that generates the SphereRot benchmark on the fly.

    Config fields (all optional):
    - sphere_n_points      : number of seed points on S^2 (default 2048)
    - sphere_n_timepoints  : number of snapshots (default 9)
    - sphere_omega         : total rotation angle in radians (default pi/2)
    - sphere_alpha_true    : true normal penalty alpha (default 8.0)
    - sphere_noise_sigma   : projected normal noise std (default 0.0)
    - sphere_seed          : random seed (default 42)

    The generated frames are 3-D points on (noisy) S^2.  Timepoint labels are
    integers ``0, 1, ..., n_timepoints - 1``.
    """

    def __init__(self, args: Any):
        self.n_points = int(getattr(args, "sphere_n_points", 2048))
        self.n_timepoints = int(getattr(args, "sphere_n_timepoints", 9))
        self.omega = float(getattr(args, "sphere_omega", 0.5 * np.pi))
        self.alpha_true = float(getattr(args, "sphere_alpha_true", 8.0))
        self.noise_sigma = float(getattr(args, "sphere_noise_sigma", 0.0))
        self.sphere_seed = int(getattr(args, "sphere_seed", 42))
        super().__init__(args)

    def _load_timepoint_frames(self) -> dict[Any, np.ndarray]:
        rng = np.random.default_rng(self.sphere_seed)

        # Uniform samples on S^2 via normal-normalization.
        seeds = rng.standard_normal((self.n_points, 3)).astype(np.float32)
        seeds = seeds / (np.linalg.norm(seeds, axis=1, keepdims=True) + 1e-8)

        # Rotation matrix about z.
        frames: dict[Any, np.ndarray] = {}
        for t in range(self.n_timepoints):
            tau = t / max(self.n_timepoints - 1, 1)
            theta = self.omega * tau
            R = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            x = seeds @ R.T
            if self.noise_sigma > 0:
                eps = rng.normal(0.0, self.noise_sigma, size=x.shape).astype(np.float32)
                x = x + eps
                x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            frames[t] = x

        # Store particle IDs so trajectory-level metrics (ADE) can align seeds.
        particle_ids = np.arange(self.n_points, dtype=np.int64)
        self.particle_ids = particle_ids

        # Store true geometry helpers for downstream evaluation.
        self.true_metric_fn = lambda x: np.eye(3, dtype=np.float32)[None, :, :] + self.alpha_true * np.einsum(
            "ni,nj->nij", x, x
        )
        self.true_velocity_fn = lambda x: np.cross(
            np.array([0.0, 0.0, self.omega], dtype=np.float32), x
        )
        return frames

    def _load_optional_array(self, data, key):
        """Override to return the generated particle IDs when requested."""
        if key == "particle_ids":
            return getattr(self, "particle_ids", None)
        return None


@torch.no_grad()
def true_sphere_metric(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return ``I + alpha * x x^T`` for points on S^2."""
    dim = x.shape[-1]
    I = torch.eye(dim, device=x.device, dtype=x.dtype).unsqueeze(0)
    return I + alpha * torch.einsum("ni,nj->nij", x, x)


@torch.no_grad()
def true_sphere_velocity(x: torch.Tensor, omega: float) -> torch.Tensor:
    """Rigid rotation velocity about the z-axis: v = omega * e_z × x."""
    e_z = torch.tensor([0.0, 0.0, 1.0], device=x.device, dtype=x.dtype)
    return omega * torch.cross(e_z.expand_as(x), x, dim=-1)
