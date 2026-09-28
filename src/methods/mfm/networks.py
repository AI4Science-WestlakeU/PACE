"""MLP-based networks for Metric Flow Matching.

Provides:
- SimpleDenseNet: configurable MLP backbone
- GeoPathMLP: geodesic interpolant network  psi(x0, x1, t) -> correction
- VelocityNet: velocity field  v(t, x) -> dx/dt
- FlowModelWrapper: torchdyn-compatible wrapper
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional


ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "lrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        target_size: int,
        activation: str = "selu",
        batch_norm: bool = False,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
        dims = [input_size, *hidden_dims, target_size]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(ACTIVATION_MAP[activation]())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GeoPathMLP(nn.Module):
    """Geodesic interpolant network: psi(x0, x1, t) -> geodesic correction vector."""

    def __init__(
        self,
        input_dim: int,
        activation: str = "selu",
        batch_norm: bool = False,
        hidden_dims: Optional[List[int]] = None,
        time_geopath: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_geopath = time_geopath
        self.mainnet = SimpleDenseNet(
            input_size=2 * input_dim + (1 if time_geopath else 0),
            target_size=input_dim,
            activation=activation,
            batch_norm=batch_norm,
            hidden_dims=hidden_dims,
        )

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
        if self.time_geopath:
            x = torch.cat([x, t], dim=1)
        return self.mainnet(x)


class VelocityNet(SimpleDenseNet):
    """Velocity field network: v(t, x) -> velocity vector."""

    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(input_size=dim + 1, target_size=dim, *args, **kwargs)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() < 1 or t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])[:, None]
        if t.dim() < 2:
            t = t[:, None]
        inp = torch.cat([t, x], dim=-1)
        return self.model(inp)


class FlowModelWrapper(nn.Module):
    """Wraps a velocity model to torchdyn-compatible format: forward(t, x)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, t: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(t, x)
