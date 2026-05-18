"""PACE Flow Matcher — interpolation with the PACE geodesic correction.

Core class:
    PACEFlowMatcher  – provides the same interface as MetricFlowMatcher
    but uses the PACE interpolant:

        mu_t = (1-s)*x0 + s*x1 + s*(1-s)*psi(x0, x1, s)

    where s = (t - t_min) / (t_max - t_min) is the local segment time.

    The conditional velocity is:
        u_t = 1/(t_max - t_min) * [(x1-x0) + (1-2s)*psi + s*(1-s)*d(psi)/ds]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import jvp
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, pad_t_like_x

from src.methods.pace.stage2.flow_matcher import labels_to_timesteps  # noqa: reexport


class PACEFlowMatcher(ConditionalFlowMatcher):
    """Flow matcher using the PACE s(1-s) modulation."""

    def __init__(
        self,
        psi_net: nn.Module | None = None,
        sigma: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(sigma=sigma, *args, **kwargs)
        self.psi_net = psi_net
        self.psi_net_output: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # PACE modulation:  gamma(s) = s * (1 - s)
    # ------------------------------------------------------------------
    @staticmethod
    def gamma(s: torch.Tensor) -> torch.Tensor:
        return s * (1.0 - s)

    @staticmethod
    def d_gamma(s: torch.Tensor) -> torch.Tensor:
        return 1.0 - 2.0 * s

    # ------------------------------------------------------------------
    # Interpolant  mu_t
    # ------------------------------------------------------------------
    def compute_mu_t(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        t_min: float,
        t_max: float,
    ) -> torch.Tensor:
        with torch.enable_grad():
            t_padded = pad_t_like_x(t, x0)
            s = (t_padded - t_min) / (t_max - t_min)

            if self.psi_net is None:
                return (1.0 - s) * x0 + s * x1

            s_input = pad_t_like_x(
                (t - t_min) / (t_max - t_min), x0[:, :1]
            )  # [B, 1]
            self.psi_net_output = self.psi_net(x0, x1, s_input)
            self._dpsi_ds = self._compute_dpsi_ds(self.psi_net, x0, x1, s_input)

        return (
            (1.0 - s) * x0
            + s * x1
            + self.gamma(s) * self.psi_net_output
        )

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        epsilon: torch.Tensor,
        t_min: float,
        t_max: float,
    ) -> torch.Tensor:
        mu_t = self.compute_mu_t(x0, x1, t, t_min, t_max)
        sigma_t = pad_t_like_x(self.compute_sigma_t(t), x0)
        return mu_t + sigma_t * epsilon

    # ------------------------------------------------------------------
    # Sample (t, x_t, u_t) triplets for Stage 2 training
    # ------------------------------------------------------------------
    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t_min: float,
        t_max: float,
        t: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            if t is None:
                t = torch.rand(x0.shape[0], requires_grad=True).type_as(x0)
                t = t * (t_max - t_min) + t_min
        assert len(t) == x0.shape[0]

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps, t_min, t_max)
        ut = self.compute_conditional_flow(x0, x1, t, xt, t_min, t_max)
        return t, xt, ut

    # ------------------------------------------------------------------
    # Target velocity  u_t
    # ------------------------------------------------------------------
    def compute_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        xt: torch.Tensor,
        t_min: float,
        t_max: float,
    ) -> torch.Tensor:
        del xt
        dt_inv = 1.0 / (t_max - t_min)
        t_padded = pad_t_like_x(t, x0)
        s = (t_padded - t_min) / (t_max - t_min)

        if self.psi_net is None:
            return (x1 - x0) * dt_inv

        # u_t = dt_inv * [ (x1-x0) + d_gamma(s)*psi + gamma(s)*d(psi)/ds ]
        return dt_inv * (
            (x1 - x0)
            + self.d_gamma(s) * self.psi_net_output
            + self.gamma(s) * self._dpsi_ds
        )

    # ------------------------------------------------------------------
    # d(psi)/ds via forward-mode AD
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_dpsi_ds(
        model: nn.Module,
        x0: torch.Tensor,
        x1: torch.Tensor,
        s_raw: torch.Tensor,
    ) -> torch.Tensor:
        def f(ss):
            return model(x0, x1, ss)

        _, dpsi = jvp(f, (s_raw,), (torch.ones_like(s_raw),))
        return dpsi


    # ------------------------------------------------------------------
    # Alias for compatibility with FlowNetTrain (which checks .alpha
    # and .geopath_net_output)
    # ------------------------------------------------------------------
    @property
    def alpha(self) -> float:
        return 1.0 if self.psi_net is not None else 0.0

    @property
    def geopath_net(self):
        return self.psi_net

    @geopath_net.setter
    def geopath_net(self, value):
        self.psi_net = value

    @property
    def geopath_net_output(self):
        return self.psi_net_output
