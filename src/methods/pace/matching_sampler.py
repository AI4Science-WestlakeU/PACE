"""Pre-computed matching sampler for Stage 2 of PACE.

Uses the Hungarian matchings from Stage 1's psi training to provide
geometrically meaningful (x0, x1) pairs for the velocity field training,
instead of random independent sampling per timepoint.
"""

from __future__ import annotations

import torch


class PrecomputedMatchingSampler:
    """OT-sampler-compatible wrapper around pre-computed matchings.

    Replaces random pairing in Stage 2 with matched pairs from Stage 1.
    Call ``reset()`` before iterating over segments in each training step.
    """

    def __init__(
        self,
        train_anchors: torch.Tensor,
        matchings: list[torch.Tensor],
    ):
        self.train_anchors = train_anchors  # [T, N, dim]
        self.matchings = matchings           # list of T-1 index tensors
        self._segment_idx = 0

    def reset(self) -> None:
        self._segment_idx = 0

    def to(self, device: torch.device) -> "PrecomputedMatchingSampler":
        self.train_anchors = self.train_anchors.to(device)
        self.matchings = [m.to(device) for m in self.matchings]
        return self

    def sample_plan(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        replace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = self._segment_idx
        self._segment_idx += 1

        N = self.train_anchors.shape[1]
        batch_size = x0.shape[0]

        if replace:
            indices = torch.randint(0, N, (batch_size,), device=self.train_anchors.device)
        else:
            indices = torch.randperm(N, device=self.train_anchors.device)[:batch_size]

        matched_x0 = self.train_anchors[k][indices]
        matched_x1 = self.train_anchors[k + 1][self.matchings[k][indices]]

        return matched_x0, matched_x1
