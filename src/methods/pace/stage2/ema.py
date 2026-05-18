"""Exponential Moving Average wrapper for model parameters.

Adapted from Meta Platforms, Inc. reference implementation.
"""

import torch
import torch.nn as nn


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.model = model
        self.decay = decay

        # Propagate attributes from wrapped model (e.g. time_geopath)
        if hasattr(self.model, "time_geopath"):
            self.time_geopath = self.model.time_geopath

        self.register_buffer("num_updates", torch.tensor(0))

        self.shadow_params = nn.ParameterList(
            [
                nn.Parameter(p.clone().detach(), requires_grad=False)
                for p in model.parameters()
                if p.requires_grad
            ]
        )
        self.backup_params: list[torch.Tensor] = []

    def train(self, mode: bool = True):
        if self.training and not mode:
            # train -> eval: backup params and load shadow (EMA) params
            self.backup()
            self.copy_to_model()
        elif not self.training and mode:
            # eval -> train: restore original params
            self.restore_to_model()
        super().train(mode)
        return self

    def update_ema(self):
        self.num_updates += 1
        num_updates = self.num_updates.item()
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        with torch.no_grad():
            params = [p for p in self.model.parameters() if p.requires_grad]
            for shadow, param in zip(self.shadow_params, params):
                shadow.sub_((1 - decay) * (shadow - param))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def copy_to_model(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        for shadow, param in zip(self.shadow_params, params):
            param.data.copy_(shadow.data)

    def backup(self):
        if len(self.backup_params) > 0:
            for p, b in zip(self.model.parameters(), self.backup_params):
                b.data.copy_(p.data)
        else:
            self.backup_params = [p.clone() for p in self.model.parameters()]

    def restore_to_model(self):
        if len(self.backup_params) > 0:
            for p, b in zip(self.model.parameters(), self.backup_params):
                p.data.copy_(b.data)
            self.backup_params = []
