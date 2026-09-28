"""Stage 1 — GeoPath (Psi) network training.

Learns geodesic interpolants that minimize tangential velocity on the
data manifold.  The loss is the mean squared Riemannian speed of the
conditional flow, normalised by a reference loss computed at epoch start
(alpha=0, i.e. the linear-interpolant baseline).

PyTorch Lightning module — plug directly into a pl.Trainer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import pytorch_lightning as pl

from src.methods.mfm.ema import EMA
from src.methods.mfm.flow_matcher import labels_to_timesteps


class GeoPathNetTrain(pl.LightningModule):
    def __init__(
        self,
        flow_matcher,
        geopath_net,
        data_manifold_metric,
        num_timesteps: int,
        ot_sampler=None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        optimizer_name: str = "adam",
        multiply_validation: int = 4,
        has_validation: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "flow_matcher", "geopath_net", "data_manifold_metric", "ot_sampler",
        ])
        self.flow_matcher = flow_matcher
        self.geopath_net = geopath_net
        self.ot_sampler = ot_sampler
        self.data_manifold_metric = data_manifold_metric
        self.num_timesteps = num_timesteps

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.multiply_validation = multiply_validation
        self.has_validation = has_validation

        self.first_loss: float | None = None
        self.timesteps: list[float] | None = None
        self.computing_reference_loss = False
        self._t_val: list[torch.Tensor] = []

    def forward(self, x0, x1, t):
        return self.geopath_net(x0, x1, t)

    # ------------------------------------------------------------------
    # Reference loss (alpha=0 baseline for normalization)
    # ------------------------------------------------------------------
    def on_train_start(self):
        train_labels = sorted(self.trainer.datamodule.unique_train_labels)
        self.timesteps = labels_to_timesteps(train_labels)
        self.first_loss = self._compute_initial_loss()

    def _compute_initial_loss(self) -> float:
        self.geopath_net.train(mode=False)
        total_loss = 0.0
        total_count = 0
        self.computing_reference_loss = True

        with torch.no_grad():
            old_alpha = self.flow_matcher.alpha
            self.flow_matcher.alpha = 0

            for batch in self.trainer.datamodule.train_dataloader():
                train_frames = self._extract_ordered_frames(batch, "train")
                metric_frames = self._extract_ordered_frames(batch, "metric")
                loss = self._compute_loss(train_frames, metric_frames)
                total_loss += loss.item()
                total_count += 1

            self.flow_matcher.alpha = old_alpha

        self.computing_reference_loss = False
        self.geopath_net.train(mode=True)
        return total_loss / total_count if total_count > 0 else 1.0

    # ------------------------------------------------------------------
    # Core loss computation
    # ------------------------------------------------------------------
    def _compute_loss(
        self,
        main_frames: list[torch.Tensor],
        metric_frames: list[torch.Tensor],
    ) -> torch.Tensor:
        main = [x.to(self.device) for x in main_frames]
        metric = [x.to(self.device) for x in metric_frames]

        x0s, x1s = main[:-1], main[1:]
        s0s, s1s = metric[:-1], metric[1:]

        ts, xts, uts = self._process_flow(x0s, x1s)

        velocities = []
        for i in range(len(ts)):
            samples = torch.cat([s0s[i], s1s[i]], dim=0)
            vel = self.data_manifold_metric.calculate_velocity(
                xts[i], uts[i], samples, i
            )
            velocities.append(vel)

        loss = torch.mean(torch.cat(velocities) ** 2)
        self.log(
            "GeoPathNet/mean_velocity",
            loss,
            on_step=False, on_epoch=True, prog_bar=True,
        )
        return loss

    def _process_flow(
        self,
        x0s: list[torch.Tensor],
        x1s: list[torch.Tensor],
    ):
        ts, xts, uts = [], [], []
        t_start = self.timesteps[0]

        for i, (x0, x1) in enumerate(zip(x0s, x1s)):
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)

            if self.trainer.validating or self.computing_reference_loss:
                reps = (self.multiply_validation,) + (1,) * (x0.dim() - 1)
                x0 = x0.repeat(*reps)
                x1 = x1.repeat(*reps)

            if self.ot_sampler is not None:
                x0, x1 = self.ot_sampler.sample_plan(x0, x1, replace=True)

            t_start_next = self.timesteps[i + 1]

            t = None
            if self.trainer.validating or self.computing_reference_loss:
                if i < len(self._t_val) and self._t_val[i].shape[0] == x0.shape[0]:
                    t = self._t_val[i]
                else:
                    batch_size = x0.shape[0]
                    t = torch.rand(batch_size, requires_grad=True)
                    if i < len(self._t_val):
                        self._t_val[i] = t
                    else:
                        self._t_val.append(t)

            t_out, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                x0, x1, t_start, t_start_next,
                training_geopath_net=True, t=t,
            )
            ts.append(t_out)
            xts.append(xt)
            uts.append(ut)
            t_start = t_start_next

        return ts, xts, uts

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        train_frames = self._extract_ordered_frames(batch, "train")
        metric_frames = self._extract_ordered_frames(batch, "metric")
        loss = self._compute_loss(train_frames, metric_frames)
        if self.first_loss:
            loss = loss / self.first_loss

        self.log(
            "GeoPathNet/mean_psi_magnitude",
            self.flow_matcher.geopath_net_output.abs().mean(),
            on_step=False, on_epoch=True, prog_bar=True,
        )
        self.log(
            "GeoPathNet/train_loss",
            loss,
            on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        train_frames = self._extract_ordered_frames(batch, "val")
        metric_frames = self._extract_ordered_frames(batch, "metric")
        loss = self._compute_loss(train_frames, metric_frames)
        if self.first_loss:
            loss = loss / self.first_loss

        self.log(
            "GeoPathNet/val_loss",
            loss,
            on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )
        return loss

    # ------------------------------------------------------------------
    # Optimizer + EMA
    # ------------------------------------------------------------------
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.geopath_net, EMA):
            self.geopath_net.update_ema()

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.geopath_net.parameters(), lr=self.lr)
        elif self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.geopath_net.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    # ------------------------------------------------------------------
    # Helpers: extract ordered frames from MFM-wrapped batch
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_ordered_frames(
        batch, split: str
    ) -> list[torch.Tensor]:
        """Extract time-ordered frame tensors from an MFM-wrapped batch.

        The wrapper produces batches like:
            {"train_samples": {t_0: tensor, t_2: tensor, ...},
             "metric_samples": {t_0: tensor, t_2: tensor, ...}}

        For "train"  -> batch["train_samples"]
        For "val"    -> batch["val_samples"]
        For "metric" -> batch["metric_samples"]
        """
        if isinstance(batch, tuple) and len(batch) == 3 and isinstance(batch[0], dict):
            batch = batch[0]

        if split == "train":
            data = batch["train_samples"]
        elif split == "val":
            data = batch["val_samples"]
        elif split == "metric":
            data = batch["metric_samples"]
        else:
            raise ValueError(f"Unknown split: {split}")

        if isinstance(data, tuple) and len(data) == 3 and isinstance(data[0], (dict, list, tuple)):
            data = data[0]

        # Handle CombinedLoader output: could be dict or list/tuple
        if isinstance(data, dict):
            # Sort by key name (t_0, t_2, t_4, ...) to ensure order
            return [data[k] for k in sorted(data.keys())]
        if isinstance(data, (list, tuple)):
            return list(data)
        raise TypeError(f"Unexpected batch structure: {type(data)}")
