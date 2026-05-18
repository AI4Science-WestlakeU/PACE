"""Stage 2 — Flow (velocity field) network training.

Learns a velocity field v(t, x) that matches the conditional flow derived
from the (possibly geodesic-corrected) interpolant trained in Stage 1.

The frozen GeoPath network from Stage 1 enters through the MetricFlowMatcher:
when alpha != 0, the target velocity u_t includes geodesic correction terms.

PyTorch Lightning module — plug directly into a pl.Trainer.
"""

from __future__ import annotations

import os

import torch
import pytorch_lightning as pl
from torchmetrics.functional import mean_squared_error
from torchdyn.core import NeuralODE

from src.methods.pace.stage2.eval_metrics import compute_distribution_metrics
from src.methods.pace.stage2.networks import FlowModelWrapper
from src.methods.pace.stage2.ema import EMA
from src.methods.pace.stage2.flow_matcher import labels_to_timesteps


class FlowNetTrain(pl.LightningModule):
    def __init__(
        self,
        flow_matcher,
        flow_net,
        ot_sampler=None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        optimizer_name: str = "adamw",
        has_validation: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["flow_matcher", "flow_net", "ot_sampler"])
        self.flow_matcher = flow_matcher
        self.flow_net = flow_net
        self.ot_sampler = ot_sampler

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.has_validation = has_validation
        self.timesteps: list[float] | None = None

    def forward(self, t, xt):
        return self.flow_net(t, xt)

    def on_train_start(self):
        """Move the frozen geopath_net (inside flow_matcher) to the training device."""
        if getattr(self.flow_matcher, 'geopath_net', None) is not None:
            self.flow_matcher.geopath_net = self.flow_matcher.geopath_net.to(self.device)
        if getattr(self.flow_matcher, 'psi_net', None) is not None:
            self.flow_matcher.psi_net = self.flow_matcher.psi_net.to(self.device)
        if self.ot_sampler is not None and hasattr(self.ot_sampler, 'to'):
            self.ot_sampler.to(self.device)
        train_labels = sorted(self.trainer.datamodule.unique_train_labels)
        self.timesteps = labels_to_timesteps(train_labels)

    # ------------------------------------------------------------------
    # Core loss: MSE(v_theta(t, x_t),  u_t)
    # ------------------------------------------------------------------
    def _compute_loss(self, main_frames: list[torch.Tensor]) -> torch.Tensor:
        main = [x.to(self.device) for x in main_frames]
        x0s, x1s = main[:-1], main[1:]
        ts, xts, uts = self._process_flow(x0s, x1s)

        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)
        vt = self(t[:, None], xt)

        return mean_squared_error(vt, ut)

    def _process_flow(self, x0s, x1s):
        if self.ot_sampler is not None and hasattr(self.ot_sampler, 'reset'):
            self.ot_sampler.reset()
        ts, xts, uts = [], [], []
        t_start = self.timesteps[0]

        for i, (x0, x1) in enumerate(zip(x0s, x1s)):
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)

            if self.ot_sampler is not None:
                x0, x1 = self.ot_sampler.sample_plan(x0, x1, replace=True)

            t_start_next = self.timesteps[i + 1]

            t_out, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                x0, x1, t_start, t_start_next,
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
        loss = self._compute_loss(train_frames)

        if self.flow_matcher.alpha != 0 and self.flow_matcher.geopath_net_output is not None:
            self.log(
                "FlowNet/mean_psi_magnitude",
                self.flow_matcher.geopath_net_output.abs().mean(),
                on_step=False, on_epoch=True, prog_bar=True,
            )
        self.log(
            "FlowNet/train_loss",
            loss,
            on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        val_frames = self._extract_ordered_frames(batch, "val")
        val_loss = self._compute_loss(val_frames)
        self.log(
            "FlowNet/val_loss",
            val_loss,
            on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )
        return val_loss

    # ------------------------------------------------------------------
    # Test: ODE rollout to held-out frames
    # ------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        train_labels = sorted(self.trainer.datamodule.unique_train_labels)
        test_labels = sorted(self.trainer.datamodule.unique_test_labels)
        
        # Use the TRAINING time mapping (identical to training_step)
        train_timesteps = labels_to_timesteps(train_labels)
        label_to_t = {label: train_timesteps[i] for i, label in enumerate(train_labels)}

        node = NeuralODE(
            FlowModelWrapper(self.flow_net),
            solver="euler",
            sensitivity="adjoint",
            atol=1e-5,
            rtol=1e-5,
        )

        # Get train source frames
        if hasattr(self.trainer.datamodule, "selected_train_frames"):
            source_dict = self.trainer.datamodule.selected_train_frames
        else:
            source_dict = self.trainer.datamodule.train_frames

        # Build test frames from test_frames dict (keyed by numeric label)
        test_frame_dict = self.trainer.datamodule.test_frames

        results = {}
        for test_label in test_labels:
            if test_label not in test_frame_dict:
                continue
            test_frame = test_frame_dict[test_label].to(self.device)

            # Find bracketing train anchors
            prev_label = None
            next_label = None
            for tl in train_labels:
                if tl <= test_label:
                    prev_label = tl
                if tl >= test_label and next_label is None:
                    next_label = tl
            if prev_label is None or next_label is None:
                continue
            if prev_label == test_label or next_label == test_label:
                continue
            if prev_label not in source_dict or next_label not in source_dict:
                continue

            source = source_dict[prev_label].to(self.device)
            t_source = label_to_t[prev_label]
            t_target = label_to_t[next_label]
            ratio = (test_label - prev_label) / (next_label - prev_label)
            n_steps = 101

            with torch.no_grad():
                traj = node.trajectory(
                    source,
                    t_span=torch.linspace(t_source, t_target, n_steps).to(self.device),
                )
                query_idx = int(round(ratio * (n_steps - 1)))
                pred = traj[query_idx]

            metrics = compute_distribution_metrics(pred, test_frame)

            results[test_label] = {
                "pred": pred,
                "gt": test_frame,
                "mmd": metrics["mmd"],
                "w1": metrics["w1"],
                "w2": metrics["w2"],
            }

        return results

    # ------------------------------------------------------------------
    # Optimizer + EMA
    # ------------------------------------------------------------------
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.flow_net, EMA):
            self.flow_net.update_ema()

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    # ------------------------------------------------------------------
    # Batch extraction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_ordered_frames(batch, split: str) -> list[torch.Tensor]:
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

        if isinstance(data, dict):
            return [data[k] for k in sorted(data.keys())]
        if isinstance(data, (list, tuple)):
            return list(data)
        raise TypeError(f"Unexpected batch structure: {type(data)}")

    @staticmethod
    def _extract_test_frames(batch) -> dict:
        """Extract test frames from a test batch (dict or list)."""
        if isinstance(batch, dict):
            return batch
        if isinstance(batch, (list, tuple)):
            return {i: t for i, t in enumerate(batch)}
        return {}
