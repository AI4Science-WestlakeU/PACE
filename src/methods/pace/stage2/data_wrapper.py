"""PACE Stage 2 DataModule wrapper.

Wraps any BalancedTimepointDataModule to produce the nested batch format
expected by GeoPathNetTrain and FlowNetTrain:

    {
      "train_samples": {t_0: tensor, t_2: tensor, ...},
      "metric_samples": {t_0: tensor, t_2: tensor, ...},
    }

Does NOT modify the underlying DataModule -- just overrides the dataloader
methods to produce the CombinedLoader outputs expected by PACE Stage 2.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

try:
    from pytorch_lightning.trainer.supporters import CombinedLoader
except Exception:
    try:
        from lightning.pytorch.utilities.combined_loader import CombinedLoader
    except Exception:
        from pytorch_lightning.utilities.combined_loader import CombinedLoader


class PaceDataModuleWrapper(pl.LightningDataModule):
    """Thin wrapper that adapts a BalancedTimepointDataModule for PACE training.

    The wrapped datamodule handles all data loading / splitting / whitening.
    This wrapper only reorganises the dataloaders into the nested dict format
    that the Stage 2 LightningModules expect.
    """

    def __init__(self, base_datamodule: pl.LightningDataModule):
        super().__init__()
        self.base = base_datamodule

    # ------------------------------------------------------------------
    # Delegate attributes to the base datamodule
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name in ("base", "_parameters", "_buffers", "_modules"):
            raise AttributeError(name)
        return getattr(self.base, name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        self.base.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        self.base.setup(stage)

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self):
        self.base.setup()
        combined: dict[str, Any] = {
            "train_samples": CombinedLoader(
                dict(self.base.train_dataloaders), mode="min_size"
            ),
            "metric_samples": CombinedLoader(
                self._metric_loaders_dict(), mode="min_size"
            ),
        }
        return CombinedLoader(combined, mode="max_size_cycle")

    def val_dataloader(self):
        self.base.setup()
        if not self.base.val_dataloaders or not self._has_val_data():
            return []
        combined: dict[str, Any] = {
            "val_samples": CombinedLoader(
                dict(self.base.val_dataloaders), mode="min_size"
            ),
            "metric_samples": CombinedLoader(
                self._metric_loaders_dict(), mode="min_size"
            ),
        }
        return CombinedLoader(combined, mode="max_size_cycle")

    def test_dataloader(self):
        self.base.setup()
        return self.base.test_dataloader()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _metric_loaders_dict(self) -> dict[str, DataLoader]:
        """Convert the list of metric-sample loaders to a sorted dict."""
        labels = sorted(self.base.selected_train_frames.keys())
        loaders = self.base.metric_samples_dataloaders
        return {
            self.base._timepoint_name(label): loader
            for label, loader in zip(labels, loaders)
        }

    def _has_val_data(self) -> bool:
        """Check whether any validation data actually exists."""
        if not self.base.val_frames:
            return False
        return any(f.shape[0] > 0 for f in self.base.val_frames.values())
