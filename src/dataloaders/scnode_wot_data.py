from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse

from src.dataloaders.balanced_timepoint_data import BalancedTimepointDataModule


class ScNodeWOTDataModule(BalancedTimepointDataModule):
    """Lightning datamodule for scNODE Schiebinger2019/WOT AnnData files.

    Default settings read:
    - vectors from AnnData ``.X``; for WOT tutorial files these are FLE coordinates
    - time labels from ``obs["day"]``
    """

    def __init__(self, args: Any):
        self.time_column = str(getattr(args, "time_column", "day"))
        super().__init__(args)

    def _load_timepoint_frames(self) -> dict[Any, np.ndarray]:
        adata = ad.read_h5ad(self.data_path)
        if self.time_column not in adata.obs.columns:
            raise ValueError(
                f"Missing time column {self.time_column!r} in AnnData obs.",
            )

        if sparse.issparse(adata.X):
            expression = adata.X.toarray()
        else:
            expression = np.asarray(adata.X)
        expression = expression.astype(np.float32, copy=False)
        labels = adata.obs[self.time_column].to_numpy()

        frames: dict[Any, np.ndarray] = {}
        label_keys = np.asarray(
            [self._label_key(label) for label in labels],
            dtype=object,
        )
        for label in self._sorted_labels(np.unique(labels)):
            mask = label_keys == self._label_key(label)
            frames[label] = expression[mask]
        return frames
