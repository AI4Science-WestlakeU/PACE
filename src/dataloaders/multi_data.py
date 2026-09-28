from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse

from src.dataloaders.balanced_timepoint_data import BalancedTimepointDataModule


class MultiDataModule(BalancedTimepointDataModule):
    """Lightning datamodule for Open Problems Multiome AnnData files.

    Default settings read:
    - per-cell PC scores from ``obsm["X_pca"]`` (fallback: ``X @ varm["PCs"]``)
    - time labels from ``obs["day"]``
    """

    def __init__(self, args: Any):
        self.time_column = str(getattr(args, "time_column", "day"))
        self.feature_key = str(getattr(args, "feature_key", "X_pca"))
        super().__init__(args)

    def _load_timepoint_frames(self) -> dict[Any, np.ndarray]:
        adata = ad.read_h5ad(self.data_path)

        if self.time_column not in adata.obs.columns:
            raise ValueError(
                f"Missing time column {self.time_column!r} in AnnData obs.",
            )

        # Extract per-cell PC scores
        if self.feature_key in adata.obsm:
            features = np.asarray(adata.obsm[self.feature_key], dtype=np.float32)
        elif "PCs" in adata.varm:
            X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
            features = (X @ adata.varm["PCs"]).astype(np.float32)
        else:
            raise ValueError(
                f"Cannot find feature_key {self.feature_key!r} in obsm "
                f"or 'PCs' in varm for fallback projection.",
            )

        labels = adata.obs[self.time_column].to_numpy()

        frames: dict[Any, np.ndarray] = {}
        label_keys = np.asarray(
            [self._label_key(label) for label in labels],
            dtype=object,
        )
        for label in self._sorted_labels(np.unique(labels)):
            mask = label_keys == self._label_key(label)
            frames[label] = features[mask]
        return frames
