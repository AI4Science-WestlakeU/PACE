from __future__ import annotations

import re
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse

from src.dataloaders.balanced_timepoint_data import BalancedTimepointDataModule
# @article{liu2020reprogramming,
#   title={Reprogramming roadmap reveals route to human induced trophoblast stem cells},
#   author={Liu, Xiaodong and Ouyang, John F and Rossello, Fernando J and Tan, Jia Ping and Davidson, Kathryn C and Valdes, Daniela S and Schroeder, Jan and Sun, Yu BY and Chen, Joseph and Knaupp, Anja S and others},
#   journal={Nature},
#   volume={586},
#   number={7827},
#   pages={101--107},
#   year={2020},
#   publisher={Nature Publishing Group UK London}
# }

class IPSCDataModule(BalancedTimepointDataModule):
    """Lightning datamodule for iPSC Liu PCA AnnData files.

    Reads PCA features from ``obsm["X_pca"]`` and timepoint labels from
    ``obs["timepoint"]``.  Timepoint strings like ``"D24"`` are converted
    to integers (``24``) so that temporal sorting is correct.
    """

    def __init__(self, args: Any):
        self.time_column = str(getattr(args, "time_column", "timepoint"))
        self.feature_key = str(getattr(args, "feature_key", "X_pca"))
        super().__init__(args)

    @staticmethod
    def _parse_timepoint(label: str) -> int:
        """Convert ``'D24'`` → ``24``.  Plain integers pass through."""
        if isinstance(label, (int, float, np.integer, np.floating)):
            return int(label)
        m = re.match(r"[Dd]?(\d+)", str(label).strip())
        if m:
            return int(m.group(1))
        raise ValueError(f"Cannot parse timepoint label: {label!r}")

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

        raw_labels = adata.obs[self.time_column].to_numpy()
        int_labels = np.array([self._parse_timepoint(l) for l in raw_labels])

        frames: dict[int, np.ndarray] = {}
        for t in sorted(np.unique(int_labels)):
            mask = int_labels == t
            frames[int(t)] = features[mask]
        return frames
