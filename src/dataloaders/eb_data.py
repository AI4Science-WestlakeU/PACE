from __future__ import annotations

from typing import Any

import numpy as np

from src.dataloaders.balanced_timepoint_data import BalancedTimepointDataModule


class EBDataModule(BalancedTimepointDataModule):
    """Lightning datamodule for embryoid body (EB) velocity data.

    The ``.npz`` file is expected to contain at least:
    - ``pcs``           — principal components, shape ``(N, D)`` with ``D >= 100``
    - ``phate``         — 2-D PHATE embedding, shape ``(N, 2)``
    - ``sample_labels`` — integer time labels, shape ``(N,)``

    Feature selection rule:
    - ``dim == 2`` → uses ``phate``
    - ``dim > 2``  → uses ``pcs[:, :dim]``

    An explicit ``feature_key`` in args overrides autoselection.
    """

    def __init__(self, args: Any):
        self.feature_key: str | None = getattr(args, "feature_key", None)
        self.label_key: str = str(getattr(args, "label_key", "sample_labels"))
        super().__init__(args)

    def _load_timepoint_frames(self) -> dict[Any, np.ndarray]:
        data = np.load(self.data_path, allow_pickle=True)

        # Resolve which feature matrix to use
        if self.feature_key is not None:
            key = self.feature_key
        elif self.dim == 2:
            key = "phate"
        else:
            key = "pcs"

        if key not in data:
            raise ValueError(
                f"Expected key {key!r} in {self.data_path}. "
                f"Available keys: {list(data.keys())}"
            )

        coordinates = data[key].astype(np.float32)
        labels = data[self.label_key]

        frames: dict[Any, np.ndarray] = {}
        for label in self._sorted_labels(np.unique(labels)):
            mask = labels == label
            frames[label] = coordinates[mask]
        return frames
