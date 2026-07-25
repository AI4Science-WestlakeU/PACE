from __future__ import annotations

from typing import Any

import numpy as np

from src.dataloaders.balanced_timepoint_data import BalancedTimepointDataModule


class LARRYDataModule(BalancedTimepointDataModule):
    """Lightning datamodule for preprocessed LARRY hematopoiesis data.

    The ``.npz`` file is expected to contain at least:
    - ``positions``     — 2-D embedding, shape ``(N, 2)``
    - ``timepoints``    — time labels, shape ``(N,)`` (strings or numeric)
    - ``clone_ids``     — clone barcode ids, shape ``(N,)`` (optional)
    - ``fate_labels``   — fate labels, shape ``(N,)`` (optional)
    - ``lineage_labels`` — lineage labels, shape ``(N,)`` (optional)

    String time labels (e.g. ``"day2"``, ``"day4"``, ``"day6"``) are automatically
    remapped to sorted numeric indices ``0, 1, 2`` so that downstream PACE code,
    which uses labels as numeric time values, works without modification.
    The original label mapping is stored in ``timepoint_label_map``.
    """

    def __init__(self, args: Any):
        self.position_key: str = str(getattr(args, "position_key", "positions"))
        self.timepoint_key: str = str(getattr(args, "timepoint_key", "timepoints"))
        self.clone_id_key: str | None = getattr(args, "clone_id_key", "clone_ids")
        self.fate_label_key: str | None = getattr(args, "fate_label_key", "fate_labels")
        self.lineage_label_key: str | None = getattr(args, "lineage_label_key", "lineage_labels")
        self.timepoint_label_map: dict[Any, int] = {}
        super().__init__(args)

    def _load_timepoint_frames(self) -> dict[Any, np.ndarray]:
        data = np.load(self.data_path, allow_pickle=True)

        if self.position_key not in data:
            raise ValueError(
                f"Expected key {self.position_key!r} in {self.data_path}. "
                f"Available keys: {list(data.keys())}"
            )
        if self.timepoint_key not in data:
            raise ValueError(
                f"Expected key {self.timepoint_key!r} in {self.data_path}. "
                f"Available keys: {list(data.keys())}"
            )

        coordinates = data[self.position_key].astype(np.float32)
        raw_labels = data[self.timepoint_key]

        # Remap string time labels (e.g. 'day2') to sorted numeric indices so that
        # downstream PACE code, which treats labels as numeric time values, works
        # without modification.  Keep numeric labels as-is.
        unique_labels = self._sorted_labels(np.unique(raw_labels))
        if all(isinstance(l, (int, float, np.integer, np.floating)) for l in unique_labels):
            self.timepoint_label_map = {label: label for label in unique_labels}
            numeric_labels = np.asarray(raw_labels)
        else:
            self.timepoint_label_map = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = np.array([self.timepoint_label_map[label] for label in raw_labels])

        # Keep metadata for downstream evaluation (barcode recovery, fate).
        self.clone_ids = self._load_optional_array(data, self.clone_id_key)
        self.fate_labels = self._load_optional_array(data, self.fate_label_key)
        self.lineage_labels = self._load_optional_array(data, self.lineage_label_key)

        frames: dict[Any, np.ndarray] = {}
        for label in sorted(self.timepoint_label_map.values()):
            mask = numeric_labels == label
            frames[label] = coordinates[mask]
        return frames

    def _load_optional_array(self, data: np.lib.npyio.NpzFile, key: str | None) -> np.ndarray | None:
        if key is None or key == "" or key not in data:
            return None
        return data[key].copy()
