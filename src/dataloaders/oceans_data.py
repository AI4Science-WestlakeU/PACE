from __future__ import annotations

from typing import Any

import numpy as np

from src.dataloaders.balanced_timepoint_data import BalancedTimepointDataModule


class OceansDataModule(BalancedTimepointDataModule):
    """Lightning datamodule for frame-wise oceans data.

    The loader expects an `.npz` file with at least:
    - `positions`: shape `(time, points, dim)`

    Optional label arrays supported:
    - `timepoints`
    - `times`
    - `frame_labels`
    - `labels`
    """

    def _load_timepoint_frames(self) -> dict[Any, np.ndarray]:
        data = np.load(self.data_path, allow_pickle=True)
        if "positions" not in data:
            available = ", ".join(sorted(data.files))
            raise ValueError(
                "OceansDataModule expects an npz file with `positions`. "
                f"Available keys: {available}."
            )

        positions = np.asarray(data["positions"], dtype=np.float32)
        if positions.ndim != 3:
            raise ValueError(
                f"`positions` must have shape (time, points, dim), got {positions.shape}."
            )

        n_time = positions.shape[0]
        time_labels = self._extract_time_labels(data, n_time)
        return {
            time_labels[index]: positions[index]
            for index in range(n_time)
        }

    @staticmethod
    def _extract_time_labels(data: Any, n_time: int) -> np.ndarray:
        for key in ("timepoints", "times", "frame_labels", "labels"):
            if key not in data:
                continue
            candidate = np.asarray(data[key])
            if candidate.ndim == 1 and candidate.shape[0] == n_time:
                return candidate
        return np.arange(n_time)