from __future__ import annotations

from typing import Any

import numpy as np

from src.dataloaders.balanced_timepoint_data import BalancedTimepointDataModule


class ToyBifurcationDataModule(BalancedTimepointDataModule):
    """Lightning datamodule for the synthetic two-branch toy point cloud.

    The loader expects an `.npz` file with `positions`, either as:
    - a dense array with shape `(time, points, dim)`, or
    - an object array of length `time`, each item shaped `(points_t, dim)`.

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
                "ToyBifurcationDataModule expects an npz file with `positions`. "
                f"Available keys: {available}."
            )

        frames = self._extract_frames(data["positions"])
        time_labels = self._extract_time_labels(data, len(frames))
        return {time_labels[index]: frames[index] for index in range(len(frames))}

    @staticmethod
    def _extract_frames(raw_positions: Any) -> list[np.ndarray]:
        positions = np.asarray(raw_positions, dtype=object)

        if positions.ndim == 1 and positions.dtype == object:
            frames: list[np.ndarray] = []
            for index, frame in enumerate(positions):
                arr = np.asarray(frame, dtype=np.float32)
                if arr.ndim != 2:
                    raise ValueError(
                        f"Toy bifurcation frame {index} must be 2D, got shape {arr.shape}."
                    )
                frames.append(arr)
            return frames

        dense_positions = np.asarray(raw_positions, dtype=np.float32)
        if dense_positions.ndim != 3:
            raise ValueError(
                "`positions` must be either an object array of 2D frames or "
                f"a dense array with shape (time, points, dim), got {dense_positions.shape}."
            )
        return [dense_positions[index] for index in range(dense_positions.shape[0])]

    @staticmethod
    def _extract_time_labels(data: Any, n_time: int) -> np.ndarray:
        for key in ("timepoints", "times", "frame_labels", "labels"):
            if key not in data:
                continue
            candidate = np.asarray(data[key])
            if candidate.ndim == 1 and candidate.shape[0] == n_time:
                return candidate
        return np.arange(n_time)
