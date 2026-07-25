"""DataModule for the CurvedPitchfork synthetic benchmark.

Loads the dense ``positions`` array produced by
``src/data_preprocess/generate_curved_pitchfork.py`` and exposes the ground-truth
particle IDs, branch labels, velocities, and tangent/normal vectors as attributes
for downstream evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.dataloaders.balanced_timepoint_data import BalancedTimepointDataModule


class CurvedPitchforkDataModule(BalancedTimepointDataModule):
    """DataModule for the curved pitchfork synthetic benchmark.

    Expected NPZ keys:
    - ``positions``        : (T, N, 2) dense array
    - ``timepoints``       : (T,) integer labels
    - ``times``            : (T,) continuous times (optional)
    - ``particle_ids``     : (N,) persistent IDs (optional)
    - ``branch_labels``    : (T, N) branch labels (optional)
    - ``velocity``         : (T, N, 2) true velocities (optional)
    - ``tangent_unit``     : (T, N, 2, 2) true tangents (optional)
    - ``normal_unit``      : (T, N, 2, 2) true normals (optional)
    """

    def __init__(self, args: Any):
        self.particle_id_key = str(getattr(args, "particle_id_key", "particle_ids"))
        self.branch_label_key = str(getattr(args, "branch_label_key", "branch_labels"))
        self.velocity_key = str(getattr(args, "velocity_key", "velocity"))
        self.tangent_key = str(getattr(args, "tangent_key", "tangent_unit"))
        self.normal_key = str(getattr(args, "normal_key", "normal_unit"))
        super().__init__(args)

    def _load_timepoint_frames(self) -> dict[Any, np.ndarray]:
        data = np.load(self.data_path, allow_pickle=True)
        if "positions" not in data:
            raise ValueError(
                f"CurvedPitchforkDataModule expects 'positions' in {self.data_path}. "
                f"Available: {list(data.files)}"
            )

        positions = np.asarray(data["positions"], dtype=np.float32)
        if positions.ndim != 3:
            raise ValueError(f"positions must be dense (T, N, 2), got shape {positions.shape}")

        n_time = positions.shape[0]
        time_labels = self._extract_time_labels(data, n_time)

        frames = {time_labels[t]: positions[t] for t in range(n_time)}

        # Store GT arrays as attributes for eval scripts.
        self.times = np.asarray(data["times"], dtype=np.float32) if "times" in data else None
        self.particle_ids = self._load_optional(data, self.particle_id_key)
        self.branch_labels = self._load_optional(data, self.branch_label_key)
        self.true_velocity = self._load_optional(data, self.velocity_key)
        self.true_tangent = self._load_optional(data, self.tangent_key)
        self.true_normal = self._load_optional(data, self.normal_key)

        return frames

    @staticmethod
    def _extract_time_labels(data: Any, n_time: int) -> np.ndarray:
        for key in ("timepoints", "times", "frame_labels", "labels"):
            if key not in data:
                continue
            candidate = np.asarray(data[key])
            if candidate.ndim == 1 and candidate.shape[0] == n_time:
                return candidate
        return np.arange(n_time)

    @staticmethod
    def _load_optional(data: Any, key: str) -> np.ndarray | None:
        if key not in data or data[key] is None:
            return None
        return data[key]
