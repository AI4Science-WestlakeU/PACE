"""DataModule for the ControlledToggle2D synthetic benchmark.

Loads the dense ``positions`` array produced by
``src/data_preprocess/generate_toggle_switch.py`` and exposes ground-truth
particle IDs, basin labels, switch times, velocities and fixed points for
downstream evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.dataloaders.balanced_timepoint_data import BalancedTimepointDataModule


class ToggleSwitchDataModule(BalancedTimepointDataModule):
    """DataModule for the toggle-switch transverse-transition benchmark.

    Expected NPZ keys:
    - ``positions``            : (T, N, 2) dense array
    - ``timepoints``           : (T,) integer labels
    - ``times``                : (T,) continuous times (optional)
    - ``particle_ids``         : (N,) persistent IDs (optional)
    - ``basin_labels``         : (N,) terminal basin labels (optional)
    - ``basin_labels_by_time`` : (T, N) basin labels A/B/transit (optional)
    - ``switch_times``         : (N,) first separatrix crossing times (optional)
    - ``init_basin``           : (N,) initial basin labels (optional)
    - ``velocity``             : (T, N, 2) true drift velocities (optional)
    - ``fixed_points``         : (T, 3) object array of (A, B, saddle) (optional)
    """

    def _load_timepoint_frames(self) -> dict[Any, np.ndarray]:
        data = np.load(self.data_path, allow_pickle=True)
        if "positions" not in data:
            raise ValueError(
                f"ToggleSwitchDataModule expects 'positions' in {self.data_path}. "
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
        self.particle_ids = data["particle_ids"] if "particle_ids" in data else None
        self.basin_labels = data["basin_labels"] if "basin_labels" in data else None
        self.basin_labels_by_time = (
            data["basin_labels_by_time"] if "basin_labels_by_time" in data else None
        )
        self.switch_times = data["switch_times"] if "switch_times" in data else None
        self.init_basin = data["init_basin"] if "init_basin" in data else None
        self.true_velocity = data["velocity"] if "velocity" in data else None
        self.fixed_points = data["fixed_points"] if "fixed_points" in data else None

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
