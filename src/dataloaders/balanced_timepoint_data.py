from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

try:
    # pytorch_lightning<=1.x
    from pytorch_lightning.trainer.supporters import CombinedLoader
except Exception:  # pragma: no cover
    try:
        # lightning>=2.x
        from lightning.pytorch.utilities.combined_loader import CombinedLoader
    except Exception:  # pragma: no cover
        from pytorch_lightning.utilities.combined_loader import CombinedLoader


class BalancedTimepointDataModule(pl.LightningDataModule, ABC):
    """Base datamodule for temporal point clouds with timepoint-level sampling.

    Expected args fields:
    - data_path
    - batch_size
    - dim
    - whiten
    - split_ratios
    - train_timepoint_labels / test_timepoint_labels (optional)
    - samples_per_timepoint or train_samples_per_timepoint / test_samples_per_timepoint
    """

    def __init__(self, args: Any):
        super().__init__()
        self.args = args

        self.data_path = Path(getattr(args, "data_path"))
        self.batch_size = int(getattr(args, "batch_size", 128))
        self.dim = int(getattr(args, "dim", 2))
        self.whiten = bool(getattr(args, "whiten", False))
        self.seed = int(getattr(args, "seed", 42))
        self.split_ratios = list(getattr(args, "split_ratios", [0.9, 0.1]))
        self.train_timepoint_labels = getattr(args, "train_timepoint_labels", None)
        self.test_timepoint_labels = getattr(args, "test_timepoint_labels", None)

        shared_samples = getattr(args, "samples_per_timepoint", None)
        self.train_samples_per_timepoint = self._normalize_sample_count(
            getattr(args, "train_samples_per_timepoint", shared_samples)
        )
        self.test_samples_per_timepoint = self._normalize_sample_count(
            getattr(args, "test_samples_per_timepoint", shared_samples)
        )

        self.equalize_timepoint_counts = bool(
            getattr(args, "equalize_timepoint_counts", False)
        )
        self.allow_replacement = bool(getattr(args, "allow_replacement", False))
        self.preserve_frame_order = bool(getattr(args, "preserve_frame_order", False))
        self.full_frame_batches = bool(getattr(args, "full_frame_batches", False))
        self.test_full_frame_batches = bool(
            getattr(args, "test_full_frame_batches", True)
        )
        self.drop_last = bool(getattr(args, "drop_last", False))
        self.num_workers = int(getattr(args, "num_workers", 0))
        self.pin_memory = bool(getattr(args, "pin_memory", False))

        train_ratio = float(self.split_ratios[0])
        if not 0.0 < train_ratio <= 1.0:
            raise ValueError(
                f"split_ratios[0] must be between 0 and 1 (inclusive), got {train_ratio}."
            )
        self.train_ratio = train_ratio

        self.scaler: StandardScaler | None = None
        self.available_timepoint_labels: list[Any] = []
        self.unique_train_labels: list[Any] = []
        self.unique_test_labels: list[Any] = []
        self.train_frame_sizes: dict[Any, int] = {}
        self.val_frame_sizes: dict[Any, int] = {}
        self.test_frame_sizes: dict[Any, int] = {}

        self.selected_train_frames: dict[Any, torch.Tensor] = {}
        self.train_frames: dict[Any, torch.Tensor] = {}
        self.val_frames: dict[Any, torch.Tensor] = {}
        self.test_frames: dict[Any, torch.Tensor] = {}

        # Original indices of selected/split frames, useful for mapping back to
        # per-cell metadata such as lineage barcodes.
        self.selected_train_indices: dict[Any, np.ndarray] = {}
        self.train_indices: dict[Any, np.ndarray] = {}
        self.val_indices: dict[Any, np.ndarray] = {}
        self.test_indices: dict[Any, np.ndarray] = {}

        self.train_dataloaders: dict[str, DataLoader] = {}
        self.val_dataloaders: dict[str, DataLoader] = {}
        self.test_dataloaders: dict[str, DataLoader] = {}
        self.metric_samples_dataloaders: list[DataLoader] = []
        self._is_setup = False

    def prepare_data(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

    def setup(self, stage: str | None = None) -> None:
        if self._is_setup:
            return

        raw_frames = self._standardize_frames(self._load_timepoint_frames())
        self.available_timepoint_labels = [self._to_python_scalar(v) for v in raw_frames.keys()]

        train_labels = self._resolve_labels(
            available_labels=list(raw_frames.keys()),
            requested=self.train_timepoint_labels,
            split_name="train",
        )
        test_labels = self._resolve_labels(
            available_labels=list(raw_frames.keys()),
            requested=self.test_timepoint_labels,
            split_name="test",
        )

        selected_train = {label: raw_frames[label].copy() for label in train_labels}
        selected_test = {label: raw_frames[label].copy() for label in test_labels}

        balanced_train, self.selected_train_indices = self._subsample_frames(
            frames=selected_train,
            requested_count=self.train_samples_per_timepoint,
            split_name="train",
            seed=self.seed,
        )
        balanced_test, self.test_indices = self._subsample_frames(
            frames=selected_test,
            requested_count=self.test_samples_per_timepoint,
            split_name="test",
            seed=self.seed + 1,
        )

        split_train, split_val, self.train_indices, self.val_indices = self._split_train_val_frames(
            balanced_train, self.selected_train_indices
        )

        if self.whiten:
            fit_data = np.concatenate(list(split_train.values()), axis=0)
            self.scaler = StandardScaler()
            self.scaler.fit(fit_data)

            balanced_train = self._transform_frames(balanced_train)
            split_train = self._transform_frames(split_train)
            split_val = self._transform_frames(split_val)
            balanced_test = self._transform_frames(balanced_test)

        self.selected_train_frames = self._frames_to_tensors(balanced_train)
        self.train_frames = self._frames_to_tensors(split_train)
        self.val_frames = self._frames_to_tensors(split_val)
        self.test_frames = self._frames_to_tensors(balanced_test)

        self.unique_train_labels = [self._to_python_scalar(v) for v in train_labels]
        self.unique_test_labels = [self._to_python_scalar(v) for v in test_labels]
        self.num_timesteps = len(self.unique_train_labels)

        self.train_frame_sizes = {
            self._to_python_scalar(label): int(frame.shape[0])
            for label, frame in self.train_frames.items()
        }
        self.val_frame_sizes = {
            self._to_python_scalar(label): int(frame.shape[0])
            for label, frame in self.val_frames.items()
        }
        self.test_frame_sizes = {
            self._to_python_scalar(label): int(frame.shape[0])
            for label, frame in self.test_frames.items()
        }

        self.train_dataloaders = self._build_loader_map(
            self.train_frames,
            shuffle=not self.preserve_frame_order,
            drop_last=self.drop_last,
            full_frame_batches=self.full_frame_batches,
        )
        self.val_dataloaders = self._build_loader_map(
            self.val_frames,
            shuffle=False,
            drop_last=False,
            full_frame_batches=self.full_frame_batches,
        )
        self.test_dataloaders = self._build_loader_map(
            self.test_frames,
            shuffle=False,
            drop_last=False,
            full_frame_batches=self.test_full_frame_batches,
        )
        self.metric_samples_dataloaders = [
            self._build_loader(
                frame,
                shuffle=False,
                drop_last=False,
                full_frame_batch=True,
            )
            for frame in self.selected_train_frames.values()
        ]

        self._is_setup = True

    def train_dataloader(self):
        self.setup()
        return CombinedLoader(self.train_dataloaders, mode="min_size")

    def val_dataloader(self):
        self.setup()
        return CombinedLoader(self.val_dataloaders, mode="min_size")

    def test_dataloader(self):
        self.setup()
        return CombinedLoader(self.test_dataloaders, mode="max_size_cycle")

    @abstractmethod
    def _load_timepoint_frames(self) -> dict[Any, np.ndarray]:
        """Load raw frames keyed by timepoint label."""

    def _standardize_frames(self, frames: dict[Any, np.ndarray]) -> dict[Any, np.ndarray]:
        standardized: dict[Any, np.ndarray] = {}
        for label in self._sorted_labels(frames.keys()):
            frame = np.asarray(frames[label], dtype=np.float32)
            if frame.ndim != 2:
                raise ValueError(
                    f"Timepoint {label!r} must be a 2D array, got shape {frame.shape}."
                )
            if frame.shape[1] < self.dim:
                raise ValueError(
                    f"Timepoint {label!r} has dim {frame.shape[1]}, expected at least {self.dim}."
                )
            standardized[label] = frame[:, : self.dim]
        if not standardized:
            raise ValueError("No timepoint frames were loaded.")
        return standardized

    def _transform_frames(self, frames: dict[Any, np.ndarray]) -> dict[Any, np.ndarray]:
        if self.scaler is None:
            return {label: frame.copy() for label, frame in frames.items()}
        transformed: dict[Any, np.ndarray] = {}
        for label, frame in frames.items():
            if frame.shape[0] == 0:
                transformed[label] = frame.astype(np.float32, copy=True)
            else:
                transformed[label] = self.scaler.transform(frame).astype(np.float32)
        return transformed

    def _split_train_val_frames(
        self,
        frames: dict[Any, np.ndarray],
        indices: dict[Any, np.ndarray] | None = None,
    ) -> tuple[dict[Any, np.ndarray], dict[Any, np.ndarray], dict[Any, np.ndarray], dict[Any, np.ndarray]]:
        train_frames: dict[Any, np.ndarray] = {}
        val_frames: dict[Any, np.ndarray] = {}
        train_indices: dict[Any, np.ndarray] = {}
        val_indices: dict[Any, np.ndarray] = {}
        for label, frame in frames.items():
            idx = indices[label] if indices is not None else np.arange(frame.shape[0])
            if self.train_ratio >= 1.0:
                # No validation split requested
                train_frames[label] = frame.copy()
                val_frames[label] = frame[:0]  # empty array with correct shape
                train_indices[label] = idx.copy()
                val_indices[label] = idx[:0]
                continue
            if frame.shape[0] < 2:
                raise ValueError(
                    f"Timepoint {label!r} has only {frame.shape[0]} sample after subsampling; "
                    "need at least 2 samples to create train/val splits."
                )
            split_index = int(frame.shape[0] * self.train_ratio)
            split_index = min(max(split_index, 1), frame.shape[0] - 1)
            train_frames[label] = frame[:split_index]
            val_frames[label] = frame[split_index:]
            train_indices[label] = idx[:split_index]
            val_indices[label] = idx[split_index:]
        return train_frames, val_frames, train_indices, val_indices

    def _subsample_frames(
        self,
        frames: dict[Any, np.ndarray],
        requested_count: int | None,
        split_name: str,
        seed: int,
    ) -> tuple[dict[Any, np.ndarray], dict[Any, np.ndarray]]:
        if not frames:
            return {}, {}  # no frames to subsample (e.g. no test labels)

        effective_count = requested_count
        if effective_count is None and self.equalize_timepoint_counts:
            effective_count = min(frame.shape[0] for frame in frames.values())

        if effective_count is None:
            indices = {label: np.arange(frame.shape[0]) for label, frame in frames.items()}
            return {label: frame.copy() for label, frame in frames.items()}, indices

        if effective_count <= 0:
            raise ValueError(
                f"{split_name}_samples_per_timepoint must be positive, got {effective_count}."
            )

        rng = np.random.default_rng(seed)
        sampled: dict[Any, np.ndarray] = {}
        sampled_indices: dict[Any, np.ndarray] = {}
        for label, frame in frames.items():
            num_points = frame.shape[0]
            if num_points < effective_count and not self.allow_replacement:
                raise ValueError(
                    f"Timepoint {label!r} has {num_points} samples, but "
                    f"{effective_count} were requested for {split_name}."
                )

            replace = num_points < effective_count
            if not replace and num_points == effective_count:
                sampled[label] = frame.copy()
                sampled_indices[label] = np.arange(num_points)
                continue

            indices = rng.choice(num_points, size=effective_count, replace=replace)
            if self.preserve_frame_order:
                indices = np.sort(indices)
            sampled[label] = frame[indices]
            sampled_indices[label] = indices
        return sampled, sampled_indices

    def _build_loader_map(
        self,
        frames: dict[Any, torch.Tensor],
        shuffle: bool,
        drop_last: bool,
        full_frame_batches: bool,
    ) -> dict[str, DataLoader]:
        return {
            self._timepoint_name(label): self._build_loader(
                frame,
                shuffle=shuffle,
                drop_last=drop_last,
                full_frame_batch=full_frame_batches,
            )
            for label, frame in frames.items()
            if frame.shape[0] > 0  # skip empty frames (e.g. no-validation split)
        }

    def _build_loader(
        self,
        frame: torch.Tensor,
        shuffle: bool,
        drop_last: bool,
        full_frame_batch: bool,
    ) -> DataLoader:
        batch_size = int(frame.shape[0]) if full_frame_batch else self.batch_size
        can_drop_last = bool(drop_last and not full_frame_batch and frame.shape[0] >= batch_size)
        can_shuffle = bool(shuffle and not full_frame_batch and frame.shape[0] > 1)
        return DataLoader(
            frame,
            batch_size=batch_size,
            shuffle=can_shuffle,
            drop_last=can_drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @staticmethod
    def _frames_to_tensors(frames: dict[Any, np.ndarray]) -> dict[Any, torch.Tensor]:
        return {
            label: torch.tensor(frame, dtype=torch.float32)
            for label, frame in frames.items()
        }

    @staticmethod
    def _normalize_sample_count(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return int(value)

    @staticmethod
    def _label_key(value: Any) -> str:
        if isinstance(value, str):
            text = value.strip()
            try:
                return f"{float(text):g}"
            except ValueError:
                return text
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{float(value):g}"
        return str(value)

    @classmethod
    def _sorted_labels(cls, labels) -> list[Any]:
        def sort_key(value: Any):
            key = cls._label_key(value)
            try:
                return (0, float(key))
            except ValueError:
                return (1, key)

        return sorted(labels, key=sort_key)

    @classmethod
    def _resolve_labels(
        cls,
        available_labels: list[Any],
        requested: Any,
        split_name: str,
    ) -> list[Any]:
        if requested is None:
            return list(available_labels)

        requested_list = list(requested)
        if not requested_list:
            return []  # explicitly empty — e.g. train-only datamodule
        key_to_actual = {cls._label_key(label): label for label in available_labels}
        resolved: list[Any] = []
        seen: set[str] = set()

        for label in requested_list:
            key = cls._label_key(label)
            if key not in key_to_actual:
                available = [cls._to_python_scalar(v) for v in available_labels]
                raise ValueError(
                    f"Requested {split_name} timepoint {label!r} not found. "
                    f"Available labels: {available}."
                )
            if key in seen:
                continue
            seen.add(key)
            resolved.append(key_to_actual[key])
        if not resolved:
            raise ValueError(f"No {split_name} timepoints were selected.")
        return resolved

    @classmethod
    def _timepoint_name(cls, label: Any) -> str:
        key = cls._label_key(label)
        safe_key = key.replace("-", "m").replace(".", "p").replace(" ", "_")
        return f"t_{safe_key}"

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        return value.item() if hasattr(value, "item") else value
