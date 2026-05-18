#!/usr/bin/env python
"""Evaluate learned holdout velocities for PACE on EB PHATE.

Supported dataset: eb_phate (data/eb_velocity_v5.npz).
Supported methods: pace.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloaders.eb_data import EBDataModule
from src.methods.pace.stage2.flow_matcher import labels_to_timesteps
from src.methods.pace.stage2.networks import VelocityNet


SUPPORTED_VELOCITY_FIELD_METHODS = {
    "pace",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute normalized L2/cosine holdout velocity metrics."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "Method run directories containing resolved_config.yaml, or result roots "
            "with method subdirectories."
        ),
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated method names to evaluate when a result root is passed.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. Defaults to <result-root>/holdout_velocity_metrics.csv "
            "for roots, or <method-dir>/metrics/holdout_velocity_metrics.csv for one run."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--include-raw-l2",
        action="store_true",
        help="Also report unnormalized mean L2 in the model/data coordinate system.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Only valid when evaluating one method run dir.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def as_namespace(data: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def method_name_from_run(run_dir: Path, cfg: dict[str, Any]) -> str:
    method = cfg.get("method")
    if isinstance(method, dict) and method.get("name"):
        return str(method["name"])
    return run_dir.name


def discover_run_dirs(paths: list[str], methods: set[str] | None) -> list[Path]:
    run_dirs: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if (path / "resolved_config.yaml").exists():
            run_dirs.append(path)
            continue

        for child in sorted(path.iterdir() if path.exists() else []):
            if not child.is_dir() or not (child / "resolved_config.yaml").exists():
                continue
            if methods is not None and child.name not in methods:
                continue
            run_dirs.append(child)
    return run_dirs


def resolve_data_path(cfg: dict[str, Any]) -> Path:
    data_path = Path(str(cfg["data_path"]))
    if data_path.is_absolute():
        return data_path
    return REPO_ROOT / data_path


def make_datamodule(cfg: dict[str, Any]):
    data_name = str(cfg.get("data_name", ""))
    args = as_namespace({**cfg, "data_path": str(resolve_data_path(cfg))})
    if data_name == "eb_phate":
        return EBDataModule(args)
    raise ValueError(f"Unsupported data_name={data_name!r}. Supported: eb_phate.")


def sorted_labels(labels: list[Any]) -> list[Any]:
    return sorted(labels, key=lambda value: float(value))


def subsample_indices_for_labels(
    frame_sizes: dict[Any, int],
    labels: list[Any],
    requested_count: int | None,
    *,
    seed: int,
    preserve_frame_order: bool,
    allow_replacement: bool,
    equalize: bool,
) -> dict[Any, np.ndarray]:
    if requested_count is None and equalize and frame_sizes:
        requested_count = min(frame_sizes.values())

    result: dict[Any, np.ndarray] = {}
    rng = np.random.default_rng(seed)
    for label in labels:
        n = int(frame_sizes[label])
        if requested_count is None:
            result[label] = np.arange(n)
            continue
        if n < requested_count and not allow_replacement:
            raise ValueError(
                f"Timepoint {label!r} has {n} samples, but {requested_count} requested."
            )
        if n == requested_count and not allow_replacement:
            indices = np.arange(n)
        else:
            indices = rng.choice(n, size=requested_count, replace=n < requested_count)
            if preserve_frame_order:
                indices = np.sort(indices)
        result[label] = indices
    return result


def load_reference_velocity_frames(
    cfg: dict[str, Any],
    dm,
) -> dict[Any, torch.Tensor]:
    data_name = str(cfg.get("data_name", ""))
    data_path = resolve_data_path(cfg)
    data = np.load(data_path, allow_pickle=True)
    test_labels = sorted_labels(list(dm.unique_test_labels))
    requested = cfg.get("test_samples_per_timepoint", cfg.get("samples_per_timepoint"))
    requested_count = None if requested is None else int(requested)
    seed = int(cfg.get("seed", 42)) + 1
    preserve = bool(cfg.get("preserve_frame_order", False))
    allow_replacement = bool(cfg.get("allow_replacement", False))
    equalize = bool(cfg.get("equalize_timepoint_counts", False))

    if data_name == "eb_phate":
        labels = np.asarray(data["sample_labels"])
        velocities = np.asarray(data["delta_embedding"], dtype=np.float32)
        frame_sizes = {
            label: int(np.sum(labels == label))
            for label in test_labels
        }
        indices_by_label = subsample_indices_for_labels(
            frame_sizes,
            test_labels,
            requested_count,
            seed=seed,
            preserve_frame_order=preserve,
            allow_replacement=allow_replacement,
            equalize=equalize,
        )
        out = {}
        for label in test_labels:
            frame = velocities[labels == label, : int(cfg["dim"])]
            out[label] = frame[indices_by_label[label]]
    else:
        raise ValueError(f"Unsupported data_name={data_name!r}")

    if getattr(dm, "scaler", None) is not None:
        scale = dm.scaler.scale_.astype(np.float32)[: int(cfg["dim"])]
        out = {label: frame / scale for label, frame in out.items()}

    return {label: torch.tensor(frame, dtype=torch.float32) for label, frame in out.items()}


def infer_checkpoint(run_dir: Path, method: str, explicit: str | None = None) -> Path:
    if explicit is not None:
        ckpt = Path(explicit)
        return ckpt if ckpt.is_absolute() else REPO_ROOT / ckpt

    candidates = list((run_dir / "checkpoints" / "stage2_flow").glob("*.ckpt"))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for {method} under {run_dir}")

    for ckpt in candidates:
        if ckpt.name == "last.ckpt":
            return ckpt

    def sort_key(path: Path) -> tuple[int, int, float]:
        match = re.search(r"epoch=(\d+)", path.name)
        epoch = int(match.group(1)) if match else -1
        version_match = re.search(r"-v(\d+)\.ckpt$", path.name)
        version = int(version_match.group(1)) if version_match else 0
        return epoch, version, path.stat().st_mtime

    return max(candidates, key=sort_key)


def remap_state_dict(raw_state: dict[str, torch.Tensor], target_keys: set[str]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for target_key in target_keys:
        if target_key in raw_state:
            remapped[target_key] = raw_state[target_key]
            continue
        suffix = "." + target_key
        matches = [key for key in raw_state if key.endswith(suffix)]
        if matches:
            remapped[target_key] = raw_state[sorted(matches, key=len)[0]]
    return remapped


def build_velocity_model(cfg: dict[str, Any], method: str, ckpt_path: Path, device: str):
    dim = int(cfg["dim"])
    hidden_dims = list(cfg.get("hidden_dims_flow", [64, 64, 64]))
    activation = str(cfg.get("activation_flow", "selu"))

    model = VelocityNet(
        dim=dim,
        hidden_dims=hidden_dims,
        activation=activation,
        batch_norm=False,
    )
    model_for_state = model

    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state = ckpt.get("state_dict", ckpt)
    state = remap_state_dict(raw_state, set(model_for_state.state_dict().keys()))
    missing, unexpected = model_for_state.load_state_dict(state, strict=False)
    missing_non_buffers = [key for key in missing if not key.endswith("num_batches_tracked")]
    if missing_non_buffers:
        raise RuntimeError(
            f"Could not load {ckpt_path}; missing keys include {missing_non_buffers[:5]}"
        )
    if unexpected:
        print(f"Warning: ignored unexpected checkpoint keys: {unexpected[:5]}", file=sys.stderr)

    model.to(device)
    model.eval()
    return model


def eval_time_for_label(label: Any, train_labels: list[Any]) -> float:
    ordered = sorted_labels(train_labels)
    lo, hi = float(ordered[0]), float(ordered[-1])
    if hi == lo:
        return 0.0

    train_times = labels_to_timesteps(ordered)
    train_label_to_t = {float(label): train_times[i] for i, label in enumerate(ordered)}
    label_f = float(label)
    if label_f in train_label_to_t:
        return float(train_label_to_t[label_f])
    return float((label_f - lo) / (hi - lo))


def compute_metrics(
    model: torch.nn.Module,
    frames: dict[Any, torch.Tensor],
    velocity_frames: dict[Any, torch.Tensor],
    eval_labels: list[Any],
    train_labels: list[Any],
    *,
    batch_size: int,
    device: str,
    include_raw_l2: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for label in eval_labels:
        if label not in frames or label not in velocity_frames:
            continue

        x = frames[label].float()
        v_true = velocity_frames[label].float()
        n = min(x.shape[0], v_true.shape[0])
        x = x[:n]
        v_true = v_true[:n]
        t_eval = eval_time_for_label(label, train_labels)

        pred_chunks = []
        for start in range(0, n, batch_size):
            xb = x[start:start + batch_size].to(device)
            tb = torch.full((xb.shape[0],), t_eval, device=device)
            with torch.no_grad():
                pred_chunks.append(model(tb, xb).detach().cpu())
        v_pred = torch.cat(pred_chunks, dim=0)

        v_pred_unit = F.normalize(v_pred, p=2, dim=1, eps=1e-12)
        v_true_unit = F.normalize(v_true, p=2, dim=1, eps=1e-12)
        cos = (v_pred_unit * v_true_unit).sum(dim=1)
        l2_unit = torch.norm(v_pred_unit - v_true_unit, dim=1)

        row: dict[str, Any] = {
            "test_label": label,
            "n": int(n),
            "t_eval": t_eval,
            "cos": float(cos.mean().item()),
            "cos_dist": float((1.0 - cos).mean().item()),
            "l2_unit": float(l2_unit.mean().item()),
            "pred_norm_mean": float(torch.norm(v_pred, dim=1).mean().item()),
            "true_norm_mean": float(torch.norm(v_true, dim=1).mean().item()),
        }
        if include_raw_l2:
            row["l2_raw"] = float(torch.norm(v_pred - v_true, dim=1).mean().item())
        rows.append(row)

    return rows


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg_path = run_dir / "resolved_config.yaml"
    cfg = load_yaml(cfg_path)
    method = method_name_from_run(run_dir, cfg)
    if method not in SUPPORTED_VELOCITY_FIELD_METHODS:
        print(f"Skipping {run_dir}: method {method!r} has no direct v(t, x) evaluator.", file=sys.stderr)
        return []

    dm = make_datamodule(cfg)
    dm.setup()
    velocity_frames = load_reference_velocity_frames(cfg, dm)
    ckpt_path = infer_checkpoint(run_dir, method, explicit=args.checkpoint)
    model = build_velocity_model(cfg, method, ckpt_path, args.device)

    rows = compute_metrics(
        model=model,
        frames=dm.test_frames,
        velocity_frames=velocity_frames,
        eval_labels=sorted_labels(list(dm.unique_test_labels)),
        train_labels=sorted_labels(list(dm.unique_train_labels)),
        batch_size=args.batch_size,
        device=args.device,
        include_raw_l2=args.include_raw_l2,
    )

    for row in rows:
        row.update(
            {
                "result_root": str(run_dir.parent.relative_to(REPO_ROOT)),
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "method": method,
                "data_name": cfg.get("data_name"),
                "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def default_output_path(run_dirs: list[Path], explicit: str | None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_absolute() else REPO_ROOT / path
    if len(run_dirs) == 1:
        return run_dirs[0] / "metrics" / "holdout_velocity_metrics.csv"
    parents = {run_dir.parent for run_dir in run_dirs}
    if len(parents) == 1:
        return parents.pop() / "holdout_velocity_metrics.csv"
    return REPO_ROOT / "results" / "holdout_velocity_metrics.csv"


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "result_root",
        "run_dir",
        "method",
        "data_name",
        "test_label",
        "n",
        "t_eval",
        "cos",
        "cos_dist",
        "l2_unit",
        "pred_norm_mean",
        "true_norm_mean",
        "checkpoint",
    ]
    extra_fields = sorted({key for row in rows for key in row} - set(base_fields))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    requested_methods = None
    if args.methods:
        requested_methods = {part.strip() for part in args.methods.split(",") if part.strip()}

    run_dirs = discover_run_dirs(args.paths, requested_methods)
    if args.checkpoint is not None and len(run_dirs) != 1:
        raise ValueError("--checkpoint can only be used with exactly one method run directory.")
    if not run_dirs:
        raise FileNotFoundError("No run directories with resolved_config.yaml were found.")

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.extend(evaluate_run(run_dir, args))

    if not rows:
        raise RuntimeError("No velocity metrics were computed.")

    output_path = default_output_path(run_dirs, args.output)
    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
