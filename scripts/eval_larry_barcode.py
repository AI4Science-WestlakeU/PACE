"""Evaluate PACE Stage 1 coupling against LARRY/Morris clone barcodes and fate labels.

This script now implements the full **ground-truth trajectory evaluation protocol**
described in ``gt_trajectory_evaluation_protocol.md``.  Beyond the original
top-1 barcode/fate accuracy, it reports:

* Matching-level clone / fate consistency (precision/recall/F1, NMI, ARI,
  conditional entropy, transition matrices)
* Soft-neighbor clone / fate recall@K
* Within-clone random baseline

Usage
-----
    python scripts/eval_larry_barcode.py \\
        --results-dir results/larry_pca2_dim2_test1/pace \\
        --data-path data/larry_pca2.npz

The script expects ``stage1_matchings.npz`` inside
``<results-dir>/checkpoints/stage1_psi/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


try:
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        precision_recall_fscore_support,
    )

    _HAS_SKLEARN = True
except Exception as exc:  # pragma: no cover
    log.warning("scikit-learn not available; NMI/ARI/F1 will be skipped: %s", exc)
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PACE Stage-1 matchings against LARRY/Morris barcodes and fates."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="PACE results directory (contains checkpoints/stage1_psi).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the .npz file with positions/timepoints/clone_ids/fate_labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write metrics JSON. Defaults to <results-dir>/metrics.",
    )
    parser.add_argument(
        "--k-list",
        type=int,
        nargs="+",
        default=[1, 5, 10, 50],
        help="K values for Top-K soft-neighbor recall.",
    )
    parser.add_argument(
        "--matching-key",
        type=str,
        default=None,
        help="Key of the matching inside stage1_matchings.npz (auto-detected if omitted).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_larry_data(data_path: str) -> dict[str, np.ndarray]:
    data = np.load(data_path, allow_pickle=True)
    required = {"positions", "timepoints"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Missing required keys in {data_path}: {missing}")

    positions = data["positions"].astype(np.float32)
    timepoints = data["timepoints"]
    clone_ids = data["clone_ids"] if "clone_ids" in data.files else None
    fate_labels = data["fate_labels"] if "fate_labels" in data.files else None
    lineage_labels = data["lineage_labels"] if "lineage_labels" in data.files else None
    return {
        "positions": positions,
        "timepoints": timepoints,
        "clone_ids": clone_ids,
        "fate_labels": fate_labels,
        "lineage_labels": lineage_labels,
    }


def build_per_timepoint_arrays(
    data: dict[str, np.ndarray],
    selected_indices: dict[Any, np.ndarray] | None = None,
) -> tuple[dict[Any, np.ndarray], dict[Any, np.ndarray | None], dict[Any, np.ndarray | None]]:
    """Group positions, clone ids, and fate labels by timepoint.

    String time labels are remapped to sorted numeric indices to match
    ``LARRYDataModule``; numeric labels are kept as-is.
    """
    positions = data["positions"]
    raw_timepoints = data["timepoints"]
    clone_ids = data["clone_ids"]
    fate_labels = data["fate_labels"]

    unique_labels = np.unique(raw_timepoints)
    # Preserve numeric labels; remap string labels to sorted indices.
    if all(isinstance(l, (int, float, np.integer, np.floating)) for l in unique_labels):
        label_to_num = {label: label for label in unique_labels}
    else:
        sorted_labels = sorted(unique_labels, key=_label_sort_key)
        label_to_num = {label: idx for idx, label in enumerate(sorted_labels)}
    timepoints = np.array([label_to_num[label] for label in raw_timepoints])

    frames: dict[Any, np.ndarray] = {}
    clones: dict[Any, np.ndarray | None] = {}
    fates: dict[Any, np.ndarray | None] = {}

    for label in sorted(label_to_num.values()):
        mask = timepoints == label
        idx = np.where(mask)[0]
        if selected_indices is not None and label in selected_indices:
            sel = selected_indices[label]
            idx = idx[sel]
        frames[label] = positions[idx]
        clones[label] = clone_ids[idx] if clone_ids is not None else None
        fates[label] = fate_labels[idx] if fate_labels is not None else None
    return frames, clones, fates


def _label_sort_key(value: Any):
    """Sort string labels like 'day2', 'day4', 'day6' numerically if possible."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, float(value))
    text = str(value)
    m = re.search(r"(\d+(?:\.\d+)?)$", text)
    if m:
        return (0, float(m.group(1)))
    return (1, text)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def _valid_barcode_mask(clones: np.ndarray | None) -> np.ndarray:
    """Return a boolean mask for barcoded cells (clone ids starting with 'clone_')."""
    if clones is None:
        return np.array([], dtype=bool)
    return np.array([str(c).startswith("clone_") for c in clones])


def _conditional_entropy(source_labels: np.ndarray, target_labels: np.ndarray) -> float:
    """Compute H(target | source) using empirical counts.

    A lower value means the matching is more deterministic given the source label.
    """
    if source_labels.size == 0:
        return 0.0
    joint_counts = Counter(zip(source_labels, target_labels))
    source_counts = Counter(source_labels)
    total = source_labels.size
    ent = 0.0
    for (s, t), count in joint_counts.items():
        p_s = source_counts[s] / total
        p_t_given_s = count / source_counts[s]
        ent -= p_s * p_t_given_s * np.log2(p_t_given_s)
    return float(ent)


def _nmi_ari(source_labels: np.ndarray, target_labels: np.ndarray) -> dict[str, float | None]:
    """Return NMI and ARI between two partitions."""
    if not _HAS_SKLEARN or source_labels.size == 0:
        return {"nmi": None, "ari": None}
    return {
        "nmi": float(normalized_mutual_info_score(source_labels, target_labels)),
        "ari": float(adjusted_rand_score(source_labels, target_labels)),
    }


def _per_label_prf(
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute per-label precision/recall/F1 plus macro/micro/weighted averages.

    Here ``source_labels`` are the true labels (e.g. clone barcodes) and
    ``target_labels`` are the labels predicted by the matching.
    """
    if not _HAS_SKLEARN or source_labels.size == 0:
        return {"per_label": {}, "macro": None, "micro": None, "weighted": None}
    if labels is None:
        labels = np.unique(np.concatenate([source_labels, target_labels]))
    precision, recall, f1, support = precision_recall_fscore_support(
        source_labels,
        target_labels,
        labels=labels,
        average=None,
        zero_division=0.0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        source_labels, target_labels, average="macro", zero_division=0.0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        source_labels, target_labels, average="micro", zero_division=0.0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        source_labels, target_labels, average="weighted", zero_division=0.0
    )
    per_label = {
        str(label): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    return {
        "per_label": per_label,
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "micro": {
            "precision": float(micro_p),
            "recall": float(micro_r),
            "f1": float(micro_f1),
        },
        "weighted": {
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1": float(weighted_f1),
        },
    }


def _transition_matrix(
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    source_names: np.ndarray | None = None,
    target_names: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return empirical transition matrix from source labels to target labels.

    ``transition_matrix[src, tgt]`` is the fraction of source cells with label
    ``src`` that were matched to a target cell with label ``tgt``.
    """
    if source_labels.size == 0:
        return {"matrix": {}, "source_names": [], "target_names": []}
    if source_names is None:
        source_names = np.unique(source_labels)
    if target_names is None:
        target_names = np.unique(target_labels)
    src_to_idx = {str(name): i for i, name in enumerate(source_names)}
    tgt_to_idx = {str(name): i for i, name in enumerate(target_names)}
    counts = np.zeros((len(source_names), len(target_names)), dtype=np.int64)
    for s, t in zip(source_labels, target_labels):
        counts[src_to_idx[str(s)], tgt_to_idx[str(t)]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    probs = counts / row_sums
    matrix = {
        f"{s}_to_{t}": float(probs[i, j])
        for i, s in enumerate(source_names)
        for j, t in enumerate(target_names)
    }
    return {
        "matrix": matrix,
        "counts": counts.tolist(),
        "source_names": [str(x) for x in source_names],
        "target_names": [str(x) for x in target_names],
    }


def _recall_at_k(
    source_pos: np.ndarray,
    target_pos: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    k_list: list[int],
) -> dict[str, float]:
    """Generalized soft-neighbor recall@K.

    For each source cell ``i`` we rank target cells by Euclidean distance from
    ``source_pos[i]`` and ask: in the top-K set, how many have the same label as
    the source?  This tests whether the matching puts the source cell in the
    right *neighborhood* even if the one-to-one match is not perfect.
    """
    if source_labels.size == 0 or target_labels.size == 0:
        return {f"recall@{k}": 0.0 for k in k_list}
    diff = target_pos[np.newaxis, :, :] - source_pos[:, np.newaxis, :]
    dists = np.linalg.norm(diff, axis=2)
    sorted_idx = np.argsort(dists, axis=1)
    out: dict[str, float] = {}
    for k in k_list:
        kk = min(k, target_pos.shape[0])
        topk_idx = sorted_idx[:, :kk]
        topk_labels = target_labels[topk_idx]
        source_labels_tiled = np.tile(source_labels[:, np.newaxis], (1, kk))
        hits = np.any(topk_labels == source_labels_tiled, axis=1)
        out[f"recall@{k}"] = float(hits.mean())
    return out


def _within_clone_random_baseline(
    matching: np.ndarray,
    source_clones: np.ndarray,
    target_clones: np.ndarray,
) -> dict[str, Any]:
    """Random baseline that permutes matches only within each source clone group.

    This tells us how much of the top-1 accuracy comes from simply matching a
    source cell to *any* target cell of the same clone, versus matching it to the
    *correct* target cell.
    """
    valid_source = _valid_barcode_mask(source_clones)
    eval_idx = np.where(valid_source)[0]
    n_eval = len(eval_idx)
    if n_eval == 0:
        return {"top1_accuracy": 0.0, "n_eval": 0}

    rng = np.random.default_rng(42)
    random_matching = matching.copy()
    for clone in np.unique(source_clones[eval_idx]):
        mask = (source_clones == clone) & valid_source
        idx = np.where(mask)[0]
        if len(idx) > 1:
            random_matching[idx] = rng.permutation(matching[idx])

    matched_target_clone = target_clones[random_matching[eval_idx]]
    top1_hits = int(np.sum(source_clones[eval_idx] == matched_target_clone))
    return {
        "top1_accuracy": float(top1_hits / n_eval) if n_eval > 0 else 0.0,
        "top1_hits": int(top1_hits),
        "n_eval": int(n_eval),
    }


# ---------------------------------------------------------------------------
# Matching evaluation
# ---------------------------------------------------------------------------


def evaluate_matching(
    matching: np.ndarray,
    source_pos: np.ndarray,
    target_pos: np.ndarray,
    source_clones: np.ndarray | None,
    target_clones: np.ndarray | None,
    k_list: list[int],
) -> dict[str, Any]:
    """Evaluate a 1-to-1 matching between source and target timepoints.

    ``matching[i]`` is the index of the target cell matched to source cell ``i``.
    The matching may be rectangular (source length may differ from target length).
    """
    metrics: dict[str, Any] = {
        "n_source": int(len(matching)),
        "n_target": int(len(target_clones)) if target_clones is not None else None,
        "has_barcode_source": source_clones is not None,
        "has_barcode_target": target_clones is not None,
    }

    if source_clones is None or target_clones is None:
        log.warning("Clone ids missing; cannot compute barcode metrics.")
        return metrics

    valid_source = _valid_barcode_mask(source_clones)
    valid_target = _valid_barcode_mask(target_clones)
    if not valid_source.any() or not valid_target.any():
        log.warning("No valid barcodes found; cannot compute barcode metrics.")
        return metrics

    eval_source_idx = np.where(valid_source)[0]
    n_eval = len(eval_source_idx)

    # Top-1 accuracy: matched target shares the source barcode.
    matched_target_clone = target_clones[matching[eval_source_idx]]
    top1_hits = int(np.sum(source_clones[eval_source_idx] == matched_target_clone))
    metrics["top1_accuracy"] = float(top1_hits / n_eval) if n_eval > 0 else 0.0
    metrics["top1_hits"] = int(top1_hits)
    metrics["n_eval"] = int(n_eval)

    # Soft-neighbor recall@K using Euclidean distance on the target manifold.
    metrics["recall_at_k"] = _recall_at_k(
        source_pos=source_pos[eval_source_idx],
        target_pos=target_pos,
        source_labels=source_clones[eval_source_idx],
        target_labels=target_clones,
        k_list=k_list,
    )

    # Random baselines.
    rng = np.random.default_rng(42)
    random_matching = rng.permutation(matching)
    random_top1_hits = int(
        np.sum(source_clones[eval_source_idx] == target_clones[random_matching[eval_source_idx]])
    )
    metrics["random_top1_accuracy"] = float(random_top1_hits / n_eval) if n_eval > 0 else 0.0
    metrics["random_top1_hits"] = int(random_top1_hits)
    metrics["within_clone_random_baseline"] = _within_clone_random_baseline(
        matching=matching,
        source_clones=source_clones,
        target_clones=target_clones,
    )

    # Partition-level metrics (NMI, ARI, conditional entropy).
    metrics["partition"] = _nmi_ari(
        source_clones[eval_source_idx],
        target_clones[matching[eval_source_idx]],
    )
    metrics["partition"]["conditional_entropy"] = _conditional_entropy(
        source_clones[eval_source_idx],
        target_clones[matching[eval_source_idx]],
    )

    # Per-clone precision/recall/F1.
    metrics["prf"] = _per_label_prf(
        source_labels=source_clones[eval_source_idx],
        target_labels=target_clones[matching[eval_source_idx]],
        labels=np.unique(np.concatenate([source_clones[eval_source_idx], target_clones])),
    )

    # Per-clone statistics (legacy mean clone precision).
    unique_clones = np.unique(source_clones[eval_source_idx])
    clone_precisions = []
    for c in unique_clones:
        src_mask = source_clones[eval_source_idx] == c
        if not src_mask.any():
            continue
        tgt_idx = matching[eval_source_idx[src_mask]]
        tgt_idx = tgt_idx[tgt_idx < len(target_clones)]
        if len(tgt_idx) == 0:
            continue
        precision = float(np.sum(target_clones[tgt_idx] == c) / len(tgt_idx))
        clone_precisions.append(precision)
    metrics["mean_clone_precision"] = float(np.mean(clone_precisions)) if clone_precisions else 0.0

    return metrics


# ---------------------------------------------------------------------------
# Fate evaluation
# ---------------------------------------------------------------------------


def evaluate_fate_matching(
    matching: np.ndarray,
    source_fates: np.ndarray | None,
    target_fates: np.ndarray | None,
    source_pos: np.ndarray | None = None,
    target_pos: np.ndarray | None = None,
    k_list: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate a 1-to-1 matching against hard fate / cell-type labels."""
    metrics: dict[str, Any] = {
        "has_fate_source": source_fates is not None,
        "has_fate_target": target_fates is not None,
    }
    if source_fates is None or target_fates is None:
        log.warning("Fate labels missing; cannot compute fate metrics.")
        return metrics

    n = len(matching)
    matched_fates = target_fates[matching]
    correct = source_fates == matched_fates
    metrics["fate_top1_accuracy"] = float(correct.mean())
    metrics["fate_top1_hits"] = int(correct.sum())
    metrics["fate_n_eval"] = int(n)

    # Transition matrix.
    fate_names = sorted(set(source_fates) | set(target_fates))
    metrics["transition_matrix"] = _transition_matrix(
        source_labels=source_fates,
        target_labels=matched_fates,
        source_names=np.array(fate_names),
        target_names=np.array(fate_names),
    )

    # Confusion matrix (legacy).
    confusion = {
        f"{src}_to_{tgt}": int(np.sum((source_fates == src) & (matched_fates == tgt)))
        for src in fate_names
        for tgt in fate_names
    }
    metrics["fate_confusion"] = confusion

    # Per-fate accuracy.
    for fate in fate_names:
        mask = source_fates == fate
        if mask.any():
            metrics[f"fate_accuracy_{fate}"] = float(correct[mask].mean())

    # Random baseline.
    rng = np.random.default_rng(42)
    random_matching = rng.permutation(matching)
    random_correct = source_fates == target_fates[random_matching]
    metrics["random_fate_top1_accuracy"] = float(random_correct.mean())
    metrics["random_fate_top1_hits"] = int(random_correct.sum())

    # Partition-level metrics.
    metrics["partition"] = _nmi_ari(source_fates, matched_fates)
    metrics["partition"]["conditional_entropy"] = _conditional_entropy(source_fates, matched_fates)

    # Per-fate PRF.
    metrics["prf"] = _per_label_prf(
        source_labels=source_fates,
        target_labels=matched_fates,
        labels=np.array(fate_names),
    )

    # Soft-neighbor fate recall@K.
    if k_list is not None and source_pos is not None and target_pos is not None:
        metrics["fate_recall_at_k"] = _recall_at_k(
            source_pos=source_pos,
            target_pos=target_pos,
            source_labels=source_fates,
            target_labels=target_fates,
            k_list=k_list,
        )

    return metrics


def _collect_macro_f1(
    per_segment_results: list[dict[str, Any]],
    prf_key: str,
    weight_key: str,
) -> tuple[list[float], list[float]]:
    """Collect macro-F1 values and weights from per-segment PRF dicts."""
    values, weights = [], []
    for seg in per_segment_results:
        prf = seg.get(prf_key, {})
        macro_f1 = prf.get("macro", {}).get("f1") if isinstance(prf.get("macro"), dict) else None
        if macro_f1 is not None and seg.get(weight_key, 0) > 0:
            values.append(float(macro_f1))
            weights.append(float(seg[weight_key]))
    return values, weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    results_dir = Path(args.results_dir)
    matchings_path = results_dir / "checkpoints" / "stage1_psi" / "stage1_matchings.npz"
    if not matchings_path.exists():
        raise FileNotFoundError(f"Stage 1 matchings not found: {matchings_path}")

    data = load_larry_data(args.data_path)
    matchings = np.load(matchings_path, allow_pickle=True)

    matching_keys = [args.matching_key] if args.matching_key else [
        k for k in matchings.files if k.startswith("matching_")
    ]
    if not matching_keys:
        raise ValueError(f"No matching_* key found in {matchings_path}")

    per_segment_results: list[dict[str, Any]] = []

    for matching_key in matching_keys:
        parts = matching_key.replace("matching_t", "").split("_to_t")
        if len(parts) != 2:
            log.warning(f"Cannot parse timepoints from matching key: {matching_key}; skipping")
            continue
        source_label = float(parts[0]) if "." in parts[0] else int(parts[0])
        target_label = float(parts[1]) if "." in parts[1] else int(parts[1])

        selected_indices = None
        src_indices_key = f"indices_t{source_label}"
        tgt_indices_key = f"indices_t{target_label}"
        if src_indices_key in matchings.files:
            selected_indices = {source_label: matchings[src_indices_key]}
            if tgt_indices_key in matchings.files:
                selected_indices[target_label] = matchings[tgt_indices_key]

        frames, clones, fates = build_per_timepoint_arrays(data, selected_indices)
        if source_label not in frames or target_label not in frames:
            log.warning(
                f"Timepoints {source_label}, {target_label} not found. Available: {list(frames.keys())}"
            )
            continue

        matching = matchings[matching_key]
        metrics = evaluate_matching(
            matching=matching,
            source_pos=frames[source_label],
            target_pos=frames[target_label],
            source_clones=clones[source_label],
            target_clones=clones[target_label],
            k_list=args.k_list,
        )
        fate_metrics = evaluate_fate_matching(
            matching=matching,
            source_fates=fates[source_label],
            target_fates=fates[target_label],
            source_pos=frames[source_label],
            target_pos=frames[target_label],
            k_list=args.k_list,
        )
        metrics.update(fate_metrics)
        metrics["source_timepoint"] = source_label
        metrics["target_timepoint"] = target_label
        metrics["matching_key"] = matching_key
        per_segment_results.append(metrics)
        log.info(
            f"Segment {source_label} -> {target_label}: "
            f"fate_acc={metrics.get('fate_top1_accuracy', 0):.3f}, "
            f"clone_top1={metrics.get('top1_accuracy', 0):.3f}, "
            f"clone_NMI={metrics.get('partition', {}).get('nmi', 0):.3f}, "
            f"clone_ARI={metrics.get('partition', {}).get('ari', 0):.3f}"
        )

    if not per_segment_results:
        raise ValueError("No segments were successfully evaluated.")

    # Aggregate across segments (weighted by n_eval / fate_n_eval).
    aggregated: dict[str, Any] = {
        "n_segments": len(per_segment_results),
        "segments": per_segment_results,
    }

    def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
        if not values:
            return None
        return float(np.average(values, weights=weights))

    def _collect(key: str, weight_key: str) -> tuple[list[float], list[float]]:
        values, weights = [], []
        for seg in per_segment_results:
            if key in seg and seg.get(weight_key, 0) > 0:
                values.append(float(seg[key]))
                weights.append(float(seg[weight_key]))
        return values, weights

    # Top-1 accuracies and random baselines.
    aggregated["mean_top1_accuracy"] = _weighted_mean(*_collect("top1_accuracy", "n_eval"))
    aggregated["mean_random_top1_accuracy"] = _weighted_mean(*_collect("random_top1_accuracy", "n_eval"))
    aggregated["mean_fate_top1_accuracy"] = _weighted_mean(*_collect("fate_top1_accuracy", "fate_n_eval"))
    aggregated["mean_random_fate_top1_accuracy"] = _weighted_mean(*_collect("random_fate_top1_accuracy", "fate_n_eval"))

    # Clone partition metrics (NMI, ARI, conditional entropy).
    for key in ["nmi", "ari", "conditional_entropy"]:
        vals, weights = [], []
        for seg in per_segment_results:
            part = seg.get("partition", {})
            if key in part and part[key] is not None and seg.get("n_eval", 0) > 0:
                vals.append(float(part[key]))
                weights.append(float(seg["n_eval"]))
        aggregated[f"mean_clone_{key}"] = _weighted_mean(vals, weights)

    # Clone/fate macro-F1.
    aggregated["mean_macro_f1"] = _weighted_mean(
        *_collect_macro_f1(per_segment_results, prf_key="prf", weight_key="n_eval")
    )
    aggregated["mean_fate_macro_f1"] = _weighted_mean(
        *_collect_macro_f1(per_segment_results, prf_key="prf", weight_key="fate_n_eval")
    )

    # Soft-neighbor recall@K for clones and fates.
    for key, weight_key in [("recall_at_k", "n_eval"), ("fate_recall_at_k", "fate_n_eval")]:
        for k in args.k_list:
            vals, weights = [], []
            metric_name = f"recall@{k}"
            for seg in per_segment_results:
                recall_dict = seg.get(key, {})
                if metric_name in recall_dict and seg.get(weight_key, 0) > 0:
                    vals.append(float(recall_dict[metric_name]))
                    weights.append(float(seg[weight_key]))
            if vals:
                aggregated[f"mean_{key}_{metric_name}"] = _weighted_mean(vals, weights)

    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "larry_barcode_metrics.json"
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    log.info(f"Barcode metrics saved to {output_path}")
    log.info(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()
