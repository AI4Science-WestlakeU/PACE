"""Stage 2 rollout visualization utilities.

Pure drawing functions — all computation (rollout, metric evaluation)
is done by the caller.  These functions accept pre-computed numpy arrays
and metric dicts so that ``src/plot`` has zero dependency on any specific
training method.

Functions
---------
plot_rollout_trajectory_overview
    Single-panel overview: train anchors + rollout trajectories + GT/predicted
    test points.

plot_test_predictions
    Per-test-label GT (left) vs predicted (right) scatter with metrics.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.plot._colors import (
    TRAIN_COLORS,
    TRAJECTORY_COLORS,
    TEST_TRUE_COLORS,
    TEST_PRED_COLORS,
)


# ======================================================================
# 1. Rollout Trajectory Overview
# ======================================================================

def plot_rollout_trajectory_overview(
    train_frames: dict[int, np.ndarray],
    train_labels: list[int],
    test_gt: dict[int, np.ndarray],
    test_pred: dict[int, np.ndarray],
    test_labels: list[int],
    rollout_trajectories: list[dict] | None = None,
    predictor_label: str = "SDE",
) -> Figure:
    """Single-panel: train anchors + rollout trajectories + GT test (x) + predicted test (o).

    Parameters
    ----------
    train_frames : {label: (N, 2) ndarray}
    train_labels : sorted list of train timepoint labels
    test_gt : {label: (N, 2) ndarray}  ground-truth test frames
    test_pred : {label: (N, 2) ndarray}  predicted test frames
    test_labels : sorted list of test timepoint labels
    rollout_trajectories : list of dicts, one per consecutive train segment::

        [{"l0": int, "l1": int, "traj": (steps, n_pts, 2) ndarray}, ...]

        If *None*, trajectory lines are omitted.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # ----- train anchor points -----
    for i, label in enumerate(train_labels):
        frame = train_frames[label]
        color = TRAIN_COLORS[i % len(TRAIN_COLORS)]
        ax.scatter(
            frame[:, 0], frame[:, 1],
            c=color, s=30, alpha=0.7, zorder=3,
            label=f"Train t={label}",
        )
        cx, cy = frame[:, 0].mean(), frame[:, 1].mean()
        ax.annotate(
            f"t={label}", (cx, cy),
            fontsize=9, fontweight="bold", color=color, alpha=0.8,
            ha="center", va="bottom",
            xytext=(0, 8), textcoords="offset points",
        )

    # ----- rollout trajectories (optional) -----
    if rollout_trajectories is not None:
        for seg_i, seg in enumerate(rollout_trajectories):
            traj = seg["traj"]  # (steps, n, 2)
            seg_color = TRAJECTORY_COLORS[seg_i % len(TRAJECTORY_COLORS)]
            n_pts = traj.shape[1]
            for j in range(n_pts):
                ax.plot(
                    traj[:, j, 0], traj[:, j, 1],
                    color=seg_color, alpha=0.15, linewidth=1.2, zorder=1,
                    label=(
                        f"{predictor_label} {seg['l0']}->{seg['l1']}" if j == 0 else None
                    ),
                )

    # ----- test points: true (x) and predicted (o) -----
    for ti, label in enumerate(test_labels):
        true_color = TEST_TRUE_COLORS[ti % len(TEST_TRUE_COLORS)]
        pred_color = TEST_PRED_COLORS[ti % len(TEST_PRED_COLORS)]

        if label in test_gt:
            gt = test_gt[label]
            ax.scatter(
                gt[:, 0], gt[:, 1],
                marker="x", s=40, linewidths=1.2, c=true_color,
                alpha=0.8, zorder=4,
                label=f"Test true t={label}",
            )

        if label in test_pred:
            pred = test_pred[label]
            ax.scatter(
                pred[:, 0], pred[:, 1],
                marker="o", s=40, facecolors="none", edgecolors=pred_color,
                linewidths=1.2, alpha=0.8, zorder=4,
                label=f"Test pred t={label}",
            )

    ax.set_title(
        f"Stage 2: {predictor_label} rollouts with train anchors, GT and predicted test points",
        fontsize=13,
    )
    ax.legend(
        loc="upper right", fontsize=7, framealpha=0.9,
        markerscale=0.8, handletextpad=0.4,
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# ======================================================================
# 2. Per-test-frame: predicted vs GT scatter with MMD / W1 / W2
# ======================================================================

def plot_test_predictions(
    test_gt: dict[int, np.ndarray],
    test_pred: dict[int, np.ndarray],
    test_labels: list[int],
    metrics_per_label: dict[int, dict[str, float]] | None = None,
    predictor_label: str = "ODE",
) -> Figure:
    """For each test timepoint, show GT (left) vs predicted (right) scatter.

    Parameters
    ----------
    test_gt, test_pred : {label: (N, 2) ndarray}
    test_labels : sorted list of test labels
    metrics_per_label : {label: {"mmd": float, "w1": float, "w2": float}}
        If *None*, metric text is omitted from titles.
    """
    n_test = len(test_labels)
    fig, axes = plt.subplots(n_test, 2, figsize=(10, 4 * n_test), squeeze=False)

    for row, label in enumerate(test_labels):
        gt = test_gt.get(label)
        pred = test_pred.get(label)

        # GT
        ax_gt = axes[row, 0]
        if gt is not None:
            ax_gt.scatter(gt[:, 0], gt[:, 1], c="darkred", s=12, alpha=0.7)
        ax_gt.set_title(f"Ground Truth  t={label}")
        ax_gt.set_aspect("equal")

        # Predicted
        ax_pred = axes[row, 1]
        if pred is not None:
            ax_pred.scatter(pred[:, 0], pred[:, 1], c="steelblue", s=12, alpha=0.7)

        title = f"{predictor_label} Predicted  t={label}"
        if metrics_per_label and label in metrics_per_label:
            m = metrics_per_label[label]
            title += (
                f"\nMMD={m['mmd']:.4f}  W1={m['w1']:.4f}  W2={m['w2']:.4f}  GW={m.get('gw', 0):.4f}"
            )
        ax_pred.set_title(title)
        ax_pred.set_aspect("equal")

    fig.suptitle(f"Stage 2: {predictor_label} Rollout — GT vs Predicted", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_rollout_prediction_snapshots(
    pred_frames: dict[int, np.ndarray],
    labels: list[int],
    predictor_label: str = "ODE",
) -> Figure:
    """Overlay rollout-predicted point clouds from all labels in one axes."""
    fig, ax = plt.subplots(figsize=(7, 7))

    for idx, label in enumerate(labels):
        pred = pred_frames.get(label)
        if pred is not None:
            ax.scatter(
                pred[:, 0],
                pred[:, 1],
                c=TRAIN_COLORS[idx % len(TRAIN_COLORS)],
                s=12,
                alpha=0.7,
                label=f"t={label}",
            )

    ax.set_title(f"{predictor_label} Sequential Rollout Predictions")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.suptitle(
        f"Stage 2: {predictor_label} Rollout Point Clouds in a Shared Coordinate System",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout()
    return fig


def plot_full_trajectory_predictions(
    gt_frames: dict[int, np.ndarray],
    pred_frames: dict[int, np.ndarray],
    labels: list[int],
    metrics_per_label: dict[int, dict[str, float]] | None = None,
    predictor_label: str = "ODE",
    source_label: int | None = None,
) -> Figure:
    """Show GT vs rollout-predicted point clouds for every target label.

    Each row corresponds to one label, with GT on the left and rollout
    prediction on the right. This is useful for diagnostics when the model is
    trained on all timepoints and evaluated by rolling out from a shared source
    frame, e.g. ``t=0 -> t_k`` for every ``k``.
    """
    n_labels = len(labels)
    fig, axes = plt.subplots(n_labels, 2, figsize=(10, 4 * n_labels), squeeze=False)

    for row, label in enumerate(labels):
        gt = gt_frames.get(label)
        pred = pred_frames.get(label)

        ax_gt = axes[row, 0]
        if gt is not None:
            ax_gt.scatter(gt[:, 0], gt[:, 1], c="darkred", s=12, alpha=0.7)
        ax_gt.set_title(f"Ground Truth  t={label}")
        ax_gt.set_aspect("equal")

        ax_pred = axes[row, 1]
        if pred is not None:
            ax_pred.scatter(pred[:, 0], pred[:, 1], c="steelblue", s=12, alpha=0.7)

        title = f"{predictor_label} Rollout  t={label}"
        if source_label is not None:
            title += f"  (from t={source_label})"
        if metrics_per_label and label in metrics_per_label:
            m = metrics_per_label[label]
            title += (
                f"\nMMD={m['mmd']:.4f}  W1={m['w1']:.4f}  "
                f"W2={m['w2']:.4f}  GW={m.get('gw', 0):.4f}"
            )
        ax_pred.set_title(title)
        ax_pred.set_aspect("equal")

    fig.suptitle(
        f"Full-Trajectory {predictor_label} Rollout — GT vs Predicted",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_ode_trajectory_overview(
    train_frames: dict[int, np.ndarray],
    train_labels: list[int],
    test_gt: dict[int, np.ndarray],
    test_pred: dict[int, np.ndarray],
    test_labels: list[int],
    ode_trajectories: list[dict] | None = None,
) -> Figure:
    """Backward-compatible wrapper for existing ODE visualizations."""
    return plot_rollout_trajectory_overview(
        train_frames=train_frames,
        train_labels=train_labels,
        test_gt=test_gt,
        test_pred=test_pred,
        test_labels=test_labels,
        rollout_trajectories=ode_trajectories,
        predictor_label="ODE",
    )
