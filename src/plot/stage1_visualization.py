"""Stage 1 (Psi / GeoPath) visualization utilities.

Pure drawing functions — all computation (geodesic interpolation, metric
evaluation) is done by the caller.  These functions accept pre-computed numpy
arrays and metric dicts so that ``src/plot`` has zero dependency on any
specific training method.

Functions
---------
plot_interpolation_paths
    Linear vs geodesic interpolation curves for consecutive train frame pairs.

plot_midpoint_comparison
    GT vs linear vs geodesic midpoint scatter with MMD / W1 / W2 / GW metrics.

plot_psi_magnitude
    Colored scatter of ||psi|| correction magnitude at the midpoint.

plot_trajectory_overview
    Single-panel: train anchors + geodesic trajectories + GT/predicted test.
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
# 1. Interpolation paths: linear vs geodesic
# ======================================================================

def plot_interpolation_paths(
    linear_paths: list[np.ndarray],
    geo_paths: list[np.ndarray],
    bg_frames: list[tuple[np.ndarray, np.ndarray]],
    train_labels: list[int],
) -> Figure:
    """For each consecutive frame pair, draw point-pair interpolation curves.

    Parameters
    ----------
    linear_paths : list of (n_steps, n_pairs, 2) ndarrays, one per transition.
    geo_paths : list of (n_steps, n_pairs, 2) ndarrays, one per transition.
    bg_frames : list of (frame0_ndarray, frame1_ndarray) for scatter background.
    train_labels : sorted list of train timepoint labels.
    """
    n_transitions = len(train_labels) - 1
    fig, axes = plt.subplots(
        n_transitions, 2,
        figsize=(12, 4 * n_transitions),
        squeeze=False,
    )

    for row in range(n_transitions):
        label0, label1 = train_labels[row], train_labels[row + 1]
        f0, f1 = bg_frames[row]
        lin = linear_paths[row]   # (steps, n, 2)
        geo = geo_paths[row]      # (steps, n, 2)
        n_pairs = lin.shape[1]

        # Linear
        ax_lin = axes[row, 0]
        ax_lin.scatter(f0[:, 0], f0[:, 1], c="gray", s=5, alpha=0.3, label=f"t={label0}")
        ax_lin.scatter(f1[:, 0], f1[:, 1], c="silver", s=5, alpha=0.3, label=f"t={label1}")
        for j in range(n_pairs):
            ax_lin.plot(lin[:, j, 0], lin[:, j, 1], "C3-", alpha=0.5, linewidth=0.8)
        ax_lin.set_title(f"Linear  (t={label0} → t={label1})")
        ax_lin.set_aspect("equal")

        # Geodesic
        ax_geo = axes[row, 1]
        ax_geo.scatter(f0[:, 0], f0[:, 1], c="gray", s=5, alpha=0.3, label=f"t={label0}")
        ax_geo.scatter(f1[:, 0], f1[:, 1], c="silver", s=5, alpha=0.3, label=f"t={label1}")
        for j in range(n_pairs):
            ax_geo.plot(geo[:, j, 0], geo[:, j, 1], "C0-", alpha=0.5, linewidth=0.8)
        ax_geo.set_title(f"Geodesic (psi)  (t={label0} → t={label1})")
        ax_geo.set_aspect("equal")

    fig.suptitle("Stage 1: Interpolation Paths — Linear vs Geodesic", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ======================================================================
# 2. Midpoint comparison: GT vs linear vs geodesic
# ======================================================================

def plot_midpoint_comparison(
    test_gt: dict[int, np.ndarray],
    linear_pred: dict[int, np.ndarray],
    geodesic_pred: dict[int, np.ndarray],
    test_labels: list[int],
    metrics_per_label: dict[int, dict[str, dict[str, float]]] | None = None,
) -> Figure:
    """For each test frame, compare GT vs linear vs geodesic midpoint scatter.

    Parameters
    ----------
    test_gt : {label: (N, 2) ndarray}
    linear_pred : {label: (N, 2) ndarray}
    geodesic_pred : {label: (N, 2) ndarray}
    test_labels : sorted list of test labels
    metrics_per_label : {label: {"linear": {"mmd": …, "w1": …, "w2": …},
                                 "geodesic": {…}}}
    """
    n_test = len(test_labels)
    fig, axes = plt.subplots(n_test, 3, figsize=(15, 4 * n_test), squeeze=False)

    for row, label in enumerate(test_labels):
        gt = test_gt.get(label)
        lin = linear_pred.get(label)
        geo = geodesic_pred.get(label)

        # GT
        ax_gt = axes[row, 0]
        if gt is not None:
            ax_gt.scatter(gt[:, 0], gt[:, 1], c="darkred", s=8, alpha=0.7)
        ax_gt.set_title(f"Ground Truth  t={label}")
        ax_gt.set_aspect("equal")

        # Linear
        ax_lin = axes[row, 1]
        if lin is not None:
            ax_lin.scatter(lin[:, 0], lin[:, 1], c="gray", s=8, alpha=0.7)
        title_lin = "Linear Interp."
        if metrics_per_label and label in metrics_per_label:
            m = metrics_per_label[label]["linear"]
            title_lin += f"\nMMD={m['mmd']:.4f}  W1={m['w1']:.4f}  W2={m['w2']:.4f}  GW={m.get('gw', 0):.4f}"
        ax_lin.set_title(title_lin)
        ax_lin.set_aspect("equal")

        # Geodesic
        ax_geo = axes[row, 2]
        if geo is not None:
            ax_geo.scatter(geo[:, 0], geo[:, 1], c="steelblue", s=8, alpha=0.7)
        title_geo = "Geodesic (psi)"
        if metrics_per_label and label in metrics_per_label:
            m = metrics_per_label[label]["geodesic"]
            title_geo += f"\nMMD={m['mmd']:.4f}  W1={m['w1']:.4f}  W2={m['w2']:.4f}  GW={m.get('gw', 0):.4f}"
        ax_geo.set_title(title_geo)
        ax_geo.set_aspect("equal")

    fig.suptitle("Stage 1: Midpoint Comparison — GT vs Linear vs Geodesic", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ======================================================================
# 3. Psi correction magnitude
# ======================================================================

def plot_psi_magnitude(
    midpoints: list[np.ndarray],
    psi_norms: list[np.ndarray],
    transition_labels: list[tuple[int, int]],
) -> Figure:
    """For each frame pair, plot ||psi|| colored scatter at the midpoint location.

    Parameters
    ----------
    midpoints : list of (N, 2) ndarrays, one per transition.
    psi_norms : list of (N,) ndarrays, one per transition.
    transition_labels : list of (label0, label1) tuples.
    """
    n_transitions = len(transition_labels)
    fig, axes = plt.subplots(1, n_transitions, figsize=(5 * n_transitions, 4), squeeze=False)

    for col, ((l0, l1), mid, norms) in enumerate(
        zip(transition_labels, midpoints, psi_norms)
    ):
        ax = axes[0, col]
        sc = ax.scatter(
            mid[:, 0], mid[:, 1],
            c=norms, cmap="viridis", s=12, alpha=0.8,
        )
        plt.colorbar(sc, ax=ax, label="||psi||")
        ax.set_title(f"||psi||  (t={l0} → t={l1})")
        ax.set_aspect("equal")

    fig.suptitle("Stage 1: Psi Correction Magnitude at t=0.5", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ======================================================================
# 4. Trajectory overview
# ======================================================================

def plot_trajectory_overview(
    train_frames: dict[int, np.ndarray],
    train_labels: list[int],
    test_gt: dict[int, np.ndarray],
    test_pred: dict[int, np.ndarray],
    test_labels: list[int],
    trajectories: list[dict] | None = None,
) -> Figure:
    """Single-panel: train anchors + geodesic trajectories + GT/predicted test.

    Parameters
    ----------
    train_frames : {label: (N, 2) ndarray}
    train_labels : sorted list of train labels
    test_gt : {label: (N, 2) ndarray}
    test_pred : {label: (N, 2) ndarray}
    test_labels : sorted list of test labels
    trajectories : list of dicts, one per consecutive train segment::

        [{"l0": int, "l1": int, "traj": (steps, n_pts, 2) ndarray}, ...]
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Train anchors
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

    # Geodesic trajectories
    if trajectories is not None:
        for seg_i, seg in enumerate(trajectories):
            traj = seg["traj"]  # (steps, n, 2)
            seg_color = TRAJECTORY_COLORS[seg_i % len(TRAJECTORY_COLORS)]
            n_pts = traj.shape[1]
            for j in range(n_pts):
                ax.plot(
                    traj[:, j, 0], traj[:, j, 1],
                    color=seg_color, alpha=0.15, linewidth=1.2, zorder=1,
                    label=(
                        f"Trajectory {seg['l0']}->{seg['l1']}" if j == 0 else None
                    ),
                )

    # Test points: true (x) and predicted (o)
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
        "Stage 1: Geodesic trajectories with train anchors, GT and predicted test points",
        fontsize=13,
    )
    ax.legend(
        loc="upper right", fontsize=7, framealpha=0.9,
        markerscale=0.8, handletextpad=0.4,
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig
