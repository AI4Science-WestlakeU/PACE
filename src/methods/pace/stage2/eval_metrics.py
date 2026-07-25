"""Evaluation metrics for comparing predicted and ground-truth point clouds."""

from __future__ import annotations

import csv
import math
import os
from collections import Counter

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

try:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from scipy.stats import pearsonr, spearmanr

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


def _import_pot():
    try:
        import ot as pot
    except ImportError:
        return None
    return pot


def _get_module_device(module: torch.nn.Module, fallback: str = "cpu") -> torch.device:
    """Return the device backing a module's parameters or buffers."""
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device(fallback)


def _median_bandwidth_squared(x: torch.Tensor, y: torch.Tensor) -> float:
    joined = torch.cat([x, y], dim=0)
    if joined.shape[0] < 2:
        return 1.0

    dist2 = torch.pdist(joined).pow(2)
    positive = dist2[dist2 > 0]
    if positive.numel() == 0:
        return 1.0
    return float(positive.median().item())


def gaussian_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    """Gaussian-kernel MMD with a median-heuristic bandwidth.

    Returns the square-rooted MMD so the reported value is on the same scale as
    the underlying point coordinates.
    """
    x = x.detach().float().cpu()
    y = y.detach().float().cpu()

    sigma2 = max(_median_bandwidth_squared(x, y), 1e-12)

    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)

    k_xx = torch.exp(-xx / (2.0 * sigma2))
    k_yy = torch.exp(-yy / (2.0 * sigma2))
    k_xy = torch.exp(-xy / (2.0 * sigma2))

    if x.shape[0] > 1 and y.shape[0] > 1:
        term_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (x.shape[0] * (x.shape[0] - 1))
        term_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (y.shape[0] * (y.shape[0] - 1))
    else:
        term_xx = k_xx.mean()
        term_yy = k_yy.mean()

    term_xy = k_xy.mean()
    mmd2 = torch.clamp(term_xx + term_yy - 2.0 * term_xy, min=0.0)
    return float(torch.sqrt(mmd2).item())


def wasserstein_1(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute W1 with Euclidean ground cost."""
    x = x.detach().float().cpu()
    y = y.detach().float().cpu()
    pot = _import_pot()
    if pot is None:
        n = min(x.shape[0], y.shape[0])
        cost = torch.cdist(x[:n], y[:n], p=2).numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        return float(cost[row_ind, col_ind].mean())

    a = pot.unif(x.shape[0])
    b = pot.unif(y.shape[0])
    cost = torch.cdist(x, y, p=2).numpy()
    return float(pot.emd2(a, b, cost))


def wasserstein_2(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute W2 with Euclidean ground cost."""
    x = x.detach().float().cpu()
    y = y.detach().float().cpu()
    pot = _import_pot()
    if pot is None:
        n = min(x.shape[0], y.shape[0])
        cost_sq = torch.cdist(x[:n], y[:n], p=2).pow(2).numpy()
        row_ind, col_ind = linear_sum_assignment(cost_sq)
        return math.sqrt(max(float(cost_sq[row_ind, col_ind].mean()), 0.0))

    a = pot.unif(x.shape[0])
    b = pot.unif(y.shape[0])
    cost_sq = torch.cdist(x, y, p=2).pow(2).numpy()
    w2_sq = float(pot.emd2(a, b, cost_sq))
    return math.sqrt(max(w2_sq, 0.0))


def gromov_wasserstein(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute the Gromov-Wasserstein distance (squared Euclidean intra-cost).

    Uses ``ot.gromov.gromov_wasserstein2`` with uniform weights and squared
    Euclidean intra-distance matrices, consistent with the W2 convention.
    """
    x = x.detach().float().cpu()
    y = y.detach().float().cpu()
    pot = _import_pot()
    if pot is None:
        return float("nan")

    C1 = torch.cdist(x, x, p=2).pow(2).numpy()
    C2 = torch.cdist(y, y, p=2).pow(2).numpy()
    p = pot.unif(x.shape[0])
    q = pot.unif(y.shape[0])
    gw_val = float(pot.gromov.gromov_wasserstein2(C1, C2, p, q, loss_fun="square_loss"))
    return math.sqrt(max(gw_val, 0.0))


def compute_distribution_metrics(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Compute MMD, W1, W2 and GW for two point clouds."""
    return {
        "mmd": gaussian_mmd(x, y),
        "w1": wasserstein_1(x, y),
        "w2": wasserstein_2(x, y),
        "gw": gromov_wasserstein(x, y),
    }


def evaluate_stage1_midpoint_metrics(
    geopath_net: torch.nn.Module,
    flow_matcher,
    train_frames: dict,
    test_frames: dict,
    train_labels: list,
    test_labels: list,
    device: str = "cpu",
) -> list[dict[str, float | int | str]]:
    """Evaluate Stage 1 linear and geodesic midpoint predictions on held-out labels."""
    from src.methods.pace.stage2.flow_matcher import labels_to_timesteps

    model_device = _get_module_device(geopath_net, fallback=device)
    all_labels = sorted(set(train_labels) | set(test_labels))
    timesteps = labels_to_timesteps(all_labels)
    label_to_t = {label: timesteps[i] for i, label in enumerate(all_labels)}

    geopath_net.eval()
    rows: list[dict[str, float | int | str]] = []

    for test_label in test_labels:
        idx = all_labels.index(test_label)
        prev_label = all_labels[idx - 1] if idx > 0 else None
        next_label = all_labels[idx + 1] if idx < len(all_labels) - 1 else None

        if prev_label is None or next_label is None:
            continue
        if prev_label not in train_frames or next_label not in train_frames:
            continue
        if test_label not in test_frames:
            continue

        f0 = train_frames[prev_label].to(model_device)
        f1 = train_frames[next_label].to(model_device)
        gt = test_frames[test_label].float().cpu()

        t_min = label_to_t[prev_label]
        t_max = label_to_t[next_label]
        t_mid = label_to_t[test_label]
        s = (t_mid - t_min) / (t_max - t_min)

        n = min(f0.shape[0], f1.shape[0])
        x0 = f0[:n]
        x1 = f1[:n]

        with torch.no_grad():
            lin_mid = (1 - s) * x0 + s * x1
            t_batch = torch.full((n,), t_mid, device=model_device)
            gamma_val = flow_matcher.gamma(t_batch, t_min, t_max).unsqueeze(-1)
            psi = geopath_net(x0, x1, t_batch.unsqueeze(-1))
            geo_mid = lin_mid + gamma_val * psi

        lin_metrics = compute_distribution_metrics(lin_mid.cpu(), gt)
        geo_metrics = compute_distribution_metrics(geo_mid.cpu(), gt)

        rows.append(
            {
                "stage": "stage1",
                "test_label": test_label,
                "predictor": "linear",
                "prev_train_label": prev_label,
                "next_train_label": next_label,
                "mmd": lin_metrics["mmd"],
                "w1": lin_metrics["w1"],
                "w2": lin_metrics["w2"],
                "gw": lin_metrics["gw"],
            }
        )
        rows.append(
            {
                "stage": "stage1",
                "test_label": test_label,
                "predictor": "geodesic",
                "prev_train_label": prev_label,
                "next_train_label": next_label,
                "mmd": geo_metrics["mmd"],
                "w1": geo_metrics["w1"],
                "w2": geo_metrics["w2"],
                "gw": geo_metrics["gw"],
            }
        )

    return rows


def _run_ode_rollout(
    flow_net: torch.nn.Module,
    source: torch.Tensor,
    t_source: float,
    t_target: float,
    n_steps: int,
    device: str,
) -> torch.Tensor:
    from torchdyn.core import NeuralODE

    from src.methods.pace.stage2.networks import FlowModelWrapper

    model_device = _get_module_device(flow_net, fallback=device)
    node = NeuralODE(
        FlowModelWrapper(flow_net),
        solver="euler",
        sensitivity="adjoint",
        atol=1e-5,
        rtol=1e-5,
    )
    source = source.to(model_device)
    t_span = torch.linspace(t_source, t_target, n_steps).to(model_device)
    with torch.no_grad():
        return node.trajectory(source, t_span=t_span)


def _find_train_bracket(
    test_label,
    train_labels: list,
) -> tuple:
    """Find the surrounding train anchors for a held-out test label.

    Returns (prev_train, next_train) or (None, None) if the test label
    falls outside the train range.
    """
    sorted_train = sorted(train_labels)
    prev_train = None
    next_train = None
    for tl in sorted_train:
        if tl <= test_label:
            prev_train = tl
        if tl >= test_label and next_train is None:
            next_train = tl
    if prev_train == test_label or next_train == test_label:
        return None, None  # test label coincides with a train label
    return prev_train, next_train


def evaluate_stage2_rollout_metrics(
    flow_net: torch.nn.Module,
    train_frames: dict,
    test_frames: dict,
    train_labels: list,
    test_labels: list,
    n_ode_steps: int = 101,
    device: str = "cpu",
) -> list[dict[str, float | int | str]]:
    """Evaluate Stage 2 ODE rollouts on held-out labels.

    For each test label, finds the bracketing train anchors (prev, next),
    runs ODE from prev→next using the **training time scale**, and extracts
    the prediction at the intermediate time corresponding to the test label.
    This is consistent with how the velocity field was trained.
    """
    from src.methods.pace.stage2.flow_matcher import labels_to_timesteps

    sorted_train = sorted(train_labels)
    # Training time mapping — must be identical to flow_train.py training_step
    train_timesteps = labels_to_timesteps(sorted_train)
    train_label_to_t = {label: train_timesteps[i] for i, label in enumerate(sorted_train)}

    flow_net.eval()
    rows: list[dict[str, float | int | str]] = []

    for test_label in test_labels:
        if test_label not in test_frames:
            continue

        prev_label, next_label = _find_train_bracket(test_label, train_labels)
        if prev_label is None or next_label is None:
            continue
        if prev_label not in train_frames or next_label not in train_frames:
            continue
            
        t_prev = train_label_to_t[prev_label]
        t_next = train_label_to_t[next_label]

        # Ratio of the test label within the bracket
        ratio = (test_label - prev_label) / (next_label - prev_label)

        source = train_frames[prev_label]
        gt = test_frames[test_label].float().cpu()

        # ODE rollout from prev_train to next_train (full bracket)
        traj = _run_ode_rollout(
            flow_net=flow_net,
            source=source,
            t_source=t_prev,
            t_target=t_next,
            n_steps=n_ode_steps,
            device=device,
        )

        # Extract prediction at the intermediate time
        query_idx = int(round(ratio * (n_ode_steps - 1)))
        pred = traj[query_idx].cpu()
        metrics = compute_distribution_metrics(pred, gt)

        rows.append(
            {
                "stage": "stage2",
                "test_label": test_label,
                "predictor": "ode_rollout",
                "prev_train_label": prev_label,
                "next_train_label": next_label,
                "mmd": metrics["mmd"],
                "w1": metrics["w1"],
                "w2": metrics["w2"],
                "gw": metrics["gw"],
            }
        )

    return rows


def compute_velocity_alignment_metrics(
    flow_net: torch.nn.Module,
    frames: dict,
    velocity_frames: dict,
    eval_labels: list,
    train_labels: list,
    batch_size: int = 2048,
    device: str = "cpu",
    stage: str = "stage2",
    predictor: str = "velocity_field",
) -> list[dict[str, float | int | str]]:
    """Compare learned velocities with reference velocities.

    This reports the Table-4-style RNA-velocity metrics:
    mean cosine distance and mean L2 norm.
    """
    from src.methods.pace.stage2.flow_matcher import labels_to_timesteps

    sorted_train = sorted(train_labels)
    train_timesteps = labels_to_timesteps(sorted_train)
    train_label_to_t = {label: train_timesteps[i] for i, label in enumerate(sorted_train)}

    model_device = _get_module_device(flow_net, fallback=device)
    flow_net.eval()
    rows: list[dict[str, float | int | str]] = []

    for label in eval_labels:
        if label not in frames or label not in velocity_frames:
            continue

        prev_label, next_label = _find_train_bracket(label, train_labels)
        if prev_label is None or next_label is None:
            if label not in train_label_to_t:
                continue
            t_eval = train_label_to_t[label]
        else:
            t_prev = train_label_to_t[prev_label]
            t_next = train_label_to_t[next_label]
            ratio = (label - prev_label) / (next_label - prev_label)
            t_eval = t_prev + ratio * (t_next - t_prev)

        x = frames[label].float()
        v_true = velocity_frames[label].float()
        n = min(x.shape[0], v_true.shape[0])
        x = x[:n]
        v_true = v_true[:n]

        pred_chunks = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                xb = x[start:start + batch_size].to(model_device)
                tb = torch.full((xb.shape[0],), float(t_eval), device=model_device)
                pred_chunks.append(flow_net(tb, xb).detach().cpu())
        v_pred = torch.cat(pred_chunks, dim=0)

        cos_dist = (
            1.0 - torch.nn.functional.cosine_similarity(v_pred, v_true, dim=1)
        ).mean()
        l2 = torch.norm(v_pred - v_true, dim=1).mean()

        rows.append(
            {
                "stage": stage,
                "test_label": label,
                "predictor": predictor,
                "cos_dist": float(cos_dist.item()),
                "l2": float(l2.item()),
            }
        )

    return rows


def build_full_trajectory_rollout_predictions(
    flow_net: torch.nn.Module,
    frames: dict,
    labels: list,
    source_label,
    target_labels: list | None = None,
    n_ode_steps: int = 101,
    device: str = "cpu",
) -> dict[int, torch.Tensor]:
    """Roll out from one source label to multiple target labels.

    Parameters
    ----------
    flow_net:
        Velocity field model.
    frames:
        Mapping ``label -> point cloud tensor``.
    labels:
        Ordered label universe used to build the time mapping.
    source_label:
        Label used as the rollout source.
    target_labels:
        Optional subset of labels to predict. Defaults to ``labels`` order.
    """
    from src.methods.pace.stage2.flow_matcher import labels_to_timesteps

    if source_label not in frames:
        return {}

    ordered_labels = list(labels)
    label_to_t = {
        label: labels_to_timesteps(ordered_labels)[i]
        for i, label in enumerate(ordered_labels)
    }
    targets = list(ordered_labels if target_labels is None else target_labels)
    source = frames[source_label]

    pred_frames: dict[int, torch.Tensor] = {}
    for target_label in targets:
        if target_label not in frames or target_label not in label_to_t:
            continue
        if target_label == source_label:
            pred_frames[target_label] = source.detach().float().cpu()
            continue

        traj = _run_ode_rollout(
            flow_net=flow_net,
            source=source,
            t_source=label_to_t[source_label],
            t_target=label_to_t[target_label],
            n_steps=n_ode_steps,
            device=device,
        )
        pred_frames[target_label] = traj[-1].detach().float().cpu()

    return pred_frames


def evaluate_full_trajectory_rollout_metrics(
    flow_net: torch.nn.Module,
    frames: dict,
    labels: list,
    source_label,
    target_labels: list | None = None,
    n_ode_steps: int = 101,
    device: str = "cpu",
    predictor: str = "ode_rollout_from_t0",
    stage: str = "stage2_fulltraj",
) -> list[dict[str, float | int | str]]:
    """Evaluate rollout fidelity from one source label to every target label.

    This is intended for full-trajectory diagnostics where all timepoints are
    available during training and we want to measure how well the learned flow
    reproduces each frame when rolled out from a shared source frame.
    """
    targets = list(labels if target_labels is None else target_labels)
    pred_frames = build_full_trajectory_rollout_predictions(
        flow_net=flow_net,
        frames=frames,
        labels=labels,
        source_label=source_label,
        target_labels=targets,
        n_ode_steps=n_ode_steps,
        device=device,
    )

    rows: list[dict[str, float | int | str]] = []
    for target_label in targets:
        if target_label not in frames or target_label not in pred_frames:
            continue

        gt = frames[target_label].detach().float().cpu()
        pred = pred_frames[target_label].detach().float().cpu()
        metrics = compute_distribution_metrics(pred, gt)
        rows.append(
            {
                "stage": stage,
                "test_label": target_label,
                "predictor": predictor,
                "prev_train_label": source_label,
                "next_train_label": target_label,
                "mmd": metrics["mmd"],
                "w1": metrics["w1"],
                "w2": metrics["w2"],
                "gw": metrics["gw"],
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Ground-truth trajectory-level metrics
# ---------------------------------------------------------------------------


def _estimate_label_probs_knn(
    predicted: np.ndarray,
    reference: np.ndarray,
    reference_labels: np.ndarray,
    k: int = 10,
    kernel: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate label probabilities for ``predicted`` points using kNN on ``reference``.

    Parameters
    ----------
    predicted:
        Array of shape ``(n_pred, d)``.
    reference:
        Array of shape ``(n_ref, d)``.
    reference_labels:
        Array of shape ``(n_ref,)``.
    k:
        Number of neighbors.
    kernel:
        ``"uniform"`` (simple voting) or ``"distance"`` (inverse-distance weighted).

    Returns
    -------
    probs:
        Array of shape ``(n_pred, n_classes)``.
    classes:
        Array of label names, length ``n_classes``.
    """
    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    reference_labels = np.asarray(reference_labels)

    classes, inv = np.unique(reference_labels, return_inverse=True)
    n_classes = len(classes)
    n_pred = predicted.shape[0]

    if reference.shape[0] == 0:
        return np.ones((n_pred, n_classes)) / max(n_classes, 1), classes

    k_eff = min(int(k), reference.shape[0])
    tree = cKDTree(reference)
    dists, idx = tree.query(predicted, k=k_eff)
    if k_eff == 1:
        dists = dists[:, np.newaxis]
        idx = idx[:, np.newaxis]

    probs = np.zeros((n_pred, n_classes), dtype=np.float64)
    if kernel == "uniform":
        for i in range(n_pred):
            neighbor_labels = inv[idx[i]]
            probs[i] = np.bincount(neighbor_labels, minlength=n_classes) / k_eff
    elif kernel == "distance":
        eps = 1e-8
        weights = 1.0 / (dists + eps)
        weights = weights / (weights.sum(axis=1, keepdims=True) + eps)
        for i in range(n_pred):
            neighbor_labels = inv[idx[i]]
            probs[i] = np.bincount(neighbor_labels, weights=weights[i], minlength=n_classes)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    return probs, classes


def _onehot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return a one-hot encoding of ``labels`` using ``classes`` as the universe."""
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    out = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for i, lab in enumerate(labels):
        out[i, class_to_idx[str(lab)]] = 1.0
    return out


def _top1_accuracy(probs: np.ndarray, true_labels: np.ndarray, classes: np.ndarray) -> float:
    pred_labels = classes[np.argmax(probs, axis=1)]
    return float(np.mean(pred_labels == true_labels))


def _brier_score(probs: np.ndarray, true_onehot: np.ndarray) -> float:
    return float(np.mean(np.sum((probs - true_onehot) ** 2, axis=1)))


def _cross_entropy(probs: np.ndarray, true_onehot: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.mean(np.sum(true_onehot * np.log(probs + eps), axis=1)))


def _mean_entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(np.sum(probs * np.log(probs + eps), axis=1)))


def _total_variation(p_pred: np.ndarray, p_true: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p_pred - p_true)))


def _js_divergence(p_pred: np.ndarray, p_true: np.ndarray, eps: float = 1e-12) -> float:
    m = 0.5 * (p_pred + p_true)
    kl = lambda p, q: np.sum(p * np.log((p + eps) / (q + eps)))
    return float(0.5 * (kl(p_pred, m) + kl(p_true, m)))


def _expected_calibration_error(
    true_onehot: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Multi-class expected calibration error using the true-class probability."""
    true_class_prob = np.sum(true_onehot * probs, axis=1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = true_onehot.shape[0]
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == 1.0:
            mask = (true_class_prob >= lo) & (true_class_prob <= hi)
        else:
            mask = (true_class_prob >= lo) & (true_class_prob < hi)
        if mask.sum() == 0:
            continue
        conf = float(true_class_prob[mask].mean())
        acc = float(true_onehot[mask].sum() / mask.sum())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def evaluate_stage2_label_probability_over_time(
    flow_net: torch.nn.Module,
    train_frames: dict,
    test_frames: dict,
    train_labels: list,
    test_labels: list,
    label_frames: dict,
    label_name: str = "label",
    k: int = 10,
    n_ode_steps: int = 101,
    device: str = "cpu",
) -> list[dict[str, float | int | str]]:
    """Evaluate Stage-2 ODE rollout on a held-out label using kNN label probabilities.

    For each test label we roll from the bracketing train anchors, extract the
    predicted cloud at the held-out time, and estimate a probability distribution
    over labels using kNN on the empirical test frame.  This measures whether the
    learned dynamics place each cell in the correct label basin, not merely whether
    the marginal distribution matches.
    """
    sorted_train = sorted(train_labels)
    from src.methods.pace.stage2.flow_matcher import labels_to_timesteps

    train_timesteps = labels_to_timesteps(sorted_train)
    train_label_to_t = {label: train_timesteps[i] for i, label in enumerate(sorted_train)}

    flow_net.eval()
    rows: list[dict[str, float | int | str]] = []

    for test_label in test_labels:
        if test_label not in test_frames or test_label not in label_frames:
            continue
        prev_label, next_label = _find_train_bracket(test_label, train_labels)
        if prev_label is None or next_label is None:
            continue
        if prev_label not in train_frames or next_label not in train_frames:
            continue

        t_prev = train_label_to_t[prev_label]
        t_next = train_label_to_t[next_label]
        ratio = (test_label - prev_label) / (next_label - prev_label)

        traj = _run_ode_rollout(
            flow_net=flow_net,
            source=train_frames[prev_label],
            t_source=t_prev,
            t_target=t_next,
            n_steps=n_ode_steps,
            device=device,
        )
        query_idx = int(round(ratio * (n_ode_steps - 1)))
        pred = traj[query_idx].detach().cpu().numpy()

        gt = test_frames[test_label].detach().float().cpu().numpy()
        gt_labels = np.asarray(label_frames[test_label])

        probs, classes = _estimate_label_probs_knn(pred, gt, gt_labels, k=k, kernel="uniform")
        true_onehot = _onehot(gt_labels, classes)

        # Empirical marginal of the true labels.
        empirical = true_onehot.mean(axis=0)
        pred_marginal = probs.mean(axis=0)

        rows.append(
            {
                "stage": "stage2",
                "test_label": test_label,
                "predictor": f"{label_name}_prob_knn",
                "prev_train_label": prev_label,
                "next_train_label": next_label,
                "k": k,
                "n_pred": int(pred.shape[0]),
                "n_ref": int(gt.shape[0]),
                "n_classes": int(len(classes)),
                "top1_accuracy": _top1_accuracy(probs, gt_labels, classes),
                "brier_score": _brier_score(probs, true_onehot),
                "cross_entropy": _cross_entropy(probs, true_onehot),
                "mean_entropy": _mean_entropy(probs),
                "total_variation": _total_variation(pred_marginal, empirical),
                "js_divergence": _js_divergence(pred_marginal, empirical),
                "ece": _expected_calibration_error(true_onehot, probs, n_bins=10),
            }
        )

    return rows


def evaluate_stage2_fate_probability_over_time(
    flow_net: torch.nn.Module,
    train_frames: dict,
    test_frames: dict,
    train_labels: list,
    test_labels: list,
    fate_frames: dict,
    k: int = 10,
    n_ode_steps: int = 101,
    device: str = "cpu",
) -> list[dict[str, float | int | str]]:
    """Wrapper around ``evaluate_stage2_label_probability_over_time`` for fate labels.

    For binary fates additionally reports AUC-ROC, average precision, and ECE.
    """
    rows = evaluate_stage2_label_probability_over_time(
        flow_net=flow_net,
        train_frames=train_frames,
        test_frames=test_frames,
        train_labels=train_labels,
        test_labels=test_labels,
        label_frames=fate_frames,
        label_name="fate",
        k=k,
        n_ode_steps=n_ode_steps,
        device=device,
    )

    if not _HAS_SKLEARN:
        return rows

    # Recompute binary metrics if exactly two fate classes are present.
    sorted_train = sorted(train_labels)
    from src.methods.pace.stage2.flow_matcher import labels_to_timesteps

    train_timesteps = labels_to_timesteps(sorted_train)
    train_label_to_t = {label: train_timesteps[i] for i, label in enumerate(sorted_train)}

    for row in rows:
        test_label = row["test_label"]
        prev_label, next_label = _find_train_bracket(test_label, train_labels)
        if prev_label is None or next_label is None:
            continue

        t_prev = train_label_to_t[prev_label]
        t_next = train_label_to_t[next_label]
        ratio = (test_label - prev_label) / (next_label - prev_label)

        traj = _run_ode_rollout(
            flow_net=flow_net,
            source=train_frames[prev_label],
            t_source=t_prev,
            t_target=t_next,
            n_steps=n_ode_steps,
            device=device,
        )
        query_idx = int(round(ratio * (n_ode_steps - 1)))
        pred = traj[query_idx].detach().cpu().numpy()

        gt = test_frames[test_label].detach().float().cpu().numpy()
        gt_fates = np.asarray(fate_frames[test_label])
        classes = np.unique(gt_fates)
        if len(classes) != 2:
            continue

        probs, cls = _estimate_label_probs_knn(pred, gt, gt_fates, k=k, kernel="uniform")
        # Map the two classes to {0, 1}.
        positive_class = cls[1]
        y_true = (gt_fates == positive_class).astype(np.int64)
        y_score = probs[:, 1]

        try:
            row["auc_roc"] = float(roc_auc_score(y_true, y_score))
            row["average_precision"] = float(average_precision_score(y_true, y_score))
            row["brier_score_loss"] = float(brier_score_loss(y_true, y_score))
        except Exception:
            pass

    return rows


def evaluate_branch_accuracy(
    flow_net: torch.nn.Module,
    frames: dict,
    labels: list,
    source_label,
    terminal_label,
    branch_label_frames: dict,
    clone_label_frames: dict,
    n_ode_steps: int = 101,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate whether early clones are rolled to the correct terminal branch.

    For each clone observed at ``source_label`` we define the true terminal branch
    as the majority fate/cell-type among that clone's cells at ``terminal_label``.
    We roll source cells to ``terminal_label`` and assign each predicted cell a
    branch by nearest-neighbor vote in the terminal frame.  The predicted branch
    for the clone is the majority branch among its rolled members.
    """
    if source_label not in frames or terminal_label not in frames:
        return {"error": "source or terminal label missing"}
    if source_label not in clone_label_frames or terminal_label not in branch_label_frames:
        return {"error": "clone or branch labels missing"}

    from src.methods.pace.stage2.flow_matcher import labels_to_timesteps

    ordered_labels = sorted(labels)
    label_to_t = {label: labels_to_timesteps(ordered_labels)[i] for i, label in enumerate(ordered_labels)}

    traj = _run_ode_rollout(
        flow_net=flow_net,
        source=frames[source_label],
        t_source=label_to_t[source_label],
        t_target=label_to_t[terminal_label],
        n_steps=n_ode_steps,
        device=device,
    )
    pred_terminal = traj[-1].detach().cpu().numpy()

    true_terminal_positions = frames[terminal_label].detach().float().cpu().numpy()
    true_terminal_branches = np.asarray(branch_label_frames[terminal_label])
    source_clones = np.asarray(clone_label_frames[source_label])

    # Nearest-neighbor branch assignment for rolled cells.
    tree = cKDTree(true_terminal_positions)
    _, idx = tree.query(pred_terminal, k=1)
    predicted_branches = true_terminal_branches[idx]

    branch_names = sorted(set(true_terminal_branches))
    true_branches = []
    pred_branches = []
    clone_consistencies = []

    for clone in np.unique(source_clones):
        mask = source_clones == clone
        if mask.sum() == 0:
            continue
        true_terminal_idx = np.where(
            np.asarray(clone_label_frames[terminal_label]) == clone
        )[0]
        if len(true_terminal_idx) == 0:
            continue
        true_branch = str(Counter(true_terminal_branches[true_terminal_idx]).most_common(1)[0][0])
        pred_branch = str(Counter(predicted_branches[mask]).most_common(1)[0][0])
        true_branches.append(true_branch)
        pred_branches.append(pred_branch)
        clone_consistencies.append(float(true_branch == pred_branch))

    if not true_branches:
        return {"n_clones_evaluated": 0}

    true_branches_arr = np.array(true_branches)
    pred_branches_arr = np.array(pred_branches)

    accuracy = float(np.mean(true_branches_arr == pred_branches_arr))
    if _HAS_SKLEARN:
        balanced_acc = float(balanced_accuracy_score(true_branches_arr, pred_branches_arr))
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            true_branches_arr, pred_branches_arr, labels=branch_names, average="macro", zero_division=0.0
        )
        conf = confusion_matrix(true_branches_arr, pred_branches_arr, labels=branch_names)
    else:
        balanced_acc = accuracy
        macro_p = macro_r = macro_f1 = accuracy
        conf = None

    return {
        "source_label": source_label,
        "terminal_label": terminal_label,
        "n_clones_evaluated": int(len(true_branches)),
        "branch_accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "clone_level_consistency": float(np.mean(clone_consistencies)) if clone_consistencies else 0.0,
        "branch_names": branch_names,
        "confusion_matrix": conf.tolist() if conf is not None else None,
    }


def evaluate_pseudotime_correlation(
    flow_net: torch.nn.Module,
    frames: dict,
    labels: list,
    source_label,
    target_labels: list | None = None,
    n_ode_steps: int = 101,
    device: str = "cpu",
) -> dict[str, Any]:
    """Correlate predicted ODE time with true label time across rolled trajectories.

    We build continuous trajectories from ``source_label`` to every target label,
    collect all intermediate points with their ODE times, and for each empirical
    cell in each target frame find the nearest trajectory point.  Its ODE time is
    the predicted pseudotime for that cell.  A good model yields high
    Spearman/Pearson correlation and low MAE between predicted and true times.
    """
    from src.methods.pace.stage2.flow_matcher import labels_to_timesteps

    ordered_labels = sorted(labels)
    label_to_t = {label: labels_to_timesteps(ordered_labels)[i] for i, label in enumerate(ordered_labels)}
    targets = list(ordered_labels if target_labels is None else target_labels)

    if source_label not in frames:
        return {"error": "source label missing"}

    trajectory_points: list[np.ndarray] = []
    trajectory_times: list[np.ndarray] = []
    for target_label in targets:
        if target_label == source_label:
            continue
        if target_label not in frames:
            continue
        traj = _run_ode_rollout(
            flow_net=flow_net,
            source=frames[source_label],
            t_source=label_to_t[source_label],
            t_target=label_to_t[target_label],
            n_steps=n_ode_steps,
            device=device,
        )
        traj_np = traj.detach().cpu().numpy()
        times = np.linspace(label_to_t[source_label], label_to_t[target_label], n_ode_steps)
        trajectory_points.append(traj_np.reshape(-1, traj_np.shape[-1]))
        trajectory_times.append(np.tile(times, traj_np.shape[1]))

    if not trajectory_points:
        return {"error": "no target rollouts"}

    all_points = np.concatenate(trajectory_points, axis=0)
    all_times = np.concatenate(trajectory_times, axis=0)
    tree = cKDTree(all_points)

    pred_times = []
    true_times = []
    for target_label in targets:
        if target_label not in frames or target_label == source_label:
            continue
        gt = frames[target_label].detach().float().cpu().numpy()
        _, idx = tree.query(gt, k=1)
        pred_times.append(all_times[idx])
        true_times.append(np.full(gt.shape[0], label_to_t[target_label]))

    pred_times_arr = np.concatenate(pred_times)
    true_times_arr = np.concatenate(true_times)

    mae = float(np.mean(np.abs(pred_times_arr - true_times_arr)))
    if _HAS_SKLEARN:
        rho_s, _ = spearmanr(pred_times_arr, true_times_arr)
        rho_p, _ = pearsonr(pred_times_arr, true_times_arr)
    else:
        rho_s = rho_p = None

    return {
        "source_label": source_label,
        "target_labels": targets,
        "n_cells": int(len(pred_times_arr)),
        "mae": mae,
        "spearman": float(rho_s) if rho_s is not None else None,
        "pearson": float(rho_p) if rho_p is not None else None,
    }


def evaluate_cell_fate_error(
    flow_net: torch.nn.Module,
    train_frames: dict,
    test_frames: dict,
    train_labels: list,
    test_labels: list,
    fate_frames: dict,
    k: int = 10,
    n_ode_steps: int = 101,
    device: str = "cpu",
) -> list[dict[str, float | int | str]]:
    """Alias for ``evaluate_stage2_label_probability_over_time`` with fate labels.

    Reports per-cell fate error and marginal divergence metrics, suitable for the
    "cell-fate error" bullet in the rebuttal protocol.
    """
    return evaluate_stage2_label_probability_over_time(
        flow_net=flow_net,
        train_frames=train_frames,
        test_frames=test_frames,
        train_labels=train_labels,
        test_labels=test_labels,
        label_frames=fate_frames,
        label_name="fate",
        k=k,
        n_ode_steps=n_ode_steps,
        device=device,
    )


# ---------------------------------------------------------------------------
# Table writers
# ---------------------------------------------------------------------------


def write_distribution_metrics_table(
    rows: list[dict[str, float | int | str]],
    csv_path: str,
) -> None:
    """Write Stage 1 / Stage 2 distribution metrics into a standalone CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "stage",
        "test_label",
        "predictor",
        "prev_train_label",
        "next_train_label",
        "mmd",
        "w1",
        "w2",
        "gw",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_velocity_metrics_table(
    rows: list[dict[str, float | int | str]],
    csv_path: str,
) -> None:
    """Write Table-4-style velocity and distribution metrics into a CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["stage", "test_label", "predictor", "cos_dist", "l2", "w2"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
