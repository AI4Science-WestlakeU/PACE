"""Visualization utilities for PACE experiments."""

from importlib import import_module

__all__ = [
    "plot_interpolation_paths",
    "plot_midpoint_comparison",
    "plot_psi_magnitude",
    "plot_trajectory_overview",
    "plot_full_trajectory_predictions",
    "plot_ode_trajectory_overview",
    "plot_rollout_prediction_snapshots",
    "plot_rollout_trajectory_overview",
    "plot_test_predictions",
]

_LAZY_IMPORTS = {
    "plot_interpolation_paths": ("src.plot.stage1_visualization", "plot_interpolation_paths"),
    "plot_midpoint_comparison": ("src.plot.stage1_visualization", "plot_midpoint_comparison"),
    "plot_psi_magnitude": ("src.plot.stage1_visualization", "plot_psi_magnitude"),
    "plot_trajectory_overview": ("src.plot.stage1_visualization", "plot_trajectory_overview"),
    "plot_full_trajectory_predictions": ("src.plot.stage2_visualization", "plot_full_trajectory_predictions"),
    "plot_ode_trajectory_overview": ("src.plot.stage2_visualization", "plot_ode_trajectory_overview"),
    "plot_rollout_prediction_snapshots": ("src.plot.stage2_visualization", "plot_rollout_prediction_snapshots"),
    "plot_rollout_trajectory_overview": ("src.plot.stage2_visualization", "plot_rollout_trajectory_overview"),
    "plot_test_predictions": ("src.plot.stage2_visualization", "plot_test_predictions"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
