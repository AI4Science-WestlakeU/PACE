"""Metric Flow Matching package.

Imports are resolved lazily so modules that only reuse lightweight helpers
such as ``mfm.eval_metrics`` do not require optional MFM training dependencies.
"""

from importlib import import_module

__all__ = [
    "GeoPathMLP",
    "VelocityNet",
    "FlowModelWrapper",
    "EMA",
    "MetricFlowMatcher",
    "DataManifoldMetric",
    "GeoPathNetTrain",
    "FlowNetTrain",
    "MFMDataModuleWrapper",
    "run_mfm",
]

_LAZY_IMPORTS = {
    "GeoPathMLP": ("src.methods.mfm.networks", "GeoPathMLP"),
    "VelocityNet": ("src.methods.mfm.networks", "VelocityNet"),
    "FlowModelWrapper": ("src.methods.mfm.networks", "FlowModelWrapper"),
    "EMA": ("src.methods.mfm.ema", "EMA"),
    "MetricFlowMatcher": ("src.methods.mfm.flow_matcher", "MetricFlowMatcher"),
    "DataManifoldMetric": ("src.methods.mfm.geo_metrics", "DataManifoldMetric"),
    "GeoPathNetTrain": ("src.methods.mfm.geopath_train", "GeoPathNetTrain"),
    "FlowNetTrain": ("src.methods.mfm.flow_train", "FlowNetTrain"),
    "MFMDataModuleWrapper": ("src.methods.mfm.data_wrapper", "MFMDataModuleWrapper"),
    "run_mfm": ("src.methods.mfm.runner", "run_mfm"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
