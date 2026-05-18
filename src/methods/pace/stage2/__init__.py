"""PACE Stage 2 building blocks.

This package vendors the small set of velocity-field training helpers used by
the original iMFM Stage 2 path, without shipping another standalone method.
"""

from importlib import import_module

__all__ = [
    "GeoPathMLP",
    "VelocityNet",
    "FlowModelWrapper",
    "EMA",
    "MetricFlowMatcher",
    "FlowNetTrain",
    "PaceDataModuleWrapper",
    "build_ot_sampler",
]

_LAZY_IMPORTS = {
    "GeoPathMLP": ("src.methods.pace.stage2.networks", "GeoPathMLP"),
    "VelocityNet": ("src.methods.pace.stage2.networks", "VelocityNet"),
    "FlowModelWrapper": ("src.methods.pace.stage2.networks", "FlowModelWrapper"),
    "EMA": ("src.methods.pace.stage2.ema", "EMA"),
    "MetricFlowMatcher": ("src.methods.pace.stage2.flow_matcher", "MetricFlowMatcher"),
    "FlowNetTrain": ("src.methods.pace.stage2.flow_train", "FlowNetTrain"),
    "PaceDataModuleWrapper": ("src.methods.pace.stage2.data_wrapper", "PaceDataModuleWrapper"),
    "build_ot_sampler": ("src.methods.pace.stage2.matching", "build_ot_sampler"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
