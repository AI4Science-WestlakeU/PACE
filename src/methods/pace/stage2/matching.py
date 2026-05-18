"""Stage 2 mini-batch matching helpers for PACE.

This is the small OT pairing hook used by the original Stage 2 flow training:
when configured, each adjacent pair of timepoint batches is matched by
``torchcfm`` before conditional-flow targets are sampled.
"""

from __future__ import annotations

from omegaconf import DictConfig


def build_ot_sampler(cfg: DictConfig):
    """Build the Stage 2 OT sampler from ``optimal_transport_method``.

    ``None`` disables mini-batch OT matching. Other values are forwarded to
    ``torchcfm.optimal_transport.OTPlanSampler``; for the EB PHATE PACE config
    the default is ``exact``.
    """
    method = cfg.get("optimal_transport_method", "None")
    if method is None or str(method) == "None":
        return None

    from torchcfm.optimal_transport import OTPlanSampler

    return OTPlanSampler(method=str(method))