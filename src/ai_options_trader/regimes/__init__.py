"""Uniform regime feature vectors (ML-friendly).

This package provides a small schema + helpers to turn heterogeneous regime
calculations (macro, tariff, liquidity, etc.) into a single flat mapping of
scalar features: dict[str, float].
"""

from .schema import RegimeVector, merge_feature_dicts  # noqa: F401


