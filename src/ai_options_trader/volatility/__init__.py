from __future__ import annotations

"""
Volatility regime (MVP).

FRED-first inputs:
- VIX spot index (VIXCLS)
- Optional term structure anchors (VIX9D, VIX3M) when available

Goal:
Provide stable, ML-friendly features that flag:
- "vol is elevated" (risk-off / hedging bid)
- "vol is spiking" (shock)
- "term structure inversion" (backwardation-ish risk)
"""


