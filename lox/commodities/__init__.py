from __future__ import annotations

"""
Commodities regime (MVP).

Goal:
Capture commodity inflation / reflation dynamics (especially energy shocks) in a stable,
ML-friendly way to support "fiscal dominance / inflation risk" style macro theses.

FRED-first inputs (best-effort):
- WTI spot oil (DCOILWTICO) [daily]
- Gold (LBMA, USD) (GOLDAMGBD228NLBM) [daily]
- Broad commodity price index (IMF, all commodities) (PALLFNFINDEXQ) [monthly/quarterly; ffilled on daily grid]
"""


