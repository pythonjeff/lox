from __future__ import annotations

import pandas as pd
import numpy as np


def build_fci_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simplified "financial conditions" feature set from the full regime matrix.

    IMPORTANT (no leakage):
    - This function does NOT fit global scalers.
    - It only combines existing regime features that are already computed with rolling context
      (z-scores / scores computed historically).

    Outputs columns:
    - fci_rates: rates shock / duration pressure proxy
    - fci_usd: USD strength proxy
    - fci_funding: funding tightness proxy
    - fci_vol: volatility pressure proxy
    - fci_score: weighted composite (higher = tighter conditions)
    """
    if X is None or X.empty:
        return pd.DataFrame(index=getattr(X, "index", None))

    # Components (best-effort)
    rates = None
    if "rates_z_ust_10y_chg_20d" in X.columns:
        rates = pd.to_numeric(X["rates_z_ust_10y_chg_20d"], errors="coerce")
    elif "rates_z_ust_10y" in X.columns:
        rates = pd.to_numeric(X["rates_z_ust_10y"], errors="coerce")

    usd = pd.to_numeric(X["usd_strength_score"], errors="coerce") if "usd_strength_score" in X.columns else None
    funding = pd.to_numeric(X["funding_tightness_score"], errors="coerce") if "funding_tightness_score" in X.columns else None
    vol = pd.to_numeric(X["vol_pressure_score"], errors="coerce") if "vol_pressure_score" in X.columns else None

    out = pd.DataFrame(index=X.index)
    out["fci_rates"] = rates
    out["fci_usd"] = usd
    out["fci_funding"] = funding
    out["fci_vol"] = vol

    # Drop components that are entirely missing. This prevents downstream `dropna(how="any")`
    # from deleting the whole dataset just because one component is unavailable.
    for c in ["fci_rates", "fci_usd", "fci_funding", "fci_vol"]:
        if c in out.columns and not out[c].notna().any():
            out = out.drop(columns=[c])

    # Composite (simple, stable weights)
    # Higher = tighter financial conditions (equity headwind).
    w = {"fci_rates": 0.35, "fci_usd": 0.25, "fci_funding": 0.25, "fci_vol": 0.15}
    num = pd.Series(0.0, index=out.index, dtype=float)
    den = pd.Series(0.0, index=out.index, dtype=float)
    for k, wk in w.items():
        if k not in out.columns:
            continue
        s = pd.to_numeric(out[k], errors="coerce")
        ok = s.notna()
        num = num + (s.where(ok, 0.0) * float(wk))
        den = den + (ok.astype(float) * float(wk))

    # Use np.nan (not pd.NA) to keep dtype numeric and avoid NAType casting issues.
    out["fci_score"] = num / den.replace(0.0, np.nan)

    # Keep only stable columns (components + score) and drop rows where score is missing.
    # Note: downstream code may still drop NaNs; returning fewer columns helps.
    return out


