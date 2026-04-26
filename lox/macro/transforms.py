from __future__ import annotations

import pandas as pd
import numpy as np


def to_daily_index(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """Create a daily date index starting at start_date through df max date."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(df["date"].max())
    idx = pd.date_range(start=start, end=end, freq="D")
    return pd.DataFrame({"date": idx})


def merge_series_daily(daily_index: pd.DataFrame, series_map: dict[str, pd.DataFrame], ffill: bool = True) -> pd.DataFrame:
    out = daily_index.copy()
    for sid, sdf in series_map.items():
        tmp = sdf.copy()
        tmp = tmp.rename(columns={"value": sid})
        before_cols = set(out.columns)
        out = out.merge(tmp, on="date", how="left")
        if ffill:
            # Forward-fill all newly merged columns (not just the primary series),
            # so derived monthly metrics (e.g., CPI_YOY) remain valid on the daily grid.
            new_cols = [c for c in out.columns if c not in before_cols and c != "date"]
            for c in new_cols:
                out[c] = out[c].ffill()
    return out


def yoy_from_index_level(level: pd.Series) -> pd.Series:
    """YoY percent change for index levels like CPI.

    Note: In this project we forward-fill monthly CPI onto a daily grid, so the
    series is effectively daily. Using pct_change(12) would mean "12 days".
    We therefore compute YoY as ~365 days.
    """
    return level.pct_change(365) * 100.0


def annualized_rate_from_levels(level: pd.Series, months: int) -> pd.Series:
    """
    Annualized inflation rate from index levels over 'months'.
    Example: 6-month annualized = ((CPI_t / CPI_{t-6})^(12/6) - 1) * 100
    """
    # Same note as yoy_from_index_level: after forward-fill onto a daily grid,
    # treat "months" as an approximate number of days.
    days = int(round(365.0 / 12.0 * months))
    ratio = level / level.shift(days)
    ann = (ratio ** (365.0 / days) - 1.0) * 100.0
    return ann


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    return (series - mu) / sd


def seasonal_zscore(
    series: pd.Series,
    min_years: int = 3,
    bin_days: int = 21,
) -> pd.Series:
    """Z-score adjusted for calendar seasonality.

    Bins dates into ~17 seasonal windows (each *bin_days* wide) and computes
    z-scores using only data from prior calendar years in the same bin.
    Strips patterns like spring planting premium and fall harvest pressure
    so the signal reflects *unusual* moves for that time of year.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return pd.Series(np.nan, index=series.index)

    result = pd.Series(np.nan, index=series.index)
    s = series.dropna()
    if s.empty:
        return result

    doy = s.index.dayofyear.values
    years = s.index.year.values
    vals = s.values
    bins = ((doy - 1) // bin_days).astype(int)

    unique_years = np.unique(years)
    if len(unique_years) < min_years + 1:
        return result

    for b in np.unique(bins):
        bin_mask = bins == b
        bin_years = years[bin_mask]
        bin_vals = vals[bin_mask]
        bin_idx = s.index[bin_mask]

        for yr in unique_years[min_years:]:
            prior_vals = bin_vals[bin_years < yr]
            prior_vals = prior_vals[~np.isnan(prior_vals)]
            if len(prior_vals) < min_years:
                continue

            mu = np.mean(prior_vals)
            sd = np.std(prior_vals, ddof=0)
            if sd < 1e-10:
                continue

            yr_mask = bin_years == yr
            result.loc[bin_idx[yr_mask]] = (bin_vals[yr_mask] - mu) / sd

    return result
