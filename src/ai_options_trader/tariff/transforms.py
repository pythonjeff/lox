from __future__ import annotations

import pandas as pd
import numpy as np


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    return (series - mu) / sd


def returns(px: pd.Series) -> pd.Series:
    return px.pct_change()


def rel_returns(stock_ret: pd.Series, bench_ret: pd.Series) -> pd.Series:
    return (stock_ret - bench_ret).rename("rel_ret")


def cost_momentum(cost_proxy: pd.Series, short: int, long: int) -> pd.Series:
    """
    Momentum = short-horizon change - long-horizon change
    Both horizons in observation units (days after resampling/alignment).
    """
    s = cost_proxy.diff(short)
    l = cost_proxy.diff(long)
    return (s - l).rename("cost_mom")


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """
    Rolling OLS beta of y on x (with intercept removed via cov/var).
    """
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return (cov / var).rename("beta")
