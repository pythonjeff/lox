"""
Backtest engine for congress / government-insider trade signals.

Usage:
    engine = BacktestEngine(settings, horizons=[5, 10, 20, 30])
    results = engine.run(signals)
    summary = engine.summarize(results)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import pandas as pd
from pandas.tseries.offsets import BDay


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Signal:
    ticker: str
    signal_date: date
    side: str                       # "Buy" | "Sell"
    cluster_count: int = 1
    lag_days: int = 30
    amount_usd: float = 0.0
    source: str = ""
    regime_aligned: bool = False    # regime supports the direction
    insider_confirmed: bool = False
    committee_aligned: bool = False # trader sits on committee with oversight of this sector
    committee_note: str = ""        # e.g. "Armed Services — defense contractor oversight"
    sector: str = ""                # GICS sector (populated by backtest pipeline)
    rsi_at_entry: Optional[float] = None  # RSI-14 at entry date (populated by engine.run)


@dataclass
class SignalResult:
    signal: Signal
    tradeable: bool                         # False if no price data
    entry_price: Optional[float]
    spy_entry: Optional[float]
    returns: dict[int, float]               # direction-adjusted % returns at each horizon (positive = win)
    spy_returns: dict[int, float]
    alpha: dict[int, float]                 # returns[h] - spy_returns[h]


@dataclass
class BacktestSummary:
    n_signals: int
    n_tradeable: int
    horizons: list[int]
    win_rates: dict[int, float]
    avg_returns: dict[int, float]
    avg_alpha: dict[int, float]
    t_stats: dict[int, float]               # t-stat for avg_return being != 0
    median_returns: dict[int, float]


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(self, settings, horizons: list[int] = None):
        self.settings = settings
        self.horizons = horizons if horizons is not None else [5, 10, 20, 30]

    def run(self, signals: list[Signal]) -> list[SignalResult]:
        """
        Fetch prices for all unique tickers + SPY, then for each signal:
        - entry = signal_date + 1 business day
        - exit at each horizon = entry + horizon business days
        - returns[h] = (exit_price / entry_price - 1) * 100  (for Buy)
        - returns[h] = (entry_price / exit_price - 1) * 100  (for Sell, flip)
        - alpha[h] = returns[h] - spy_returns[h]
        """
        if not signals:
            return []

        from lox.data.market import fetch_equity_daily_closes_fmp

        # Determine date range — extend 30 extra days before earliest signal for RSI history
        all_dates = [s.signal_date for s in signals if s.signal_date is not None]
        earliest = min(all_dates) - pd.Timedelta(days=35)
        start_str = str(earliest.isoformat() if hasattr(earliest, "isoformat") else earliest)

        unique_tickers = sorted({s.ticker for s in signals if s.ticker})
        all_symbols = unique_tickers + ["SPY"]

        # Fetch per-ticker so one bad symbol doesn't kill the whole run.
        frames: dict[str, pd.Series] = {}
        for sym in all_symbols:
            try:
                df = fetch_equity_daily_closes_fmp(
                    settings=self.settings,
                    symbols=[sym],
                    start=start_str,
                )
                if sym in df.columns:
                    frames[sym] = df[sym]
            except Exception:
                pass

        prices = pd.DataFrame(frames) if frames else pd.DataFrame()

        # Pre-compute RSI-14 at each signal's entry date and store on the Signal object.
        # Mutates signals in-place so filter predicates can use s.rsi_at_entry.
        for sig in signals:
            if sig.ticker in prices.columns:
                entry_ts = pd.Timestamp(sig.signal_date) + BDay(1)
                sig.rsi_at_entry = _rsi_at_ts(prices[sig.ticker], entry_ts)

        results: list[SignalResult] = []
        for sig in signals:
            result = self._evaluate_signal(sig, prices)
            results.append(result)

        return results

    def _evaluate_signal(self, sig: Signal, prices: pd.DataFrame) -> SignalResult:
        """Evaluate a single signal against the price DataFrame."""
        try:
            # Entry = signal_date + 1 business day
            entry_ts = pd.Timestamp(sig.signal_date) + BDay(1)

            returns: dict[int, float] = {}
            spy_returns: dict[int, float] = {}
            alpha: dict[int, float] = {}

            # Fetch entry price
            if sig.ticker not in prices.columns:
                return SignalResult(
                    signal=sig,
                    tradeable=False,
                    entry_price=None,
                    spy_entry=None,
                    returns={},
                    spy_returns={},
                    alpha={},
                )

            ticker_series = prices[sig.ticker]
            spy_series = prices["SPY"] if "SPY" in prices.columns else None

            entry_price = ticker_series.asof(entry_ts)
            spy_entry = spy_series.asof(entry_ts) if spy_series is not None else None

            if pd.isna(entry_price):
                return SignalResult(
                    signal=sig,
                    tradeable=False,
                    entry_price=None,
                    spy_entry=None,
                    returns={},
                    spy_returns={},
                    alpha={},
                )

            for h in self.horizons:
                exit_ts = entry_ts + BDay(h)

                exit_price = ticker_series.asof(exit_ts)
                spy_exit = spy_series.asof(exit_ts) if spy_series is not None else None

                if pd.isna(exit_price):
                    continue

                if sig.side == "Buy":
                    ret = (exit_price / entry_price - 1.0) * 100.0
                else:
                    ret = (entry_price / exit_price - 1.0) * 100.0

                returns[h] = round(ret, 4)

                if spy_series is not None and not pd.isna(spy_entry) and not pd.isna(spy_exit):
                    spy_ret = (spy_exit / spy_entry - 1.0) * 100.0
                    spy_returns[h] = round(spy_ret, 4)
                    alpha[h] = round(ret - spy_ret, 4)

            return SignalResult(
                signal=sig,
                tradeable=True,
                entry_price=float(entry_price),
                spy_entry=float(spy_entry) if spy_entry is not None and not pd.isna(spy_entry) else None,
                returns=returns,
                spy_returns=spy_returns,
                alpha=alpha,
            )

        except Exception:
            return SignalResult(
                signal=sig,
                tradeable=False,
                entry_price=None,
                spy_entry=None,
                returns={},
                spy_returns={},
                alpha={},
            )

    def summarize(self, results: list[SignalResult]) -> BacktestSummary:
        """Compute win rates, avg returns, alpha, t-stats across all tradeable results."""
        tradeable = [r for r in results if r.tradeable]

        win_rates: dict[int, float] = {}
        avg_returns: dict[int, float] = {}
        avg_alpha: dict[int, float] = {}
        t_stats: dict[int, float] = {}
        median_returns: dict[int, float] = {}

        for h in self.horizons:
            rets = [r.returns[h] for r in tradeable if h in r.returns]
            alphas = [r.alpha[h] for r in tradeable if h in r.alpha]

            if not rets:
                win_rates[h] = 0.0
                avg_returns[h] = 0.0
                avg_alpha[h] = 0.0
                t_stats[h] = 0.0
                median_returns[h] = 0.0
                continue

            wins = sum(1 for x in rets if x > 0)
            win_rates[h] = round(wins / len(rets) * 100.0, 1)
            avg_ret = sum(rets) / len(rets)
            avg_returns[h] = round(avg_ret, 4)
            avg_alpha[h] = round(sum(alphas) / len(alphas), 4) if alphas else 0.0

            # t-stat: mean / (std / sqrt(n))
            t_stats[h] = _tstat(rets)

            # Median
            sorted_rets = sorted(rets)
            n = len(sorted_rets)
            if n % 2 == 0:
                median_returns[h] = round((sorted_rets[n // 2 - 1] + sorted_rets[n // 2]) / 2.0, 4)
            else:
                median_returns[h] = round(sorted_rets[n // 2], 4)

        return BacktestSummary(
            n_signals=len(results),
            n_tradeable=len(tradeable),
            horizons=self.horizons,
            win_rates=win_rates,
            avg_returns=avg_returns,
            avg_alpha=avg_alpha,
            t_stats=t_stats,
            median_returns=median_returns,
        )


def _rsi_at_ts(series: pd.Series, ts: pd.Timestamp, period: int = 14) -> Optional[float]:
    """RSI-14 computed from the price series up to (and including) ts."""
    try:
        import numpy as np
        hist = series[series.index <= ts].dropna()
        if len(hist) < period + 1:
            return None
        closes = hist.values[-(period + 1):].astype(np.float64)
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss < 1e-12:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100.0 - (100.0 / (1.0 + rs)), 1)
    except Exception:
        return None


def _tstat(values: list[float]) -> float:
    """t-stat for mean != 0. Uses scipy if available, else manual."""
    n = len(values)
    if n < 2:
        return 0.0
    try:
        from scipy.stats import ttest_1samp
        stat, _ = ttest_1samp(values, 0)
        return round(float(stat), 3)
    except ImportError:
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        return round(mean / (std / math.sqrt(n)), 3)


# ── Compare filters ───────────────────────────────────────────────────────────

def compare_filters(
    all_signals: list[Signal],
    named_filters: list[tuple[str, Callable]],
    engine: BacktestEngine,
    horizon: int = 10,
) -> list[tuple[str, BacktestSummary]]:
    """
    Run the backtest for each named filter (a predicate function on Signal).
    Return list of (filter_name, summary) in order.
    The first filter should always be "Baseline (all buys)" with lambda s: True.
    """
    output: list[tuple[str, BacktestSummary]] = []
    for name, predicate in named_filters:
        subset = [s for s in all_signals if predicate(s)]
        results = engine.run(subset)
        summary = engine.summarize(results)
        output.append((name, summary))
    return output
