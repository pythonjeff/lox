"""
Volatility-tilted factor backtest.

Restricts the universe to its most-volatile trailing-vol slice at each
rebalance date (using only data available as of that date — no lookahead),
then tests whether momentum, mean-reversion (RSI), or vol-persistence
factors predict forward returns *within that slice*. Built to answer: among
the kind of name you'd actually buy options on, which factor (if any) has
real, tradable edge?

Known gap: lox has no historical single-name options/IV data anywhere
(only live chain snapshots via Alpaca/Polygon — see lox/data/alpaca.py,
lox/data/polygon.py). This backtest scores the UNDERLYING's realized
forward move — the thing that determines whether a long option would have
paid off — not actual option P&L. Premium cost, IV crush, and spread aren't
modeled. Treat results as a filter for which factor is worth building an
options picker around, not a backtested options track record.

Statistical caveat: forward-return windows for the same ticker across
consecutive rebalances can overlap, and names within one rebalance date
share market/sector moves — neither is fully independent, so t-stats here
are indicative, not textbook-clean. Default rebalance spacing equals the
longest horizon tested to cut down (not eliminate) the overlap.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lox.config import Settings
from lox.data.market import fetch_equity_daily_closes_fmp


def fetch_universe_closes(
    *,
    settings: Settings,
    symbols: list[str],
    start: str,
    max_workers: int = 4,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch daily closes for many symbols concurrently.

    Each call hits fetch_equity_daily_closes_fmp's existing per-symbol
    on-disk cache (data/cache/fmp_prices/), so reruns are fast — only a
    cold cache pays the network cost. fetch_equity_daily_closes_fmp itself
    has no rate-limit handling, so a cold multi-hundred-ticker pull WILL
    trip FMP's 429s; this wrapper retries with backoff instead of silently
    dropping the ticker (dropping matters a lot here — SPY failing silently
    used to zero out every alpha column with no error).
    """
    import time
    import requests

    out: dict[str, pd.Series] = {}
    failed: list[str] = []

    def _one(sym: str):
        delay = 1.5
        for attempt in range(max_retries):
            try:
                df = fetch_equity_daily_closes_fmp(settings=settings, symbols=[sym], start=start)
                return sym, df[sym] if sym in df.columns else None
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 429 and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return sym, None
            except Exception:
                return sym, None
        return sym, None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, s): s for s in symbols}
        for fut in as_completed(futures):
            sym, ser = fut.result()
            if ser is not None and not ser.empty:
                out[sym] = ser
            else:
                failed.append(sym)

    if failed:
        import sys
        print(f"[fetch_universe_closes] {len(failed)}/{len(symbols)} tickers failed after retries: "
              f"{failed[:10]}{'…' if len(failed) > 10 else ''}", file=sys.stderr)

    return pd.DataFrame(out).sort_index()


def realized_vol(closes: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Trailing annualized realized vol from daily log returns.

    Row T uses only data <= T (pandas .rolling is inherently trailing), so
    it's safe to use as a same-day ranking signal without lookahead.
    """
    log_ret = np.log(closes / closes.shift(1))
    return log_ret.rolling(window).std() * math.sqrt(252)


def momentum(closes: pd.DataFrame, window: int) -> pd.DataFrame:
    return closes / closes.shift(window) - 1.0


def rsi(closes: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


@dataclass
class FactorBucketResult:
    factor: str
    quintile: int  # 1 = lowest factor value, 5 = highest
    horizon: int
    n: int
    win_rate: float
    avg_return: float
    avg_alpha: float       # vs SPY over the same window
    avg_abs_move: float    # avg |forward return| — magnitude proxy for option payoff
    t_stat: float


_FACTORS = {
    "momentum_20d": lambda closes: momentum(closes, 20),
    "momentum_60d": lambda closes: momentum(closes, 60),
    "rsi_14": lambda closes: rsi(closes, 14),
    "vol_20d": lambda closes: realized_vol(closes, 20),  # does today's elevated vol persist/pay off?
}


def run_vol_factor_backtest(
    closes: pd.DataFrame,
    spy: pd.Series | None = None,
    *,
    vol_window: int = 20,
    vol_top_pct: float = 0.2,
    horizons: tuple[int, ...] = (5, 10, 20),
    min_price: float = 5.0,
    factors: tuple[str, ...] = ("momentum_20d", "momentum_60d", "rsi_14"),
) -> list[FactorBucketResult]:
    """
    Walk `closes` forward. At each rebalance date T (data <= T only):
      1. Rank all tickers by trailing realized vol; keep the top vol_top_pct.
      2. Within that high-vol slice, compute each requested factor.
      3. Bucket the factor into quintiles among the selected names.
      4. Enter at T + 1 business day, record forward return / |return| /
         alpha-vs-SPY at each horizon.
    Rebalances are spaced by max(horizons) trading days apart so the
    forward windows for a given horizon don't overlap in time.
    """
    rv = realized_vol(closes, vol_window)
    factor_frames = {f: _FACTORS[f](closes) for f in factors}

    # Align SPY to the universe's date index explicitly — spy is fetched as a
    # separate frame, so relying on matching positions instead of matching
    # dates would silently misalign entry/exit prices if either series is
    # missing even one trading day the other has.
    if spy is not None:
        spy = spy.reindex(closes.index)

    dates = closes.index
    min_lookback = max(vol_window, 60, 14) + 5
    step = max(horizons)
    rebalance_dates = dates[min_lookback::step]

    records: dict[str, list[tuple[int, int, float, float]]] = {f: [] for f in factors}

    for t in rebalance_dates:
        pos = dates.get_loc(t)
        if pos + max(horizons) + 1 >= len(dates):
            continue
        entry_pos = pos + 1

        vol_row = rv.iloc[pos]
        price_row = closes.iloc[pos]
        eligible = [s for s in vol_row.dropna().index if price_row.get(s, 0) >= min_price]
        if len(eligible) < 20:
            continue

        vol_sorted = vol_row[eligible].sort_values(ascending=False)
        cutoff = max(10, int(len(vol_sorted) * vol_top_pct))
        hi_vol_names = list(vol_sorted.index[:cutoff])

        entry_prices = closes.iloc[entry_pos]
        spy_entry = spy.iloc[entry_pos] if spy is not None and entry_pos < len(spy) else None

        for fname in factors:
            frow = factor_frames[fname].iloc[pos][hi_vol_names].dropna()
            if len(frow) < 10:
                continue
            try:
                quintiles = pd.qcut(frow, 5, labels=False, duplicates="drop") + 1
            except Exception:
                continue

            for h in horizons:
                exit_pos = entry_pos + h
                if exit_pos >= len(dates):
                    continue
                exit_prices = closes.iloc[exit_pos]
                spy_exit = spy.iloc[exit_pos] if spy is not None and exit_pos < len(spy) else None
                spy_ret = None
                if spy_entry is not None and spy_exit is not None and pd.notna(spy_entry) and pd.notna(spy_exit) and spy_entry > 0:
                    spy_ret = (spy_exit / spy_entry - 1.0) * 100.0

                for sym, q in quintiles.items():
                    ep = entry_prices.get(sym)
                    xp = exit_prices.get(sym)
                    if ep is None or xp is None or pd.isna(ep) or pd.isna(xp) or ep <= 0:
                        continue
                    fwd_ret = (xp / ep - 1.0) * 100.0
                    alpha = (fwd_ret - spy_ret) if spy_ret is not None else float("nan")
                    records[fname].append((int(q), h, fwd_ret, alpha))

    results: list[FactorBucketResult] = []
    for fname, rows in records.items():
        by_bucket: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for q, h, ret, alpha in rows:
            by_bucket.setdefault((q, h), []).append((ret, alpha))
        for (q, h), pairs in by_bucket.items():
            n = len(pairs)
            if n < 5:
                continue
            rets = [r for r, _ in pairs]
            alphas = [a for _, a in pairs if not math.isnan(a)]
            wins = sum(1 for r in rets if r > 0) / n * 100.0
            avg = sum(rets) / n
            avg_abs = sum(abs(r) for r in rets) / n
            avg_alpha = sum(alphas) / len(alphas) if alphas else float("nan")
            t = _tstat(rets)
            results.append(FactorBucketResult(
                fname, q, h, n, round(wins, 1), round(avg, 2),
                round(avg_alpha, 2) if alphas else float("nan"),
                round(avg_abs, 2), t,
            ))

    return sorted(results, key=lambda r: (r.factor, r.horizon, r.quintile))


def _tstat(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return round(mean / (std / math.sqrt(n)), 3)
