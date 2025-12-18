from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ai_options_trader.config import Settings
from ai_options_trader.ideas.models import Idea
from ai_options_trader.macro.regime import classify_macro_regime
from ai_options_trader.macro.signals import build_macro_dataset, build_macro_state
from ai_options_trader.macro.equity import delta as series_delta
from ai_options_trader.macro.equity import latest_sensitivity_table, returns as price_returns
from ai_options_trader.tariff.proxies import DEFAULT_COST_PROXY_SERIES
from ai_options_trader.tariff.signals import build_tariff_regime_state
from ai_options_trader.tariff.universe import BASKETS
from ai_options_trader.data.fred import FredClient
from ai_options_trader.data.market import fetch_equity_daily_closes


DEFAULT_AI_TECH_TICKERS = [
    # Mega-cap / “AI infrastructure” proxies
    "NVDA",
    "AMD",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    # High-beta / narrative-driven names
    "TSLA",
    "PLTR",
    "SMCI",
]


def _rank01(values: pd.Series) -> pd.Series:
    """Rank-normalize to [0,1]."""
    s = values.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return values * 0.0
    r = s.rank(pct=True)
    out = pd.Series(index=values.index, dtype=float)
    out.loc[r.index] = r.values
    return out.fillna(0.0)


def build_ai_bubble_ideas(
    *,
    settings: Settings,
    start_date: str = "2016-01-01",
    refresh: bool = False,
    macro_window: int = 252,
    macro_benchmark: str = "QQQ",
    tariff_benchmark: str = "XLY",
    tariff_baskets: List[str] | None = None,
    tech_tickers: List[str] | None = None,
) -> Tuple[Dict[str, Any], List[Idea]]:
    """
    Turn the thesis into ranked ideas:
    - AI bubble exists → focus on high-beta AI narrative names (starter universe)
    - Inflation underpriced in tech → prioritize names with negative real-yield beta
    - Tariffs underpriced → include tickers from import-exposed baskets when tariff regime is on

    Returns: (context, ideas)
    """
    tech = [t.upper() for t in (tech_tickers or DEFAULT_AI_TECH_TICKERS)]
    baskets = tariff_baskets or list(BASKETS.keys())
    baskets = [b for b in baskets if b in BASKETS]

    # --- Macro regime ---
    macro_state = build_macro_state(settings=settings, start_date=start_date, refresh=refresh)
    macro_regime = classify_macro_regime(
        inflation_momentum_minus_be=macro_state.inputs.inflation_momentum_minus_be5y,
        real_yield=macro_state.inputs.real_yield_proxy_10y,
    )

    # Signals we already compute
    z_infl = None
    if isinstance(macro_state.inputs.components, dict):
        z_infl = macro_state.inputs.components.get("z_infl_mom_minus_be5y")

    # --- Macro sensitivity (tech) ---
    # Build macro daily series
    m = build_macro_dataset(settings=settings, start_date=start_date, refresh=refresh).set_index("date")
    d_real = series_delta(m["REAL_YIELD_PROXY_10Y"]).rename("d_real")
    d_10y = series_delta(m["DGS10"]).rename("d_10y")
    d_be5 = series_delta(m["T5YIE"]).rename("d_be5")

    # Equity closes for tickers + benchmark
    syms = sorted(set(tech + [macro_benchmark.strip().upper()]))
    px = fetch_equity_daily_closes(
        api_key=settings.ALPACA_DATA_KEY or settings.ALPACA_API_KEY,
        api_secret=settings.ALPACA_DATA_SECRET or settings.ALPACA_API_SECRET,
        symbols=syms,
        start=start_date,
    )
    rets = price_returns(px)
    sens = latest_sensitivity_table(
        rets=rets,
        d_real=d_real,
        d_10y=d_10y,
        d_be5y=d_be5,
        window=macro_window,
    )

    # Score: more negative beta_d_real → higher score (more “rates/inflation sensitive” tech)
    beta_real = sens["beta_d_real"] if "beta_d_real" in sens.columns else pd.Series(dtype=float)
    tech_score = _rank01(-beta_real).rename("tech_score")
    tech_beta_real = beta_real.copy()

    # --- Tariff regimes (baskets) ---
    if not settings.FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY in environment / .env")
    fred = FredClient(api_key=settings.FRED_API_KEY)

    frames = []
    for col, sid in DEFAULT_COST_PROXY_SERIES.items():
        df = fred.fetch_series(sid, start_date=start_date, refresh=refresh)
        df = df.rename(columns={"value": col}).set_index("date")
        frames.append(df[[col]])
    cost_df = pd.concat(frames, axis=1).sort_index().resample("D").ffill()

    all_tariff_syms = sorted({sym for b in baskets for sym in BASKETS[b].tickers})
    px_tariff = fetch_equity_daily_closes(
        api_key=settings.ALPACA_DATA_KEY or settings.ALPACA_API_KEY,
        api_secret=settings.ALPACA_DATA_SECRET or settings.ALPACA_API_SECRET,
        symbols=sorted(set(all_tariff_syms + [tariff_benchmark.strip().upper()])),
        start=start_date,
    ).sort_index().ffill().dropna(how="all")

    tariff_states: Dict[str, Any] = {}
    for b in baskets:
        basket = BASKETS[b]
        tariff_states[b] = build_tariff_regime_state(
            cost_df=cost_df,
            equity_prices=px_tariff,
            universe=basket.tickers,
            benchmark=tariff_benchmark,
            basket_name=b,
            start_date=start_date,
        )

    # --- Build idea list ---
    ideas_by_ticker: Dict[str, Idea] = {}

    # 1) Tech / AI bubble + inflation underpriced
    for t in tech:
        s = float(tech_score.get(t, 0.0)) if len(tech_score) else 0.0
        br = float(tech_beta_real.get(t)) if t in tech_beta_real.index and pd.notna(tech_beta_real.get(t)) else None
        # amplify when inflation momentum vs breakevens is positive (more likely “underpriced inflation”)
        infl_boost = 0.15 if (z_infl is not None and float(z_infl) > 0) else 0.0
        score = 0.60 * s + infl_boost
        ideas_by_ticker[t] = Idea(
            ticker=t,
            direction="bearish",
            score=float(score),
            tags=["ai_bubble", "inflation_underpriced_in_tech"],
            thesis="AI bubble + inflation risk underpriced in tech multiples",
            rationale=(
                "High-duration / narrative tech tends to underperform when real yields rise; "
                "ranked by negative real-yield beta using recent rolling sensitivity."
            ),
            why={
                "tech_sensitivity": {
                    "beta_d_real": br,
                    "rank_pct": s,
                    "macro_window": macro_window,
                    "benchmark": macro_benchmark,
                },
                "macro": {
                    "regime": macro_regime.name,
                    "infl_trend": macro_regime.inflation_trend,
                    "real_yield_trend": macro_regime.real_yield_trend,
                    "z_infl_mom_minus_be5y": z_infl,
                },
                "score_components": {"tech_component": 0.60 * s, "infl_boost": infl_boost},
            },
        )

    # 2) Tariff underpriced: add import-exposed basket tickers, weighted by tariff regime score
    for b, state in tariff_states.items():
        is_on = bool(state.inputs.is_tariff_regime)
        base = 0.55 if is_on else 0.25
        # Normalize within basket using the latest score (can be negative); just use regime flag for now
        for sym in BASKETS[b].tickers:
            t = sym.upper()
            add = base
            if t in ideas_by_ticker:
                # merge
                merged = ideas_by_ticker[t]
                merged.tags = sorted(set(merged.tags + ["tariff_underpriced"]))
                merged.score = float(merged.score + add)
                merged.thesis = (merged.thesis + " + tariffs underpriced").strip()
                merged.rationale = merged.rationale + " Also appears in import-exposed tariff basket."
                merged.why = dict(merged.why or {})
                merged.why.setdefault("tariff", {})
                merged.why["tariff"].setdefault("baskets", [])
                merged.why["tariff"]["baskets"].append(
                    {
                        "basket": b,
                        "is_tariff_regime": bool(state.inputs.is_tariff_regime),
                        "tariff_regime_score": state.inputs.tariff_regime_score,
                        "benchmark": state.benchmark,
                    }
                )
                merged.why.setdefault("score_components", {})
                merged.why["score_components"]["tariff_boost"] = merged.why["score_components"].get(
                    "tariff_boost", 0.0
                ) + add
                ideas_by_ticker[t] = merged
            else:
                ideas_by_ticker[t] = Idea(
                    ticker=t,
                    direction="bearish",
                    score=float(add),
                    tags=["tariff_underpriced"],
                    thesis="Tariffs / import-cost shock underpriced",
                    rationale=f"Member of import-exposed basket '{b}' ({BASKETS[b].description}).",
                    why={
                        "tariff": {
                            "baskets": [
                                {
                                    "basket": b,
                                    "is_tariff_regime": bool(state.inputs.is_tariff_regime),
                                    "tariff_regime_score": state.inputs.tariff_regime_score,
                                    "benchmark": state.benchmark,
                                }
                            ]
                        },
                        "score_components": {"tariff_boost": add},
                    },
                )

    ideas = sorted(ideas_by_ticker.values(), key=lambda x: x.score, reverse=True)

    context: Dict[str, Any] = {
        "macro_state": macro_state,
        "macro_regime": macro_regime,
        "tech_sensitivity": sens,
        "tariff_states": tariff_states,
        "params": {
            "start_date": start_date,
            "macro_window": macro_window,
            "macro_benchmark": macro_benchmark,
            "tariff_benchmark": tariff_benchmark,
            "tariff_baskets": baskets,
            "tech_tickers": tech,
        },
    }
    return context, ideas


