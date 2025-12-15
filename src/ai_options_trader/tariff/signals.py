from __future__ import annotations

import pandas as pd
from ai_options_trader.tariff.models import TariffRegimeState, TariffInputs
from ai_options_trader.tariff.theory import TariffRegimeSpec, DEFAULT_TARIFF_SPEC
from ai_options_trader.tariff.transforms import (
    zscore,
    returns,
    rel_returns,
    cost_momentum,
    rolling_beta,
)


def build_tariff_regime_state(
    cost_df: pd.DataFrame | None = None,
    equity_prices: pd.DataFrame | None = None,
    cost_proxies: pd.DataFrame | None = None,
    equity_prices_df: pd.DataFrame | None = None,
    universe: list[str] | None = None,
    benchmark: str = "XLY",
    basket_name: str = "import_retail_apparel",
    start_date: str = "2016-01-01",
    spec: TariffRegimeSpec = DEFAULT_TARIFF_SPEC,
) -> TariffRegimeState:
    """
    Build tariff regime state.

    Backward compatible parameter aliases:
    - cost_df or cost_proxies
    - equity_prices or equity_prices_df
    """
    if cost_df is None:
        cost_df = cost_proxies
    if equity_prices is None:
        equity_prices = equity_prices_df
    if cost_df is None or equity_prices is None:
        raise ValueError("Must provide cost_df and equity_prices (or aliases).")
    if universe is None:
        raise ValueError("Must provide universe tickers list.")

    # --- Data ---
    cost_df = cost_df.sort_index()
    px = equity_prices.sort_index()

    # Choose first available cost proxy for v0; later we’ll use a weighted basket
    cost_proxy = cost_df.iloc[:, 0].rename("cost_proxy")

    # --- Cost pressure momentum (CPM) ---
    cpm = cost_momentum(cost_proxy, short=spec.cost_short_days, long=spec.cost_long_days)
    z_cpm = zscore(cpm, window=spec.z_window_days)

    # --- Equity price denial (EPD) ---
    r = px.apply(returns).dropna(how="all")
    bench_ret = r[benchmark].rename("bench")

    # Basket-level: equal-weight average rel return
    rels = []
    for sym in universe:
        if sym not in r.columns:
            continue
        rels.append(rel_returns(r[sym], bench_ret).rename(sym))
    rel_mat = pd.concat(rels, axis=1).dropna(how="all")
    basket_rel = rel_mat.mean(axis=1).rename("basket_rel")

    # Denial beta: basket relative returns vs cost momentum (if beta ~ 0 or positive while costs rise -> denial)
    aligned = pd.concat([basket_rel, cpm], axis=1).dropna()
    beta_cost = rolling_beta(aligned["basket_rel"], aligned["cost_mom"], window=spec.denial_window_days)

    # --- Earnings fragility (EF) placeholder ---
    # We keep the slot; will be populated once we select an estimates/revisions dataset.
    z_ef = pd.Series(index=beta_cost.index, data=0.0, name="z_earnings_fragility")

    # --- Composite score ---
    # Denial component uses negative sign: if beta_cost is not negative while z_cpm high, that indicates denial.
    # We standardize beta_cost to compare scale.
    z_beta_cost = zscore(beta_cost, window=spec.z_window_days)

    score = (
        spec.w_cost * z_cpm.reindex(z_beta_cost.index)
        + spec.w_denial * (-z_beta_cost)
        + spec.w_fragility * z_ef
    ).rename("tariff_regime_score")

    score_nonnull = score.dropna()
    if score_nonnull.empty:
        raise ValueError(
            "No valid tariff_regime_score computed (empty after dropna). "
            "Check cost_df/equity_prices coverage and missing data handling."
        )
    latest_idx = score_nonnull.index[-1]
    latest_val = float(score_nonnull.iloc[-1])
    asof = str(pd.to_datetime(latest_idx).date())

    inputs = TariffInputs(
        z_cost_pressure=float(z_cpm.reindex(score.index).iloc[-1]) if pd.notna(z_cpm.reindex(score.index).iloc[-1]) else None,
        equity_denial_beta=float(beta_cost.reindex(score.index).iloc[-1]) if pd.notna(beta_cost.reindex(score.index).iloc[-1]) else None,
        z_earnings_fragility=float(z_ef.reindex(score.index).iloc[-1]) if pd.notna(z_ef.reindex(score.index).iloc[-1]) else None,
        tariff_regime_score=latest_val,
        is_tariff_regime=bool(latest_val > spec.threshold),
        components={
            "z_cpm": float(z_cpm.reindex(score.index).iloc[-1]) if pd.notna(z_cpm.reindex(score.index).iloc[-1]) else None,
            "beta_cost": float(beta_cost.reindex(score.index).iloc[-1]) if pd.notna(beta_cost.reindex(score.index).iloc[-1]) else None,
            "z_beta_cost": float(z_beta_cost.reindex(score.index).iloc[-1]) if pd.notna(z_beta_cost.reindex(score.index).iloc[-1]) else None,
        },
    )

    return TariffRegimeState(
        asof=asof,
        start_date=start_date,
        universe=universe,
        inputs=inputs,
        notes="Tariff regime score uses cost pressure momentum + equity price denial; earnings fragility placeholder (0.0) until wired.",
    )
