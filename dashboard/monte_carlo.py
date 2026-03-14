"""
Monte Carlo simulation engine for the LOX FUND Dashboard.

Real MC using lox v01 engine with regime-conditional assumptions,
plus simplified fallback for empty portfolios.
"""

import time
from datetime import datetime, timezone

from lox.config import load_settings

from dashboard.cache import (
    MC_CACHE, MC_CACHE_LOCK, MC_REFRESH_INTERVAL,
)
from dashboard.data_fetchers import get_vix, get_hy_oas
from dashboard.regime_utils import get_regime_label
from dashboard.positions import get_positions_data
from dashboard.portfolio import build_portfolio_from_alpaca


def run_monte_carlo_simulation(portfolio, vix_val, hy_val, regime_label, n_scenarios=2000):
    """Run actual Monte Carlo simulation using the v01 engine."""
    import numpy as np
    from lox.llm.scenarios.monte_carlo_v01 import MonteCarloV01, ScenarioAssumptions

    regime_map = {
        "RISK-ON": "GOLDILOCKS",
        "CAUTIOUS": "ALL",
        "RISK-OFF": "RISK_OFF",
        "UNKNOWN": "ALL",
    }
    mc_regime = regime_map.get(regime_label, "ALL")

    if vix_val:
        if vix_val > 30:
            mc_regime = "RISK_OFF"
        elif vix_val > 25:
            mc_regime = "CREDIT_STRESS" if (hy_val and hy_val > 400) else "RATES_SHOCK"
        elif vix_val < 14:
            mc_regime = "VOL_CRUSH"

    assumptions = ScenarioAssumptions.for_regime(mc_regime, horizon_months=6)
    mc = MonteCarloV01(portfolio, assumptions)
    results = mc.generate_scenarios(n_scenarios=n_scenarios)
    analysis = mc.analyze_results(results)

    cvar_attr = analysis.get("cvar_attribution", {})
    top_risk_driver = None
    if cvar_attr:
        sorted_attr = sorted(cvar_attr.items(), key=lambda x: x[1])
        if sorted_attr:
            top_risk_driver = sorted_attr[0][0]

    worst_scenarios = analysis.get("top_3_losers", [])
    worst_scenario_desc = None
    if worst_scenarios:
        worst = worst_scenarios[0]
        moves = worst.get("equity_moves", {})
        if moves:
            biggest_move = max(moves.items(), key=lambda x: abs(x[1]))
            worst_scenario_desc = f"{biggest_move[0]} {biggest_move[1]*100:+.0f}%"
            if worst.get("had_jump"):
                worst_scenario_desc += " (crash)"

    return {
        "mean_pnl_pct": round(analysis.get("mean_pnl_pct", 0), 4),
        "median_pnl_pct": round(analysis.get("median_pnl_pct", 0), 4),
        "var_95_pct": round(analysis.get("var_95_pct", 0), 4),
        "cvar_95_pct": round(analysis.get("cvar_95_pct", 0), 4),
        "prob_positive": round(analysis.get("prob_positive", 0.5), 3),
        "prob_loss_gt_10pct": round(analysis.get("prob_loss_gt_10pct", 0), 3),
        "prob_loss_gt_20pct": round(analysis.get("prob_loss_gt_20pct", 0), 3),
        "skewness": round(analysis.get("skewness", 0), 2),
        "max_loss_pct": round(analysis.get("max_loss_pct", 0), 4),
        "max_gain_pct": round(analysis.get("max_gain_pct", 0), 4),
        "top_risk_driver": top_risk_driver,
        "worst_scenario": worst_scenario_desc,
        "regime": regime_label,
        "mc_regime": mc_regime,
        "horizon_months": 6,
        "n_scenarios": n_scenarios,
    }


def calculate_monte_carlo_forecast(vix_val, hy_val, regime_label, positions_data=None, cash_available=0):
    """Calculate MC forecast — real simulation if positions available, else simplified."""
    if positions_data and len(positions_data) > 0:
        try:
            portfolio = build_portfolio_from_alpaca(positions_data, cash_available)
            if portfolio.positions:
                return run_monte_carlo_simulation(portfolio, vix_val, hy_val, regime_label)
        except Exception as e:
            print(f"[MC] Real simulation failed, using fallback: {e}")

    import numpy as np

    regime_params = {
        "RISK-ON": {"mean": 0.08, "vol": 0.12, "var95": -0.10, "prob_positive": 0.68},
        "CAUTIOUS": {"mean": 0.03, "vol": 0.18, "var95": -0.18, "prob_positive": 0.55},
        "RISK-OFF": {"mean": -0.02, "vol": 0.25, "var95": -0.30, "prob_positive": 0.42},
        "UNKNOWN": {"mean": 0.04, "vol": 0.15, "var95": -0.15, "prob_positive": 0.55},
    }
    params = regime_params.get(regime_label, regime_params["UNKNOWN"])

    if vix_val and vix_val > 20:
        vol_mult = 1 + (vix_val - 20) / 30
        params["var95"] *= vol_mult
        params["vol"] *= min(vol_mult, 1.5)
    if hy_val and hy_val > 350:
        params["mean"] -= 0.02
        params["prob_positive"] -= 0.05

    return {
        "mean_pnl_pct": round(params["mean"], 3),
        "var_95_pct": round(params["var95"], 3),
        "prob_positive": round(params["prob_positive"], 2),
        "regime": regime_label,
        "horizon_months": 6,
    }


def refresh_mc_cache():
    """Refresh Monte Carlo simulation cache."""
    print(f"[MC] Refreshing simulation at {datetime.now(timezone.utc).isoformat()}")
    try:
        settings = load_settings()
        vix = get_vix(settings)
        hy_oas = get_hy_oas(settings)
        vix_val = vix.get("value") if vix else None
        hy_val = hy_oas.get("value") if hy_oas else None
        regime_label = get_regime_label(vix_val, hy_val)

        positions_data = get_positions_data()
        positions_list = positions_data.get("positions", [])
        cash_available = positions_data.get("cash_available", 0)

        forecast = calculate_monte_carlo_forecast(
            vix_val, hy_val, regime_label,
            positions_data=positions_list,
            cash_available=cash_available,
        )

        with MC_CACHE_LOCK:
            MC_CACHE["forecast"] = forecast
            MC_CACHE["timestamp"] = datetime.now(timezone.utc).isoformat()
            MC_CACHE["last_refresh"] = datetime.now(timezone.utc)

        print(f"[MC] Cache refreshed: mean={forecast.get('mean_pnl_pct')}, VaR95={forecast.get('var_95_pct')}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[MC] Cache refresh failed: {e}")


def mc_background_refresh():
    """Background thread loop — refreshes MC simulation every hour."""
    while True:
        try:
            time.sleep(MC_REFRESH_INTERVAL)
            refresh_mc_cache()
        except Exception as e:
            print(f"[MC] Background refresh error: {e}")
            time.sleep(60)
