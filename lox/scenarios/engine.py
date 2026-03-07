"""
Scenario engine — translate macro shocks into MC parameter adjustments
and run block-bootstrap simulation.

Flow:
  1. Fetch current macro values from FRED
  2. Compute deltas (user target − current)
  3. Translate deltas into MC param adjustments via sensitivity table
  4. Scale by ticker factor exposures
  5. Run block-bootstrap MC (reusing quant_scenarios infrastructure)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from lox.cli_commands.shared.book_impact import FACTOR_EXPOSURES, SECTOR_DEFAULTS
from lox.data.fred import FredClient
from lox.llm.scenarios.quant_scenarios import (
    BASE_JUMP_PROB,
    BLOCK_LEN,
    JUMP_SIZE_SIGMA,
    _extract_closes,
)

logger = logging.getLogger(__name__)

# ── FRED series IDs for each macro variable ───────────────────────────────

MACRO_FRED_MAP: dict[str, str] = {
    "oil": "DCOILWTICO",
    "cpi": "CPIAUCSL",
    "vix": "VIXCLS",
    "10y": "DGS10",
    "fed_funds": "DFF",
    "hy_spread": "BAMLH0A0HYM2",
    "dxy": "DTWEXBGS",
    "gold": "GOLDAMGBD228NLBM",
}

# Yahoo Finance tickers for live prices (FRED lags 1-2 days for market data).
# Used as primary source; FRED is the fallback.
MACRO_YF_MAP: dict[str, str] = {
    "oil": "CL=F",
    "vix": "^VIX",
    "10y": "^TNX",
    "dxy": "DX-Y.NYB",
    "gold": "GC=F",
}

MACRO_DISPLAY_NAMES: dict[str, str] = {
    "oil": "Oil (WTI)",
    "cpi": "CPI (YoY)",
    "vix": "VIX",
    "10y": "10Y Yield",
    "fed_funds": "Fed Funds",
    "hy_spread": "HY OAS",
    "dxy": "DXY",
    "gold": "Gold",
}

MACRO_UNITS: dict[str, str] = {
    "oil": "$",
    "cpi": "%",
    "vix": "pts",
    "10y": "%",
    "fed_funds": "%",
    "hy_spread": "bps",
    "dxy": "pts",
    "gold": "$",
}

# ── Sensitivity coefficients (per unit of shock) ─────────────────────────
# Each entry maps a macro variable to the MC parameter adjustments per unit.
# Units: oil per $10, CPI per 1pp, VIX per 10pts, 10Y per 100bps,
#         fed_funds per 100bps, hy_spread per 100bps, dxy per 5pts, gold per $200.

SENSITIVITY_TABLE: dict[str, dict[str, float]] = {
    "oil": {
        "unit": 10.0,
        "equity_drift_adj": -0.02,
        "equity_vol_mult": 1.07,
        "jump_prob_mult": 1.15,
    },
    "cpi": {
        "unit": 1.0,
        "equity_drift_adj": -0.03,
        "equity_vol_mult": 1.10,
        "spread_drift_adj": 0.0020,
    },
    "vix": {
        "unit": 10.0,
        "equity_vol_mult": 1.30,
        "jump_prob_mult": 1.50,
        "equity_drift_adj": -0.02,
    },
    "10y": {
        "unit": 1.0,  # 100bps
        "equity_drift_adj": -0.05,
        "duration_drift_adj": -0.08,
    },
    "fed_funds": {
        "unit": 1.0,  # 100bps
        "equity_drift_adj": -0.03,
        "rate_drift_adj": 0.01,
    },
    "hy_spread": {
        "unit": 100.0,  # 100bps
        "equity_drift_adj": -0.04,
        "equity_vol_mult": 1.15,
        "spread_drift_adj": 0.0030,
    },
    "dxy": {
        "unit": 5.0,
        "equity_drift_adj": -0.01,
        "commodity_drift_adj": -0.03,
    },
    "gold": {
        "unit": 200.0,
        "equity_drift_adj": -0.005,
        "gold_drift_adj": 0.05,
    },
}


# ── Result data structures ────────────────────────────────────────────────

@dataclass
class MacroAssumption:
    """A single macro variable shock assumption."""
    variable: str
    display_name: str
    current: float
    scenario: float
    shock_abs: float
    shock_pct: float
    unit: str


@dataclass
class FactorSensitivity:
    """Impact of a single macro factor on MC output."""
    variable: str
    display_name: str
    expected_return_delta: float   # change in expected return vs baseline (pp)
    var_95_delta: float            # change in VaR95 vs baseline (pp)
    vol_delta: float               # change in return std dev vs baseline (pp)
    median_price: float            # median terminal price under this factor alone


@dataclass
class OptimalExitEstimate:
    """Option position P&L with full-path optimal exit analysis."""
    display_name: str
    opt_type: str           # "put" or "call"
    strike: float
    expiry: str             # ISO date
    qty: float
    entry_value: float      # current option price per contract ($)
    scenario_pnl: dict[str, float] = field(default_factory=dict)
    scenario_peak_pnl: dict[str, float] = field(default_factory=dict)
    scenario_worst_pnl: dict[str, float] = field(default_factory=dict)
    # Optimal exit stats (from full-path analysis)
    median_peak_pnl: float = 0.0
    p25_peak_pnl: float = 0.0
    p75_peak_pnl: float = 0.0
    median_peak_day: int = 0
    prob_profit: float = 0.0
    prob_double: float = 0.0
    prob_triple: float = 0.0
    median_exit_price: float = 0.0


# Backward-compatible alias
PositionEstimate = OptimalExitEstimate


@dataclass
class AnalogOverlay:
    """Comparison of historical analog returns vs MC distribution."""
    n_analogs: int
    analog_mean_return: float         # mean forward return from k-NN analogs (%)
    analog_median_return: float       # median forward return (%)
    analog_hit_rate: float            # % of analogs with positive return
    analog_var_5: float               # 5th percentile analog return (%)
    analog_p25: float                 # 25th percentile (%)
    analog_p75: float                 # 75th percentile (%)
    mc_median_return: float           # MC median return for comparison (%)
    mc_mean_return: float             # MC mean return for comparison (%)
    calibration_gap: float            # analog_median - mc_median (pp)
    signal: str                       # "ALIGNED", "MC_OPTIMISTIC", "MC_PESSIMISTIC"


@dataclass
class ScenarioAnalysis:
    """Complete result of a scenario simulation."""
    symbol: str
    current_price: float
    horizon_days: int
    trading_days: int
    n_sims: int
    assumptions: list[MacroAssumption]
    regime_effect: str | None
    mc_adjustments: dict[str, float]
    mc_breakdowns: dict[str, dict[str, float]]
    scenarios: dict[str, dict[str, Any]]
    full_distribution: dict[str, float]
    risk_metrics: dict[str, float] = field(default_factory=dict)
    factor_sensitivities: list[FactorSensitivity] = field(default_factory=list)
    analog_overlay: AnalogOverlay | None = None
    histogram: dict[str, Any] = field(default_factory=dict)
    multi_horizon: list[dict[str, Any]] = field(default_factory=list)
    position_estimates: list[PositionEstimate] = field(default_factory=list)


# ── Position detection & BSM estimation ──────────────────────────────────


def detect_option_positions(settings: Any, symbol: str) -> list[dict]:
    """Auto-detect option positions on `symbol` from Alpaca.

    Returns a list of dicts with keys:
      display_name, opt_type, strike, expiry, qty, current_value, symbol (OCC)
    """
    try:
        from lox.data.alpaca import make_clients
        from lox.risk.greeks import parse_occ_symbol, _display_name

        trading, _ = make_clients(settings)
        positions = trading.get_all_positions()
    except Exception as exc:
        logger.debug("Could not fetch positions: %s", exc)
        return []

    results = []
    for pos in positions:
        sym = pos.symbol if isinstance(pos.symbol, str) else str(pos.symbol)
        parsed = parse_occ_symbol(sym)
        if parsed is None:
            continue
        if parsed["underlying"] != symbol.upper():
            continue

        qty = float(pos.qty)
        current_price = float(pos.current_price) if pos.current_price else 0.0

        results.append({
            "display_name": _display_name(sym, parsed),
            "opt_type": parsed["opt_type"],
            "strike": parsed["strike"],
            "expiry": parsed["expiry"].isoformat(),
            "qty": qty,
            "current_value": current_price,
            "symbol": sym,
        })

    return results


def estimate_optimal_exit(
    positions: list[dict],
    scenarios: dict[str, dict],
    path_matrix: np.ndarray,
    annual_vol: float,
    vol_scale: float,
    risk_free_rate: float,
    trading_days: int,
    max_sample_paths: int = 2_000,
) -> list[OptimalExitEstimate]:
    """Full-path optimal exit analysis for option positions.

    Reprices the option at every trading day along sampled MC price paths
    to find the peak P&L, when it typically occurs, and profit probabilities.

    Parameters
    ----------
    positions : list of position dicts (from detect_option_positions or manual).
    scenarios : the BULL/BASE/BEAR/TAIL_RISK dict from ScenarioAnalysis.
    path_matrix : (n_sims, trading_days) array of daily underlying prices.
    annual_vol : realized annual vol of the underlying.
    vol_scale : the MC vol adjustment multiplier.
    risk_free_rate : annualized risk-free rate.
    trading_days : number of trading days in the horizon.
    max_sample_paths : subsample for speed (default 2000).
    """
    from datetime import date as _date
    from lox.scenarios.bsm import option_price, option_price_vec

    adjusted_vol = annual_vol * vol_scale
    today = _date.today()

    n_total = path_matrix.shape[0]
    if n_total > max_sample_paths:
        idx = np.random.default_rng(99).choice(n_total, max_sample_paths, replace=False)
        sampled = path_matrix[idx]
    else:
        sampled = path_matrix

    n_paths, n_days = sampled.shape

    estimates = []
    for pos in positions:
        expiry = _date.fromisoformat(pos["expiry"])
        entry_price = pos["current_value"]
        qty = pos["qty"]
        opt_type = pos["opt_type"]
        strike = pos["strike"]

        # Scenario endpoint P&L (same as before, using scalar pricing)
        dte_at_horizon = max((expiry - today).days - trading_days, 0)
        T_horizon = dte_at_horizon / 365.0
        scenario_pnl: dict[str, float] = {}
        for key in ("BULL", "BASE", "BEAR", "TAIL_RISK"):
            target_price = scenarios[key]["target"]
            future_val = option_price(
                opt_type=opt_type, S=target_price, K=strike,
                T=T_horizon, r=risk_free_rate, sigma=adjusted_vol,
            )
            pnl = (future_val - entry_price) * qty * 100
            scenario_pnl[key] = round(pnl, 0)

        # Full-path analysis: reprice at every day on every sampled path
        # T_arr[d] = years remaining from day d to expiry
        days_to_expiry_from_today = max((expiry - today).days, 0)
        T_arr = np.array([
            max(days_to_expiry_from_today - d, 0) / 365.0
            for d in range(1, n_days + 1)
        ])  # shape (n_days,)

        # Broadcast: S_mat is (n_paths, n_days), T_mat is (n_paths, n_days)
        T_mat = np.broadcast_to(T_arr[np.newaxis, :], sampled.shape)

        opt_vals = option_price_vec(
            opt_type=opt_type, S=sampled, K=strike,
            T=T_mat, r=risk_free_rate, sigma=adjusted_vol,
        )  # (n_paths, n_days)

        pnl_mat = (opt_vals - entry_price) * qty * 100  # (n_paths, n_days)

        # Per-path: peak P&L and day of peak
        peak_pnl_per_path = np.max(pnl_mat, axis=1)        # (n_paths,)
        peak_day_per_path = np.argmax(pnl_mat, axis=1) + 1  # 1-indexed

        # Option price at the peak for each path
        peak_opt_price_per_path = np.array([
            opt_vals[i, peak_day_per_path[i] - 1] for i in range(n_paths)
        ])

        # Profit thresholds (based on return on entry cost)
        entry_cost = abs(entry_price * qty * 100)
        if entry_cost > 0:
            ever_profit = peak_pnl_per_path > 0
            ever_double = peak_pnl_per_path >= entry_cost
            ever_triple = peak_pnl_per_path >= 2 * entry_cost
        else:
            ever_profit = ever_double = ever_triple = np.zeros(n_paths, dtype=bool)

        # Peak and worst P&L per scenario bucket: bucket paths by terminal price,
        # then report the median of the best/worst exit within each bucket.
        terminal_prices_sampled = sampled[:, -1]
        scenario_peak_pnl: dict[str, float] = {}
        scenario_worst_pnl: dict[str, float] = {}
        worst_pnl_per_path = np.min(pnl_mat, axis=1)  # (n_paths,)
        bucket_bounds = {
            "BULL":      (scenarios["BASE"]["target"], float("inf")),
            "BASE":      (scenarios["BEAR"]["target"], scenarios["BASE"]["target"]),
            "BEAR":      (scenarios["TAIL_RISK"]["range"][1], scenarios["BEAR"]["target"]),
            "TAIL_RISK": (0.0, scenarios["TAIL_RISK"]["range"][1]),
        }
        for key, (lo, hi) in bucket_bounds.items():
            mask = (terminal_prices_sampled >= lo) & (terminal_prices_sampled < hi)
            if key == "BULL":
                mask = terminal_prices_sampled >= lo
            if mask.any():
                scenario_peak_pnl[key] = round(float(np.median(peak_pnl_per_path[mask])), 0)
                scenario_worst_pnl[key] = round(float(np.median(worst_pnl_per_path[mask])), 0)
            else:
                scenario_peak_pnl[key] = scenario_pnl.get(key, 0)
                scenario_worst_pnl[key] = scenario_pnl.get(key, 0)

        estimates.append(OptimalExitEstimate(
            display_name=pos["display_name"],
            opt_type=opt_type,
            strike=strike,
            expiry=pos["expiry"],
            qty=qty,
            entry_value=entry_price,
            scenario_pnl=scenario_pnl,
            scenario_peak_pnl=scenario_peak_pnl,
            scenario_worst_pnl=scenario_worst_pnl,
            median_peak_pnl=round(float(np.median(peak_pnl_per_path)), 0),
            p25_peak_pnl=round(float(np.percentile(peak_pnl_per_path, 25)), 0),
            p75_peak_pnl=round(float(np.percentile(peak_pnl_per_path, 75)), 0),
            median_peak_day=int(np.median(peak_day_per_path)),
            prob_profit=round(float(np.mean(ever_profit)) * 100, 1),
            prob_double=round(float(np.mean(ever_double)) * 100, 1),
            prob_triple=round(float(np.mean(ever_triple)) * 100, 1),
            median_exit_price=round(float(np.median(peak_opt_price_per_path)), 2),
        ))

    return estimates


# ── CPI YoY helper ───────────────────────────────────────────────────────

def _compute_cpi_yoy(fred: FredClient) -> float | None:
    """Compute trailing CPI YoY% from the CPI index level series."""
    try:
        df = fred.fetch_series("CPIAUCSL", start_date="2022-01-01")
        if df.empty or len(df) < 13:
            return None
        latest = df.iloc[-1]["value"]
        year_ago = df.iloc[-13]["value"]
        return ((latest / year_ago) - 1.0) * 100.0
    except Exception:
        return None


# ── Core engine ───────────────────────────────────────────────────────────

def _fetch_yf_live(variables: list[str]) -> dict[str, float]:
    """Try yfinance for live market prices. Returns what it can, silently."""
    live: dict[str, float] = {}
    yf_needed = {v: MACRO_YF_MAP[v] for v in variables if v in MACRO_YF_MAP}
    if not yf_needed:
        return live
    try:
        import yfinance as yf
        tickers = list(yf_needed.values())
        data = yf.download(tickers, period="1d", progress=False, threads=True)
        if data.empty:
            return live
        for var, ticker in yf_needed.items():
            try:
                close = data["Close"]
                # yfinance 1.2+ always returns multi-level columns (Price, Ticker)
                if hasattr(close, "columns"):
                    val = float(close[ticker].iloc[-1].item())
                else:
                    val = float(close.iloc[-1].item())
                if val > 0:
                    live[var] = val
            except Exception:
                continue
    except Exception as exc:
        logger.debug("yfinance live fetch failed: %s", exc)
    return live


def _fetch_current_values(
    fred: FredClient,
    variables: list[str],
) -> dict[str, float | None]:
    """Fetch current values: yfinance live first, then FRED as fallback."""
    values: dict[str, float | None] = {}

    # Try yfinance for live market data (covers oil, vix, 10y, dxy, gold)
    live = _fetch_yf_live(variables)
    for var, price in live.items():
        values[var] = price

    for var in variables:
        if var in values:
            continue

        if var == "cpi":
            values[var] = _compute_cpi_yoy(fred)
            continue

        series_id = MACRO_FRED_MAP.get(var)
        if not series_id:
            values[var] = None
            continue

        try:
            df = fred.fetch_series(series_id, start_date="2024-01-01")
            if df.empty:
                values[var] = None
            else:
                raw = float(df.iloc[-1]["value"])
                if var == "hy_spread":
                    raw *= 100.0
                values[var] = raw
        except Exception as exc:
            logger.warning("Failed to fetch %s (%s): %s", var, series_id, exc)
            values[var] = None

    return values


def _translate_shocks_to_mc(
    macro_targets: dict[str, float],
    current_values: dict[str, float | None],
    factor_exposures: dict[str, float] | None,
) -> tuple[dict[str, float], dict[str, dict[str, float]], str | None]:
    """
    Translate macro deltas into MC parameter adjustments.

    Returns (mc_adjustments, breakdowns_by_param, regime_effect_label).
    """
    mc = {
        "equity_drift_adj": 0.0,
        "equity_vol_adj": 1.0,
        "jump_prob_adj": 1.0,
    }
    breakdowns: dict[str, dict[str, float]] = {
        "equity_drift_adj": {},
        "equity_vol_adj": {},
        "jump_prob_adj": {},
    }

    # Detect regime effects
    has_oil_shock = False
    has_inflation = False
    has_vol_spike = False

    for var, target in macro_targets.items():
        current = current_values.get(var)
        if current is None:
            continue

        sens = SENSITIVITY_TABLE.get(var)
        if sens is None:
            continue

        delta = target - current
        unit_size = sens["unit"]
        n_units = delta / unit_size

        # Equity drift adjustment (additive)
        drift_per_unit = sens.get("equity_drift_adj", 0.0)
        if drift_per_unit:
            adj = drift_per_unit * n_units
            # Scale by factor exposure if available
            if factor_exposures:
                equity_beta = factor_exposures.get("equity_beta", 1.0)
                commodity_factor = factor_exposures.get("commodities", 0.0)
                if var == "oil" and commodity_factor > 0:
                    adj = adj * (1.0 - commodity_factor)
                else:
                    adj = adj * abs(equity_beta)
            mc["equity_drift_adj"] += adj
            breakdowns["equity_drift_adj"][var] = adj

        # Vol scaling (multiplicative)
        vol_mult = sens.get("equity_vol_mult")
        if vol_mult is not None:
            per_unit = vol_mult - 1.0
            total_mult = 1.0 + per_unit * abs(n_units)
            mc["equity_vol_adj"] *= total_mult
            breakdowns["equity_vol_adj"][var] = total_mult

        # Jump probability (multiplicative)
        jump_mult = sens.get("jump_prob_mult")
        if jump_mult is not None:
            per_unit = jump_mult - 1.0
            total_mult = 1.0 + per_unit * abs(n_units)
            mc["jump_prob_adj"] *= total_mult
            breakdowns["jump_prob_adj"][var] = total_mult

        # Track regime effects
        if var == "oil" and delta > 5:
            has_oil_shock = True
        if var == "cpi" and delta > 0.5:
            has_inflation = True
        if var == "vix" and delta > 5:
            has_vol_spike = True

    # Determine regime effect label
    regime_effect = None
    if has_oil_shock and has_inflation:
        regime_effect = "Stagflationary pressure"
        mc["equity_vol_adj"] *= 1.05
        mc["jump_prob_adj"] *= 1.10
    elif has_oil_shock:
        regime_effect = "Supply shock"
    elif has_inflation:
        regime_effect = "Inflationary regime"
    elif has_vol_spike:
        regime_effect = "Volatility regime shift"

    return mc, breakdowns, regime_effect


def _get_ticker_factors(settings: Any, symbol: str) -> dict[str, float] | None:
    """Look up factor exposures for a ticker."""
    if symbol in FACTOR_EXPOSURES:
        return FACTOR_EXPOSURES[symbol]
    try:
        from lox.altdata.fmp import fetch_profile
        profile = fetch_profile(settings=settings, ticker=symbol)
        if profile and profile.sector:
            return SECTOR_DEFAULTS.get(profile.sector.lower())
    except Exception:
        pass
    return None


def _run_bootstrap_mc(
    demeaned: np.ndarray,
    adjusted_mu: float,
    vol_scale: float,
    jump_prob: float,
    adjusted_sigma: float,
    current_price: float,
    trading_days: int,
    n_sims: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run block-bootstrap MC and return (terminal_prices, pct_returns)."""
    rng = np.random.default_rng(seed)
    n_returns = len(demeaned)
    n_blocks = int(np.ceil(trading_days / BLOCK_LEN))
    terminal_prices = np.empty(n_sims)

    for i in range(n_sims):
        block_starts = rng.integers(0, n_returns - BLOCK_LEN + 1, size=n_blocks)
        path_rets = np.concatenate(
            [demeaned[s: s + BLOCK_LEN] for s in block_starts]
        )[:trading_days]
        path_rets = path_rets * vol_scale + adjusted_mu
        jump_mask = rng.random(trading_days) < jump_prob
        if jump_mask.any():
            jump_signs = rng.choice([-1, 1], size=int(jump_mask.sum()))
            path_rets[jump_mask] += jump_signs * JUMP_SIZE_SIGMA * adjusted_sigma
        terminal_prices[i] = current_price * np.exp(float(np.sum(path_rets)))

    pct_returns = (terminal_prices - current_price) / current_price
    return terminal_prices, pct_returns


def _compute_factor_sensitivities(
    macro_targets: dict[str, float],
    current_values: dict[str, float | None],
    factor_exposures: dict[str, float] | None,
    demeaned: np.ndarray,
    daily_mu: float,
    daily_sigma: float,
    current_price: float,
    trading_days: int,
    baseline_expected_return: float,
    baseline_var_95: float,
    baseline_vol: float,
) -> list[FactorSensitivity]:
    """
    Run MC once per macro factor in isolation to compute marginal impact.

    Uses a reduced sim count (2000) for speed since we only need
    relative differences, not precise absolute levels.
    """
    results: list[FactorSensitivity] = []
    n_sims_sensitivity = 2_000

    for var, target in macro_targets.items():
        # Run MC with only this single factor's shock
        single_target = {var: target}
        mc, _, _ = _translate_shocks_to_mc(single_target, current_values, factor_exposures)

        drift_adj_daily = mc["equity_drift_adj"] / 252.0
        vol_sc = mc["equity_vol_adj"]
        j_prob = BASE_JUMP_PROB * mc["jump_prob_adj"]
        adj_sigma = daily_sigma * vol_sc
        adj_mu = daily_mu + drift_adj_daily

        tp, pct_rets = _run_bootstrap_mc(
            demeaned=demeaned,
            adjusted_mu=adj_mu,
            vol_scale=vol_sc,
            jump_prob=j_prob,
            adjusted_sigma=adj_sigma,
            current_price=current_price,
            trading_days=trading_days,
            n_sims=n_sims_sensitivity,
            seed=99,
        )

        factor_er = float(np.mean(pct_rets)) * 100
        factor_var = float(np.percentile(pct_rets, 5)) * 100
        factor_vol = float(np.std(pct_rets)) * 100
        factor_median = float(np.median(tp))

        results.append(FactorSensitivity(
            variable=var,
            display_name=MACRO_DISPLAY_NAMES.get(var, var),
            expected_return_delta=round(factor_er - baseline_expected_return, 2),
            var_95_delta=round(factor_var - baseline_var_95, 2),
            vol_delta=round(factor_vol - baseline_vol, 2),
            median_price=round(factor_median, 2),
        ))

    # Sort by absolute impact on expected return (most impactful first)
    results.sort(key=lambda x: abs(x.expected_return_delta), reverse=True)
    return results


def _compute_analog_overlay(
    *,
    settings: Any,
    symbol: str,
    trading_days: int,
    mc_median_return: float,
    mc_full_dist: dict[str, float],
    current_price: float,
) -> AnalogOverlay | None:
    """
    Run k-NN playbook for the symbol at the current regime state and compare
    the analog forward-return distribution against the MC output.

    Returns AnalogOverlay or None if data is insufficient.
    """
    try:
        from lox.regimes.feature_matrix import build_regime_feature_matrix
        from lox.ideas.macro_playbook import rank_macro_playbook
        from lox.data.market import fetch_equity_daily_closes
        from datetime import datetime, timedelta

        feature_matrix = build_regime_feature_matrix(
            settings=settings, start_date="2015-01-01", refresh_fred=False,
        )

        if feature_matrix is None or feature_matrix.empty or len(feature_matrix) < 30:
            return None

        start = (datetime.now() - timedelta(days=600)).strftime("%Y-%m-%d")

        # Fetch price data for the symbol
        px = fetch_equity_daily_closes(
            settings=settings, symbols=[symbol], start=start, refresh=False,
        )
        if px is None or px.empty or symbol not in px.columns:
            return None

        price_aligned = px.reindex(feature_matrix.index).ffill()

        ideas = rank_macro_playbook(
            features=feature_matrix,
            prices=price_aligned,
            tickers=[symbol],
            horizon_days=trading_days,
            k=120,
            min_matches=30,
        )

        if not ideas:
            return None

        idea = ideas[0]
        fwd = idea.fwd_returns * 100  # convert to %

        analog_mean = float(np.mean(fwd))
        analog_median = float(np.median(fwd))
        analog_hit = idea.hit_rate * 100
        analog_var5 = float(np.percentile(fwd, 5))
        analog_p25 = float(np.percentile(fwd, 25))
        analog_p75 = float(np.percentile(fwd, 75))

        # MC comparison values (from risk_metrics expected_return)
        mc_med = (mc_full_dist["p50"] / current_price - 1) * 100
        mc_mean = mc_median_return

        gap = analog_median - mc_med

        if abs(gap) < 2.0:
            signal = "ALIGNED"
        elif gap > 0:
            signal = "MC_PESSIMISTIC"
        else:
            signal = "MC_OPTIMISTIC"

        return AnalogOverlay(
            n_analogs=idea.n_analogs,
            analog_mean_return=round(analog_mean, 2),
            analog_median_return=round(analog_median, 2),
            analog_hit_rate=round(analog_hit, 1),
            analog_var_5=round(analog_var5, 2),
            analog_p25=round(analog_p25, 2),
            analog_p75=round(analog_p75, 2),
            mc_median_return=round(mc_med, 2),
            mc_mean_return=round(mc_mean, 2),
            calibration_gap=round(gap, 2),
            signal=signal,
        )
    except Exception as exc:
        logger.warning("Analog overlay computation failed: %s", exc)
        return None


def run_scenario_analysis(
    settings: Any,
    symbol: str,
    macro_targets: dict[str, float],
    horizon_days: int = 90,
    n_sims: int = 10_000,
    positions: list[dict] | None = None,
) -> ScenarioAnalysis:
    """
    Run a full macro scenario analysis.

    Parameters
    ----------
    settings : Settings
        App configuration (needs FRED_API_KEY, FMP_API_KEY).
    symbol : str
        Ticker to simulate (e.g. SPY, QQQ, XLE).
    macro_targets : dict
        Target values for macro variables (e.g. {"oil": 90, "cpi": 4.5}).
    horizon_days : int
        Calendar-day horizon (converted to trading days internally).
    n_sims : int
        Number of Monte Carlo paths.

    Returns
    -------
    ScenarioAnalysis
        Full results including assumptions, MC adjustments, and price scenarios.
    """
    trading_days = int(horizon_days * 252 / 365)

    # 1. Fetch current macro values
    fred = FredClient(api_key=settings.FRED_API_KEY)
    variables = list(macro_targets.keys())
    current_values = _fetch_current_values(fred, variables)

    # 2. Build assumptions list
    assumptions: list[MacroAssumption] = []
    for var in variables:
        current = current_values.get(var)
        target = macro_targets[var]
        if current is not None:
            shock_abs = target - current
            shock_pct = (shock_abs / current * 100) if current != 0 else 0.0
        else:
            shock_abs = 0.0
            shock_pct = 0.0
        assumptions.append(MacroAssumption(
            variable=var,
            display_name=MACRO_DISPLAY_NAMES.get(var, var),
            current=current or 0.0,
            scenario=target,
            shock_abs=shock_abs,
            shock_pct=shock_pct,
            unit=MACRO_UNITS.get(var, ""),
        ))

    # 3. Get factor exposures for the ticker
    factor_exposures = _get_ticker_factors(settings, symbol)

    # 4. Translate shocks → MC params
    mc_adjustments, mc_breakdowns, regime_effect = _translate_shocks_to_mc(
        macro_targets, current_values, factor_exposures,
    )

    # 5. Fetch historical prices and run MC
    from lox.cli_commands.research.ticker.data import fetch_price_data
    price_data = fetch_price_data(settings, symbol)
    historical = price_data.get("historical", [])
    if not historical:
        raise ValueError(f"No historical price data available for {symbol}")

    quote = price_data.get("quote", {})
    current_price = quote.get("price") or quote.get("previousClose")
    if current_price is None:
        closes = _extract_closes(historical)
        current_price = float(closes[-1])

    closes = _extract_closes(historical)
    log_rets = np.diff(np.log(closes))
    daily_mu = float(np.mean(log_rets))
    daily_sigma = float(np.std(log_rets, ddof=1))

    drift_adj_daily = mc_adjustments["equity_drift_adj"] / 252.0
    vol_scale = mc_adjustments["equity_vol_adj"]
    jump_prob = BASE_JUMP_PROB * mc_adjustments["jump_prob_adj"]

    adjusted_sigma = daily_sigma * vol_scale
    adjusted_mu = daily_mu + drift_adj_daily

    rng = np.random.default_rng(42)
    demeaned = log_rets - daily_mu
    n_returns = len(demeaned)
    n_blocks = int(np.ceil(trading_days / BLOCK_LEN))

    terminal_prices = np.empty(n_sims)
    max_drawdowns = np.empty(n_sims)
    drawdown_days = np.empty(n_sims)
    terminal_returns = np.empty(n_sims)
    # Store full price paths for optimal-exit analysis
    path_matrix = np.empty((n_sims, trading_days))

    for i in range(n_sims):
        block_starts = rng.integers(0, n_returns - BLOCK_LEN + 1, size=n_blocks)
        path_rets = np.concatenate(
            [demeaned[s: s + BLOCK_LEN] for s in block_starts]
        )[:trading_days]
        path_rets = path_rets * vol_scale + adjusted_mu
        jump_mask = rng.random(trading_days) < jump_prob
        if jump_mask.any():
            jump_signs = rng.choice([-1, 1], size=int(jump_mask.sum()))
            path_rets[jump_mask] += jump_signs * JUMP_SIZE_SIGMA * adjusted_sigma

        cum_log_rets = np.cumsum(path_rets)
        terminal_returns[i] = cum_log_rets[-1]
        terminal_prices[i] = current_price * np.exp(cum_log_rets[-1])

        path_prices = current_price * np.exp(cum_log_rets)
        path_matrix[i] = path_prices
        running_max = np.maximum.accumulate(
            np.concatenate([[current_price], path_prices])
        )
        dd_series = (
            np.concatenate([[current_price], path_prices]) - running_max
        ) / running_max
        max_drawdowns[i] = float(np.min(dd_series))
        drawdown_days[i] = float(np.argmin(dd_series))

    # 6. Extract percentiles, risk metrics, and build scenarios
    from lox.llm.scenarios.quant_scenarios import _compute_risk_metrics

    pcts = np.percentile(terminal_prices, [5, 10, 25, 50, 75, 90, 95])
    p5, p10, p25, p50, p75, p90, p95 = (float(v) for v in pcts)

    risk_metrics = _compute_risk_metrics(
        terminal_prices=terminal_prices,
        terminal_returns=terminal_returns,
        max_drawdowns=max_drawdowns,
        drawdown_days=drawdown_days,
        current_price=current_price,
        horizon_days=trading_days,
        daily_mu=adjusted_mu,
    )

    # Histogram of terminal returns
    from lox.llm.scenarios.quant_scenarios import _compute_return_histogram
    pct_returns_for_hist = (terminal_prices - current_price) / current_price * 100
    histogram = _compute_return_histogram(pct_returns_for_hist)

    # Scenario probabilities tilted by MC adjustments
    drift = mc_adjustments["equity_drift_adj"]
    if drift < -0.05:
        probs = {"BULL": 10, "BASE": 35, "BEAR": 35, "TAIL_RISK": 20}
    elif drift < -0.02:
        probs = {"BULL": 15, "BASE": 40, "BEAR": 30, "TAIL_RISK": 15}
    elif drift > 0.02:
        probs = {"BULL": 30, "BASE": 45, "BEAR": 18, "TAIL_RISK": 7}
    else:
        probs = {"BULL": 20, "BASE": 45, "BEAR": 25, "TAIL_RISK": 10}

    def _r2(v: float) -> float:
        return round(v, 2)

    def _ret(target: float) -> float:
        return round((target / current_price - 1) * 100, 1)

    scenarios = {
        "BULL": {
            "target": _r2(p75), "range": [_r2(p50), _r2(p90)],
            "return": _ret(p75), "probability": probs["BULL"],
        },
        "BASE": {
            "target": _r2(p50), "range": [_r2(p25), _r2(p75)],
            "return": _ret(p50), "probability": probs["BASE"],
        },
        "BEAR": {
            "target": _r2(p25), "range": [_r2(p10), _r2(p50)],
            "return": _ret(p25), "probability": probs["BEAR"],
        },
        "TAIL_RISK": {
            "target": _r2(p5), "range": [_r2(p5), _r2(p10)],
            "return": _ret(p5), "probability": probs["TAIL_RISK"],
        },
    }

    full_dist = {
        "p5": _r2(p5), "p10": _r2(p10), "p25": _r2(p25), "p50": _r2(p50),
        "p75": _r2(p75), "p90": _r2(p90), "p95": _r2(p95),
    }

    # 7. Factor sensitivity decomposition (only when multiple factors)
    factor_sensitivities: list[FactorSensitivity] = []
    if len(macro_targets) >= 2:
        # Compute baseline (no shocks) for comparison
        _, baseline_pct = _run_bootstrap_mc(
            demeaned=demeaned,
            adjusted_mu=daily_mu,
            vol_scale=1.0,
            jump_prob=BASE_JUMP_PROB,
            adjusted_sigma=daily_sigma,
            current_price=current_price,
            trading_days=trading_days,
            n_sims=2_000,
            seed=99,
        )
        baseline_er = float(np.mean(baseline_pct)) * 100
        baseline_var = float(np.percentile(baseline_pct, 5)) * 100
        baseline_vol = float(np.std(baseline_pct)) * 100

        factor_sensitivities = _compute_factor_sensitivities(
            macro_targets=macro_targets,
            current_values=current_values,
            factor_exposures=factor_exposures,
            demeaned=demeaned,
            daily_mu=daily_mu,
            daily_sigma=daily_sigma,
            current_price=current_price,
            trading_days=trading_days,
            baseline_expected_return=baseline_er,
            baseline_var_95=baseline_var,
            baseline_vol=baseline_vol,
        )

    # 8. Historical analog overlay (k-NN comparison vs MC distribution)
    analog_overlay = _compute_analog_overlay(
        settings=settings,
        symbol=symbol,
        trading_days=trading_days,
        mc_median_return=risk_metrics.get("expected_return", 0.0),
        mc_full_dist=full_dist,
        current_price=current_price,
    )

    # 9. Multi-horizon term structure (1M / 3M / 6M)
    multi_horizon: list[dict[str, Any]] = []
    _mh_horizons = [21, 63, 126]
    _mh_labels = {21: "1M", 63: "3M", 126: "6M"}
    _mh_n_sims = 3_000

    for mh in _mh_horizons:
        if mh == trading_days:
            # Reuse existing results for the matching horizon
            multi_horizon.append({
                "horizon_days": mh,
                "horizon_label": _mh_labels.get(mh, f"{mh}d"),
                "expected_return": risk_metrics.get("expected_return", 0),
                "var_95": risk_metrics.get("var_95", 0),
                "cvar_95": risk_metrics.get("cvar_95", 0),
                "prob_loss": risk_metrics.get("prob_loss", 0),
                "max_drawdown_median": risk_metrics.get("max_drawdown_median", 0),
                "sortino": risk_metrics.get("sortino", 0),
                "p25": _ret(p25),
                "p50": _ret(p50),
                "p75": _ret(p75),
            })
            continue

        tp_mh, pct_mh = _run_bootstrap_mc(
            demeaned=demeaned,
            adjusted_mu=adjusted_mu,
            vol_scale=vol_scale,
            jump_prob=jump_prob,
            adjusted_sigma=adjusted_sigma,
            current_price=current_price,
            trading_days=mh,
            n_sims=_mh_n_sims,
            seed=77 + mh,
        )
        pcts_mh = np.percentile(tp_mh, [25, 50, 75])
        neg_rets = pct_mh[pct_mh < 0]

        # Path-level drawdown for this horizon
        rng_mh = np.random.default_rng(77 + mh)
        n_blocks_mh = int(np.ceil(mh / BLOCK_LEN))
        mh_drawdowns = np.empty(_mh_n_sims)
        for ii in range(_mh_n_sims):
            bs = rng_mh.integers(0, len(demeaned) - BLOCK_LEN + 1, size=n_blocks_mh)
            pr = np.concatenate([demeaned[s: s + BLOCK_LEN] for s in bs])[:mh]
            pr = pr * vol_scale + adjusted_mu
            jm = rng_mh.random(mh) < jump_prob
            if jm.any():
                js = rng_mh.choice([-1, 1], size=int(jm.sum()))
                pr[jm] += js * JUMP_SIZE_SIGMA * adjusted_sigma
            pp = current_price * np.exp(np.cumsum(pr))
            rm_series = np.maximum.accumulate(np.concatenate([[current_price], pp]))
            dd_s = (np.concatenate([[current_price], pp]) - rm_series) / rm_series
            mh_drawdowns[ii] = float(np.min(dd_s))

        # Sortino
        if len(neg_rets) > 0:
            ds = float(np.sqrt(np.mean(neg_rets ** 2)))
            af = 252.0 / mh
            sortino_mh = float(np.mean(pct_mh)) * af / (ds * np.sqrt(af)) if ds > 0 else 0
        else:
            sortino_mh = 99.0

        multi_horizon.append({
            "horizon_days": mh,
            "horizon_label": _mh_labels.get(mh, f"{mh}d"),
            "expected_return": round(float(np.mean(pct_mh)) * 100, 2),
            "var_95": round(float(np.percentile(pct_mh, 5)) * 100, 2),
            "cvar_95": round(float(np.mean(pct_mh[pct_mh <= np.percentile(pct_mh, 5)])) * 100, 2)
                if (pct_mh <= np.percentile(pct_mh, 5)).any() else 0,
            "prob_loss": round(float(np.mean(pct_mh < 0)) * 100, 1),
            "max_drawdown_median": round(float(np.percentile(mh_drawdowns, 50)) * 100, 2),
            "sortino": round(sortino_mh, 2),
            "p25": round((float(pcts_mh[0]) / current_price - 1) * 100, 1),
            "p50": round((float(pcts_mh[1]) / current_price - 1) * 100, 1),
            "p75": round((float(pcts_mh[2]) / current_price - 1) * 100, 1),
        })

    # 10. Position P&L + optimal exit estimation
    position_estimates: list[OptimalExitEstimate] = []
    if positions:
        annual_vol = daily_sigma * np.sqrt(252)

        fed_funds_rate = 0.05
        try:
            ff_val = current_values.get("fed_funds")
            if ff_val is not None:
                fed_funds_rate = ff_val / 100.0
            else:
                ff_df = fred.fetch_series("DFF", start_date="2024-01-01")
                if not ff_df.empty:
                    fed_funds_rate = float(ff_df.iloc[-1]["value"]) / 100.0
        except Exception:
            pass

        position_estimates = estimate_optimal_exit(
            positions=positions,
            scenarios=scenarios,
            path_matrix=path_matrix,
            annual_vol=annual_vol,
            vol_scale=vol_scale,
            risk_free_rate=fed_funds_rate,
            trading_days=trading_days,
        )

    return ScenarioAnalysis(
        symbol=symbol,
        current_price=_r2(current_price),
        horizon_days=horizon_days,
        trading_days=trading_days,
        n_sims=n_sims,
        assumptions=assumptions,
        regime_effect=regime_effect,
        mc_adjustments=mc_adjustments,
        mc_breakdowns=mc_breakdowns,
        scenarios=scenarios,
        full_distribution=full_dist,
        risk_metrics=risk_metrics,
        factor_sensitivities=factor_sensitivities,
        analog_overlay=analog_overlay,
        histogram=histogram,
        multi_horizon=multi_horizon,
        position_estimates=position_estimates,
    )
