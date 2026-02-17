"""
Unified regime feature extraction for ML and Monte Carlo.

Pulls all regime classifications into a single ML-friendly feature vector.

12 regimes (Feb 2026 restructure):
  Core (MC inputs): Growth, Inflation, Volatility, Credit, Rates, Funding
  Extended (context): Consumer, Fiscal, Positioning, Monetary, USD, Commodities
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from lox.config import load_settings
from lox.regimes.base import categorize_regime, RegimeResult

logger = logging.getLogger(__name__)

# ── Regime weights for overall score (sum to 1.0) ────────────────────────
REGIME_WEIGHTS = {
    "Growth": 0.15,
    "Inflation": 0.10,
    "Volatility": 0.15,
    "Credit": 0.15,
    "Rates": 0.10,
    "Funding": 0.05,
    "Fiscal": 0.05,
    "Consumer": 0.10,
    "Monetary": 0.05,
    "USD": 0.03,
    "Commodities": 0.03,
    "Positioning": 0.04,
}

# ── 12 domain names (ordered for display) ────────────────────────────────
CORE_DOMAINS = ["growth", "inflation", "volatility", "credit", "rates", "liquidity"]
EXTENDED_DOMAINS = ["consumer", "fiscal", "usd", "commodities"]
ALL_DOMAINS = CORE_DOMAINS + EXTENDED_DOMAINS


@dataclass
class UnifiedRegimeState:
    """
    Unified view of all 12 regime classifications.

    Provides a consistent interface for ML feature extraction and Monte Carlo.
    """
    asof: str

    # Core 6 (used in Monte Carlo)
    growth: Optional[RegimeResult] = None
    inflation: Optional[RegimeResult] = None
    volatility: Optional[RegimeResult] = None
    credit: Optional[RegimeResult] = None
    rates: Optional[RegimeResult] = None
    liquidity: Optional[RegimeResult] = None

    # Extended 4 (for overlay/context)
    consumer: Optional[RegimeResult] = None
    fiscal: Optional[RegimeResult] = None
    usd: Optional[RegimeResult] = None
    commodities: Optional[RegimeResult] = None

    # Aggregate risk assessment
    overall_risk_score: float = 50.0
    overall_category: str = "cautious"

    # Macro quadrant (derived from Growth + Inflation)
    macro_quadrant: str = "— MIXED"

    def to_feature_dict(self) -> dict:
        """Convert all regimes to flat ML feature dictionary."""
        features = {
            "asof": self.asof,
            "overall_risk_score": self.overall_risk_score,
            "overall_is_risk_on": 1.0 if self.overall_category == "risk_on" else 0.0,
            "overall_is_risk_off": 1.0 if self.overall_category == "risk_off" else 0.0,
            "macro_quadrant": self.macro_quadrant,
        }

        for domain in ALL_DOMAINS:
            regime = getattr(self, domain, None)
            if regime:
                features[f"{domain}_regime"] = regime.name
                features[f"{domain}_score"] = regime.score
                features[f"{domain}_category"] = categorize_regime(regime.name)
            else:
                features[f"{domain}_regime"] = "unknown"
                features[f"{domain}_score"] = 50.0
                features[f"{domain}_category"] = "cautious"

        return features

    def to_monte_carlo_params(self) -> dict:
        """
        Convert regime state to Monte Carlo scenario parameters.

        Returns dict with:
        - equity_drift, equity_vol adjustments
        - iv_drift, iv_vol adjustments
        - rate_drift, spread_drift adjustments
        - jump_prob, jump_size adjustments
        - drivers: dict mapping each param to list of contributing regimes
        """
        params = {
            "equity_drift_adj": 0.0,
            "equity_vol_adj": 1.0,
            "iv_drift_adj": 0.0,
            "iv_vol_adj": 1.0,
            "rate_drift_adj": 0.0,
            "spread_drift_adj": 0.0,
            "jump_prob_adj": 1.0,
            "jump_size_adj": 1.0,
        }
        drivers: dict[str, list[str]] = {k: [] for k in params}

        # ── Growth regime adjustments ─────────────────────────────────────
        if self.growth and self.growth.score > 65:
            params["equity_drift_adj"] -= 0.03
            drivers["equity_drift_adj"].append(f"Growth({self.growth.score:.0f}) -3%")
            params["jump_prob_adj"] *= 1.25
            drivers["jump_prob_adj"].append(f"Growth({self.growth.score:.0f}) +25%")
        elif self.growth and self.growth.score < 35:
            params["equity_drift_adj"] += 0.02
            drivers["equity_drift_adj"].append(f"Growth({self.growth.score:.0f}) +2%")

        # ── Inflation regime adjustments ──────────────────────────────────
        if self.inflation and self.inflation.score > 65:
            params["iv_drift_adj"] += 0.02
            drivers["iv_drift_adj"].append(f"Inflation({self.inflation.score:.0f}) +2%")
            params["spread_drift_adj"] += 0.002
            drivers["spread_drift_adj"].append(f"Inflation({self.inflation.score:.0f}) +0.2%")
        elif self.inflation and self.inflation.score < 35:
            params["iv_drift_adj"] -= 0.01
            drivers["iv_drift_adj"].append(f"Inflation({self.inflation.score:.0f}) -1%")

        # ── Volatility regime adjustments ─────────────────────────────────
        if self.volatility:
            if "shock" in self.volatility.name:
                params["iv_drift_adj"] += 0.10
                params["iv_vol_adj"] *= 1.5
                params["jump_prob_adj"] *= 2.0
                params["jump_size_adj"] *= 1.3
                drivers["iv_drift_adj"].append(f"Vol({self.volatility.score:.0f}) shock")
                drivers["jump_prob_adj"].append(f"Vol({self.volatility.score:.0f}) shock")
            elif "elevated" in self.volatility.name:
                params["iv_drift_adj"] += 0.03
                params["iv_vol_adj"] *= 1.2
                drivers["iv_drift_adj"].append(f"Vol({self.volatility.score:.0f}) elevated")
            elif "complacent" in self.volatility.name or "normal" in self.volatility.name:
                params["iv_drift_adj"] -= 0.02
                params["iv_vol_adj"] *= 0.9

        # ── Credit regime adjustments (MOST IMPORTANT — credit leads equity vol) ─
        if self.credit:
            if self.credit.score > 70:
                params["equity_vol_adj"] *= 1.40
                drivers["equity_vol_adj"].append(f"Credit({self.credit.score:.0f}) +40%")
                params["jump_prob_adj"] *= 1.75
                drivers["jump_prob_adj"].append(f"Credit({self.credit.score:.0f}) +75%")
                params["spread_drift_adj"] += 0.005
                drivers["spread_drift_adj"].append(f"Credit({self.credit.score:.0f}) +0.5%")
            elif self.credit.score > 55:
                params["equity_vol_adj"] *= 1.15
                drivers["equity_vol_adj"].append(f"Credit({self.credit.score:.0f}) +15%")
                params["spread_drift_adj"] += 0.002
                drivers["spread_drift_adj"].append(f"Credit({self.credit.score:.0f}) +0.2%")
            elif self.credit.score < 30:
                params["equity_vol_adj"] *= 0.90
                drivers["equity_vol_adj"].append(f"Credit({self.credit.score:.0f}) -10%")

        # ── Rates regime adjustments ──────────────────────────────────────
        if self.rates:
            if "shock_up" in self.rates.name:
                params["rate_drift_adj"] += 0.01
                params["equity_drift_adj"] -= 0.02
            elif "shock_down" in self.rates.name:
                params["rate_drift_adj"] -= 0.01
                params["equity_drift_adj"] += 0.02
            elif "inverted" in self.rates.name:
                params["equity_drift_adj"] -= 0.02
                params["jump_prob_adj"] *= 1.3

        # ── Liquidity regime adjustments (continuous score) ────────────────
        if self.liquidity:
            if self.liquidity.score >= 75:
                params["spread_drift_adj"] += 0.01
                params["equity_drift_adj"] -= 0.03
                params["jump_prob_adj"] *= 1.5
                drivers["spread_drift_adj"].append(f"Liquidity({self.liquidity.score:.0f}) +1%")
                drivers["equity_drift_adj"].append(f"Liquidity({self.liquidity.score:.0f}) -3%")
                drivers["jump_prob_adj"].append(f"Liquidity({self.liquidity.score:.0f}) +50%")
            elif self.liquidity.score >= 65:
                params["spread_drift_adj"] += 0.005
                params["equity_drift_adj"] -= 0.015
                params["jump_prob_adj"] *= 1.25
                drivers["spread_drift_adj"].append(f"Liquidity({self.liquidity.score:.0f}) +0.5%")
                drivers["equity_drift_adj"].append(f"Liquidity({self.liquidity.score:.0f}) -1.5%")
                drivers["jump_prob_adj"].append(f"Liquidity({self.liquidity.score:.0f}) +25%")
            elif self.liquidity.score >= 55:
                params["spread_drift_adj"] += 0.003
                drivers["spread_drift_adj"].append(f"Liquidity({self.liquidity.score:.0f}) +0.3%")

        # ── Consumer regime adjustments ───────────────────────────────────
        if self.consumer and self.consumer.score > 65:
            params["equity_drift_adj"] -= 0.02
            drivers["equity_drift_adj"].append(f"Consumer({self.consumer.score:.0f}) -2%")
            params["spread_drift_adj"] += 0.001
            drivers["spread_drift_adj"].append(f"Consumer({self.consumer.score:.0f}) +0.1%")
        elif self.consumer and self.consumer.score < 35:
            params["equity_drift_adj"] += 0.01
            drivers["equity_drift_adj"].append(f"Consumer({self.consumer.score:.0f}) +1%")

        # ── USD regime adjustments ────────────────────────────────────────
        if self.usd:
            if "surge" in self.usd.name or "strong" in self.usd.name:
                params["equity_drift_adj"] -= 0.01
            elif "plunge" in self.usd.name or "weak" in self.usd.name:
                params["equity_drift_adj"] += 0.01

        # ── Fiscal regime adjustments (calibrated via FPI scoring engine) ─
        if self.fiscal:
            try:
                from lox.fiscal.signals import build_fiscal_state
                from lox.fiscal.scoring import score_fiscal_regime
                from lox.fiscal.mc_calibration import calibrate_fiscal_mc

                fiscal_state = build_fiscal_state(
                    settings=load_settings(), start_date="2011-01-01"
                )
                scorecard = score_fiscal_regime(fiscal_state.inputs)
                fiscal_mc = calibrate_fiscal_mc(scorecard)

                params["rate_drift_adj"] += fiscal_mc.rate_drift_adj
                params["rate_drift_adj"] += fiscal_mc.term_premium_bps / 10000.0
                params["spread_drift_adj"] += fiscal_mc.spread_adj_bps / 10000.0
                params["equity_drift_adj"] += fiscal_mc.equity_crowding_out
                params["jump_prob_adj"] *= fiscal_mc.jump_prob_multiplier
                params["jump_size_adj"] *= fiscal_mc.jump_size_multiplier
            except Exception:
                if "dominance" in self.fiscal.name or "stress" in self.fiscal.name:
                    params["rate_drift_adj"] += 0.005
                    params["spread_drift_adj"] += 0.002

        params["_drivers"] = drivers
        return params


def _compute_macro_quadrant(growth: RegimeResult | None, inflation: RegimeResult | None) -> str:
    """Derive the macro quadrant from Growth + Inflation scores."""
    if growth is None or inflation is None:
        return "— MIXED"
    gs = growth.score
    is_ = inflation.score
    if gs > 60 and is_ > 60:
        return "⚠️ STAGFLATION"
    elif gs < 40 and is_ < 40:
        return "✅ GOLDILOCKS"
    elif gs < 40 and is_ > 60:
        return "🔥 REFLATION"
    elif gs > 60 and is_ < 40:
        return "❄️ DEFLATION RISK"
    else:
        return "— MIXED"


def build_unified_regime_state(
    settings=None,
    start_date: str = "2020-01-01",
    refresh: bool = False,
) -> UnifiedRegimeState:
    """
    Build unified regime state by running all regime classifiers.

    Uses ThreadPoolExecutor to run independent regime builders in parallel,
    reducing cold-cache latency from ~10-15s to ~3-5s.

    Args:
        settings: Config settings (loaded if None)
        start_date: Historical start date for data
        refresh: Force refresh of cached data

    Returns:
        UnifiedRegimeState with all regime classifications
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time as _time

    if settings is None:
        settings = load_settings()

    _t0 = _time.monotonic()
    asof = datetime.now().strftime("%Y-%m-%d")
    state = UnifiedRegimeState(asof=asof)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: Fetch macro data (shared by Growth, Inflation, Rates, Consumer)
    # ═══════════════════════════════════════════════════════════════════════
    macro_state = None
    try:
        from lox.macro.signals import build_macro_state
        macro_state = build_macro_state(settings=settings, start_date=start_date, refresh=refresh)
    except Exception as e:
        logger.warning(f"Failed to build macro state: {e}")

    _t_macro = _time.monotonic()
    logger.info(f"[Perf] macro_state built in {(_t_macro - _t0)*1000:.0f}ms")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: Run all regime builders in parallel
    # ═══════════════════════════════════════════════════════════════════════

    def _build_growth():
        try:
            from lox.growth.regime import classify_growth

            payrolls_3m_level = None
            ism_val = None
            claims_4wk = None
            indpro_yoy = None
            payrolls_mom_val = None
            unemployment_val = None
            lei_yoy_val = None

            if macro_state:
                inp = macro_state.inputs
                if inp.payrolls_3m_annualized is not None:
                    payrolls_3m_level = inp.payrolls_3m_annualized * 157_000 / 100 / 12
                claims_4wk = inp.initial_claims_4w
                payrolls_mom_val = inp.payrolls_mom
                unemployment_val = inp.unemployment_rate

            try:
                from lox.altdata.trading_economics import get_ism_manufacturing
                ism_val = get_ism_manufacturing()
            except Exception:
                pass

            try:
                from lox.data.fred import FredClient
                fred = FredClient(api_key=settings.FRED_API_KEY)
                indpro_df = fred.fetch_series("INDPRO", start_date=start_date, refresh=refresh)
                if indpro_df is not None and len(indpro_df) >= 13:
                    indpro_df = indpro_df.sort_values("date")
                    latest = indpro_df["value"].iloc[-1]
                    yr_ago = indpro_df["value"].iloc[-13]
                    if yr_ago > 0:
                        indpro_yoy = (latest / yr_ago - 1.0) * 100.0
            except Exception:
                pass

            try:
                import pandas as _pd_lei
                from lox.data.fred import FredClient
                fred = FredClient(api_key=settings.FRED_API_KEY)
                lei_df = fred.fetch_series("USSLIND", start_date=start_date, refresh=refresh)
                if lei_df is not None and len(lei_df) >= 13:
                    lei_df = lei_df.sort_values("date")
                    latest_date = _pd_lei.to_datetime(lei_df["date"].iloc[-1])
                    if (_pd_lei.Timestamp.now() - latest_date).days < 180:
                        lei_yoy_val = (lei_df["value"].iloc[-1] / lei_df["value"].iloc[-13] - 1.0) * 100.0
            except Exception:
                pass

            ism_new_orders_val = None
            try:
                from lox.altdata.trading_economics import _fmp_latest_actual
                ism_new_orders_val = _fmp_latest_actual(["ism", "new", "orders"])
            except Exception:
                pass

            retail_mom_val = None
            try:
                from lox.altdata.trading_economics import get_retail_sales_control
                retail_mom_val = get_retail_sales_control()
            except Exception:
                pass

            mich_exp_val = None
            try:
                from lox.altdata.trading_economics import get_michigan_expectations
                mich_exp_val = get_michigan_expectations()
            except Exception:
                pass

            result = classify_growth(
                payrolls_3m_ann=payrolls_3m_level,
                ism=ism_val,
                claims_4wk=claims_4wk,
                indpro_yoy=indpro_yoy,
                payrolls_mom=payrolls_mom_val,
                unemployment_rate=unemployment_val,
                lei_yoy=lei_yoy_val,
                ism_new_orders=ism_new_orders_val,
            )

            if result:
                if macro_state:
                    inp_m = macro_state.inputs
                    result.metrics["curve_2s10s"] = inp_m.curve_2s10s
                    result.metrics["hy_oas"] = inp_m.hy_oas * 100 if inp_m.hy_oas is not None else None
                    result.metrics["real_yield"] = inp_m.real_yield_proxy_10y
                    result.metrics["vix"] = inp_m.vix
                else:
                    result.metrics["curve_2s10s"] = None
                    result.metrics["hy_oas"] = None
                    result.metrics["real_yield"] = None
                    result.metrics["vix"] = None
                result.metrics["retail_sales_mom"] = retail_mom_val
                result.metrics["consumer_expectations"] = mich_exp_val
            return result
        except Exception as e:
            logger.warning(f"Failed to build growth regime: {e}")
            return None

    def _build_inflation():
        try:
            from lox.inflation.regime import classify_inflation

            cpi_yoy = None
            core_pce_yoy = None
            breakeven_5y = None
            ppi_yoy = None
            core_cpi_yoy_val = None
            median_cpi_yoy_val = None
            cpi_3m_ann_val = None
            cpi_6m_ann_val = None
            breakeven_5y5y_val = None
            breakeven_10y_val = None
            trimmed_mean_pce_val = None
            eff_ff_val = None

            if macro_state:
                inp = macro_state.inputs
                cpi_yoy = inp.cpi_yoy
                breakeven_5y = inp.breakeven_5y
                core_cpi_yoy_val = inp.core_cpi_yoy
                median_cpi_yoy_val = inp.median_cpi_yoy
                cpi_3m_ann_val = inp.cpi_3m_annualized
                cpi_6m_ann_val = inp.cpi_6m_annualized
                breakeven_5y5y_val = inp.breakeven_5y5y
                breakeven_10y_val = inp.breakeven_10y
                eff_ff_val = inp.eff_fed_funds

            try:
                from lox.data.fred import FredClient
                fred = FredClient(api_key=settings.FRED_API_KEY)
                pce_df = fred.fetch_series("PCEPILFE", start_date=start_date, refresh=refresh)
                if pce_df is not None and len(pce_df) >= 13:
                    pce_df = pce_df.sort_values("date")
                    core_pce_yoy = (pce_df["value"].iloc[-1] / pce_df["value"].iloc[-13] - 1.0) * 100.0
            except Exception:
                pass

            try:
                from lox.data.fred import FredClient
                fred = FredClient(api_key=settings.FRED_API_KEY)
                tm_df = fred.fetch_series("PCETRIM12M159SFRBDAL", start_date=start_date, refresh=refresh)
                if tm_df is not None and not tm_df.empty:
                    trimmed_mean_pce_val = float(tm_df.sort_values("date")["value"].iloc[-1])
            except Exception:
                pass

            try:
                from lox.data.fred import FredClient
                fred = FredClient(api_key=settings.FRED_API_KEY)
                ppi_df = fred.fetch_series("PPIFIS", start_date=start_date, refresh=refresh)
                if ppi_df is not None and len(ppi_df) >= 13:
                    ppi_df = ppi_df.sort_values("date")
                    ppi_yoy = (ppi_df["value"].iloc[-1] / ppi_df["value"].iloc[-13] - 1.0) * 100.0
            except Exception:
                pass

            oil_yoy_val = None
            try:
                from lox.data.fred import FredClient
                fred = FredClient(api_key=settings.FRED_API_KEY)
                oil_df = fred.fetch_series("DCOILWTICO", start_date=start_date, refresh=refresh)
                if oil_df is not None and len(oil_df) >= 252:
                    oil_df = oil_df.sort_values("date")
                    oil_latest = oil_df["value"].iloc[-1]
                    oil_yr_ago = oil_df["value"].iloc[-252]
                    if oil_yr_ago > 0:
                        oil_yoy_val = (oil_latest / oil_yr_ago - 1.0) * 100.0
            except Exception:
                pass

            result = classify_inflation(
                cpi_yoy=cpi_yoy,
                core_pce_yoy=core_pce_yoy,
                breakeven_5y=breakeven_5y,
                ppi_yoy=ppi_yoy,
                core_cpi_yoy=core_cpi_yoy_val,
                trimmed_mean_pce_yoy=trimmed_mean_pce_val,
                median_cpi_yoy=median_cpi_yoy_val,
                cpi_3m_ann=cpi_3m_ann_val,
                cpi_6m_ann=cpi_6m_ann_val,
                breakeven_5y5y=breakeven_5y5y_val,
                breakeven_10y=breakeven_10y_val,
                oil_price_yoy_pct=oil_yoy_val,
            )

            if result:
                result.metrics["oil_yoy"] = oil_yoy_val
                if eff_ff_val is not None and cpi_yoy is not None:
                    result.metrics["real_fed_funds"] = round(eff_ff_val - cpi_yoy, 2)
                else:
                    result.metrics["real_fed_funds"] = None
                if breakeven_10y_val is not None and breakeven_5y is not None:
                    result.metrics["be_slope"] = round(breakeven_10y_val - breakeven_5y, 3)
                else:
                    result.metrics["be_slope"] = None
            return result
        except Exception as e:
            logger.warning(f"Failed to build inflation regime: {e}")
            return None

    def _build_volatility():
        try:
            from lox.volatility.signals import build_volatility_state
            from lox.volatility.regime import classify_volatility_regime

            vol_state = build_volatility_state(settings=settings, start_date=start_date, refresh=refresh)
            vol_regime = classify_volatility_regime(vol_state.inputs)

            inp = vol_state.inputs
            _z = inp.z_vix if inp.z_vix is not None else 0
            _vol_score = max(0, min(100, 50 + _z * 15))
            if inp.vol_pressure_score is not None:
                _vol_score = max(0, min(100, 50 + inp.vol_pressure_score * 15))
            return RegimeResult(
                name=vol_regime.name,
                label=vol_regime.label,
                description=vol_regime.description,
                score=_vol_score,
                domain="volatility",
                tags=list(vol_regime.tags) if hasattr(vol_regime, "tags") else [],
                metrics={
                    "VIX": f"{inp.vix:.1f}" if inp.vix is not None else None,
                    "VIX z": f"{inp.z_vix:+.1f}" if inp.z_vix is not None else None,
                    "VX Contango": f"{inp.vx_contango_pct:+.1f}%" if inp.vx_contango_pct is not None else None,
                    "Spot Basis": f"{inp.spot_basis_pct:+.1f}%" if inp.spot_basis_pct is not None else None,
                    "9d/VIX Ratio": f"{inp.vix9d_vix_ratio:.2f}" if inp.vix9d_vix_ratio is not None else None,
                    "Term Spread": f"{inp.vix_term_spread:+.2f}" if inp.vix_term_spread is not None else None,
                    "VIX 5d Chg": f"{inp.vix_chg_5d_pct:+.1f}%" if inp.vix_chg_5d_pct is not None else None,
                    "Spike 20d": f"{inp.spike_20d_pct:.0f}%" if inp.spike_20d_pct is not None else None,
                    "VIX 9d": f"{inp.vix9d:.1f}" if inp.vix9d is not None else None,
                    "VIX 3m": f"{inp.vix3m:.1f}" if inp.vix3m is not None else None,
                    "VX M1": f"{inp.vx_m1:.2f}" if inp.vx_m1 is not None else None,
                    "VX M2": f"{inp.vx_m2:.2f}" if inp.vx_m2 is not None else None,
                    "Vol Pressure": f"{inp.vol_pressure_score:+.2f}" if inp.vol_pressure_score is not None else None,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to build volatility regime: {e}")
            return None

    def _build_credit():
        try:
            from lox.credit.regime import classify_credit
            from lox.data.fred import FredClient
            import pandas as pd

            fred = FredClient(api_key=settings.FRED_API_KEY)

            hy_df = fred.fetch_series("BAMLH0A0HYM2", start_date=start_date, refresh=refresh)
            bbb_df = fred.fetch_series("BAMLC0A4CBBB", start_date=start_date, refresh=refresh)
            aaa_df = fred.fetch_series("BAMLC0A1CAAA", start_date=start_date, refresh=refresh)

            hy_oas_val = None
            bbb_oas_val = None
            aaa_oas_val = None
            hy_30d_chg = None
            hy_90d_pctl = None

            if hy_df is not None and not hy_df.empty:
                hy_df = hy_df.sort_values("date")
                hy_oas_val = float(hy_df["value"].iloc[-1]) * 100
                if len(hy_df) >= 22:
                    hy_30d_chg = (float(hy_df["value"].iloc[-1]) - float(hy_df["value"].iloc[-22])) * 100
                if len(hy_df) >= 63:
                    recent = hy_df["value"].iloc[-63:]
                    hy_90d_pctl = float((recent <= hy_df["value"].iloc[-1]).mean() * 100)

            if bbb_df is not None and not bbb_df.empty:
                bbb_oas_val = float(bbb_df.sort_values("date")["value"].iloc[-1]) * 100

            if aaa_df is not None and not aaa_df.empty:
                aaa_oas_val = float(aaa_df.sort_values("date")["value"].iloc[-1]) * 100

            ccc_oas_val = None
            bb_oas_val = None
            single_b_oas_val = None
            cc_delinq_val = None
            sloos_val = None

            try:
                ccc_df = fred.fetch_series("BAMLH0A3HYC", start_date=start_date, refresh=refresh)
                if ccc_df is not None and not ccc_df.empty:
                    ccc_oas_val = float(ccc_df.sort_values("date")["value"].iloc[-1]) * 100
            except Exception:
                pass

            try:
                bb_df = fred.fetch_series("BAMLH0A1HYBB", start_date=start_date, refresh=refresh)
                if bb_df is not None and not bb_df.empty:
                    bb_oas_val = float(bb_df.sort_values("date")["value"].iloc[-1]) * 100
            except Exception:
                pass

            try:
                b_df = fred.fetch_series("BAMLH0A2HYB", start_date=start_date, refresh=refresh)
                if b_df is not None and not b_df.empty:
                    single_b_oas_val = float(b_df.sort_values("date")["value"].iloc[-1]) * 100
            except Exception:
                pass

            try:
                delinq_df = fred.fetch_series("DRCCLACBS", start_date=start_date, refresh=refresh)
                if delinq_df is not None and not delinq_df.empty:
                    cc_delinq_val = float(delinq_df.sort_values("date")["value"].iloc[-1])
            except Exception:
                pass

            try:
                sloos_df = fred.fetch_series("DRTSCLCC", start_date=start_date, refresh=refresh)
                if sloos_df is not None and not sloos_df.empty:
                    sloos_val = float(sloos_df.sort_values("date")["value"].iloc[-1])
            except Exception:
                pass

            return classify_credit(
                hy_oas=hy_oas_val,
                bbb_oas=bbb_oas_val,
                aaa_oas=aaa_oas_val,
                hy_oas_30d_chg=hy_30d_chg,
                hy_oas_90d_percentile=hy_90d_pctl,
                ccc_oas=ccc_oas_val,
                bb_oas=bb_oas_val,
                single_b_oas=single_b_oas_val,
                cc_delinquency_rate=cc_delinq_val,
                sloos_tightening=sloos_val,
            )
        except Exception as e:
            logger.warning(f"Failed to build credit regime: {e}")
            return None

    def _build_rates():
        try:
            from lox.rates.signals import build_rates_state
            from lox.rates.regime import classify_rates_regime

            rates_state = build_rates_state(settings=settings, start_date=start_date, refresh=refresh)
            rates_regime = classify_rates_regime(rates_state.inputs)

            inp = rates_state.inputs
            _rz = abs(inp.z_ust_10y) if inp.z_ust_10y is not None else 0
            _curve_pen = 15 if (inp.curve_2s10s is not None and inp.curve_2s10s < 0) else 0
            _rates_score = max(0, min(100, 50 + _rz * 12 + _curve_pen))
            return RegimeResult(
                name=rates_regime.name,
                label=rates_regime.label,
                description=rates_regime.description,
                score=_rates_score,
                domain="rates",
                tags=list(rates_regime.tags) if hasattr(rates_regime, "tags") else [],
                metrics={
                    "10Y": f"{inp.ust_10y:.2f}%" if inp.ust_10y is not None else None,
                    "2Y": f"{inp.ust_2y:.2f}%" if inp.ust_2y is not None else None,
                    "3M": f"{inp.ust_3m:.2f}%" if inp.ust_3m is not None else None,
                    "2s10s": f"{inp.curve_2s10s * 100:+.0f}bp" if inp.curve_2s10s is not None else None,
                    "10Y 20d Chg": f"{inp.ust_10y_chg_20d * 100:+.0f}bp" if inp.ust_10y_chg_20d is not None else None,
                    "3M10Y": f"{inp.curve_3m10y * 100:+.0f}bp" if inp.curve_3m10y is not None else None,
                    "2s10s 20d Chg": f"{inp.curve_2s10s_chg_20d * 100:+.0f}bp" if inp.curve_2s10s_chg_20d is not None else None,
                    "10Y z": f"{inp.z_ust_10y:+.1f}" if inp.z_ust_10y is not None else None,
                    "Real 10Y": f"{macro_state.inputs.real_yield_proxy_10y:.2f}%" if macro_state and macro_state.inputs.real_yield_proxy_10y is not None else None,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to build rates regime: {e}")
            return None

    def _build_liquidity():
        try:
            from lox.funding.signals import build_funding_state
            from lox.funding.regime import classify_funding_regime

            funding_state = build_funding_state(settings=settings, start_date=start_date, refresh=refresh)
            funding_regime = classify_funding_regime(funding_state.inputs)
            f_inp = funding_state.inputs

            m_inp = None
            try:
                from lox.monetary.signals import build_monetary_state
                monetary_state = build_monetary_state(settings=settings, start_date=start_date, refresh=refresh)
                m_inp = monetary_state.inputs
            except Exception:
                pass

            tags = []
            if funding_regime.score >= 75:
                tags.append("risk_off")
            if funding_regime.score >= 65:
                tags.append("structural_tightening")

            liq_metrics = {
                "SOFR": f"{f_inp.sofr:.2f}%" if f_inp.sofr is not None else None,
                "EFFR": f"{f_inp.effr:.2f}%" if f_inp.effr is not None else None,
                "IORB": f"{f_inp.iorb:.2f}%" if f_inp.iorb is not None else None,
                "Corridor": f"{f_inp.spread_corridor_bps:+.1f}bp" if f_inp.spread_corridor_bps is not None else None,
                "Reserves": f"${f_inp.bank_reserves_usd_bn / 1_000_000:.1f}T" if f_inp.bank_reserves_usd_bn is not None else None,
                "Reserve z": f"{f_inp.z_bank_reserves:+.1f}" if f_inp.z_bank_reserves is not None else None,
                "RRP": f"${f_inp.on_rrp_usd_bn / 1000:.0f}B" if f_inp.on_rrp_usd_bn is not None else None,
                "RRP 13w Chg": f"${f_inp.on_rrp_chg_13w / 1000:+,.0f}B" if f_inp.on_rrp_chg_13w is not None else None,
                "TGA": f"${f_inp.tga_usd_bn / 1000:.0f}B" if f_inp.tga_usd_bn is not None else None,
            }
            if m_inp:
                liq_metrics["Fed Assets"] = f"${m_inp.fed_assets / 1e6:.1f}T" if m_inp.fed_assets is not None else None
                liq_metrics["Fed Assets 13w"] = f"${m_inp.fed_assets_chg_13w / 1000:+,.0f}B" if m_inp.fed_assets_chg_13w is not None else None
                liq_metrics["Fed Assets z"] = f"{m_inp.z_fed_assets_chg_13w:+.1f}" if m_inp.z_fed_assets_chg_13w is not None else None
                liq_metrics["Reserves 13w"] = f"${m_inp.reserves_chg_13w / 1000:+,.0f}B" if m_inp.reserves_chg_13w is not None else None

            return RegimeResult(
                name=funding_regime.name,
                label=funding_regime.label,
                description=funding_regime.description if hasattr(funding_regime, "description") else "",
                score=funding_regime.score,
                domain="liquidity",
                tags=tags,
                metrics=liq_metrics,
            )
        except Exception as e:
            logger.warning(f"Failed to build liquidity regime: {e}")
            return None

    def _build_consumer():
        try:
            from lox.consumer.regime import classify_consumer

            michigan_sent = None
            michigan_exp = None
            retail_mom = None
            personal_spend = None
            cc_debt_yoy = None
            mtg_30y = None

            try:
                from lox.data.fred import FredClient
                fred = FredClient(api_key=settings.FRED_API_KEY)
                sent_df = fred.fetch_series("UMCSENT", start_date=start_date, refresh=refresh)
                if sent_df is not None and not sent_df.empty:
                    michigan_sent = float(sent_df.sort_values("date")["value"].iloc[-1])
            except Exception:
                pass

            try:
                from lox.altdata.trading_economics import get_michigan_expectations
                michigan_exp = get_michigan_expectations()
            except Exception:
                pass

            try:
                from lox.data.fred import FredClient
                fred = FredClient(api_key=settings.FRED_API_KEY)
                rs_df = fred.fetch_series("RSXFS", start_date=start_date, refresh=refresh)
                if rs_df is not None and len(rs_df) >= 2:
                    rs_df = rs_df.sort_values("date")
                    retail_mom = (rs_df["value"].iloc[-1] / rs_df["value"].iloc[-2] - 1.0) * 100.0
            except Exception:
                pass

            try:
                from lox.data.fred import FredClient
                fred = FredClient(api_key=settings.FRED_API_KEY)
                cc_df = fred.fetch_series("TOTALSL", start_date=start_date, refresh=refresh)
                if cc_df is not None and len(cc_df) >= 13:
                    cc_df = cc_df.sort_values("date")
                    cc_debt_yoy = (cc_df["value"].iloc[-1] / cc_df["value"].iloc[-13] - 1.0) * 100.0
            except Exception:
                pass

            if macro_state and macro_state.inputs.mortgage_30y:
                mtg_30y = macro_state.inputs.mortgage_30y
            else:
                try:
                    from lox.data.fred import FredClient
                    fred = FredClient(api_key=settings.FRED_API_KEY)
                    mtg_df = fred.fetch_series("MORTGAGE30US", start_date=start_date, refresh=refresh)
                    if mtg_df is not None and not mtg_df.empty:
                        mtg_30y = float(mtg_df.sort_values("date")["value"].iloc[-1])
                except Exception:
                    pass

            result = classify_consumer(
                michigan_sentiment=michigan_sent,
                michigan_expectations=michigan_exp,
                retail_sales_control_mom=retail_mom,
                personal_spending_mom=personal_spend,
                credit_card_debt_yoy_chg=cc_debt_yoy,
                mortgage_30y=mtg_30y,
            )
            if result:
                result.metrics["credit_yoy"] = cc_debt_yoy
                if macro_state:
                    result.metrics["home_prices_yoy"] = macro_state.inputs.home_prices_yoy
                else:
                    result.metrics["home_prices_yoy"] = None
                try:
                    from lox.data.fred import FredClient
                    _fred = FredClient(api_key=settings.FRED_API_KEY)
                    _sr = _fred.fetch_series("PSAVERT", start_date=start_date, refresh=refresh)
                    result.metrics["savings_rate"] = float(_sr.sort_values("date")["value"].iloc[-1]) if _sr is not None and not _sr.empty else None
                except Exception:
                    result.metrics["savings_rate"] = None
                try:
                    from lox.data.fred import FredClient
                    _fred = FredClient(api_key=settings.FRED_API_KEY)
                    _del = _fred.fetch_series("DRALACBN", start_date=start_date, refresh=refresh)
                    result.metrics["delinquency"] = float(_del.sort_values("date")["value"].iloc[-1]) if _del is not None and not _del.empty else None
                except Exception:
                    result.metrics["delinquency"] = None
                try:
                    from lox.data.fred import FredClient
                    _fred = FredClient(api_key=settings.FRED_API_KEY)
                    _auto = _fred.fetch_series("SUBLPDCLATRNQ", start_date=start_date, refresh=refresh)
                    result.metrics["auto_tightening"] = float(_auto.sort_values("date")["value"].iloc[-1]) if _auto is not None and not _auto.empty else None
                except Exception:
                    result.metrics["auto_tightening"] = None
            return result
        except Exception as e:
            logger.warning(f"Failed to build consumer regime: {e}")
            return None

    def _build_fiscal():
        try:
            from lox.fiscal.signals import build_fiscal_state as _build_fiscal_state
            from lox.fiscal.scoring import score_fiscal_regime as _score_fiscal

            fiscal_st = _build_fiscal_state(settings=settings, start_date="2011-01-01", refresh=refresh)
            fiscal_sc = _score_fiscal(fiscal_st.inputs)

            return RegimeResult(
                name=fiscal_sc.regime_name,
                label=fiscal_sc.regime_label,
                description=fiscal_sc.regime_description,
                score=fiscal_sc.fpi,
                domain="fiscal",
                tags=["risk_off"] if fiscal_sc.fpi >= 65 else [],
                metrics={
                    "Deficit 12m": f"${fiscal_st.inputs.deficit_12m / 1e6:.2f}T" if fiscal_st.inputs.deficit_12m else None,
                    "z Deficit": f"{fiscal_st.inputs.z_deficit_12m:+.2f}" if fiscal_st.inputs.z_deficit_12m is not None else None,
                    "TGA": f"${fiscal_st.inputs.tga_level / 1000:.0f}B" if fiscal_st.inputs.tga_level is not None else None,
                    "TGA 28d Chg": f"${fiscal_st.inputs.tga_chg_28d / 1000:+.0f}B" if fiscal_st.inputs.tga_chg_28d is not None else None,
                    "Interest YoY": f"{fiscal_st.inputs.interest_expense_yoy:+.1f}%" if fiscal_st.inputs.interest_expense_yoy is not None else None,
                    "Deficit/Receipts": f"{fiscal_st.inputs.deficit_pct_receipts * 100:.0f}%" if fiscal_st.inputs.deficit_pct_receipts is not None else None,
                    "Auction Tail": f"{fiscal_st.inputs.auction_tail_bps:+.1f}bp" if fiscal_st.inputs.auction_tail_bps is not None else None,
                    "Dealer Take": f"{fiscal_st.inputs.dealer_take_pct:.1f}%" if fiscal_st.inputs.dealer_take_pct is not None else None,
                    "Deficit Trend": f"{fiscal_st.inputs.deficit_trend_slope:+.2f}" if fiscal_st.inputs.deficit_trend_slope is not None else None,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to build fiscal regime: {e}")
            return None

    def _build_usd():
        try:
            from lox.usd.signals import build_usd_state
            from lox.usd.regime import classify_usd_regime_from_state

            usd_state = build_usd_state(settings=settings, start_date=start_date, refresh=refresh)
            usd_regime = classify_usd_regime_from_state(usd_state)

            inp = usd_state.inputs

            # Continuous 0-100 score from strength z-score
            # strength_score is a composite z-score: positive = strong $, negative = weak $
            # Map z from -2.5..+2.5 → 0..100  (strong dollar = higher stress for markets)
            _ss = inp.usd_strength_score if inp.usd_strength_score is not None else 0
            _usd_score = max(0, min(100, 50 + _ss * 20))

            return RegimeResult(
                name=usd_regime.name,
                label=usd_regime.label,
                description=usd_regime.description,
                score=_usd_score,
                domain="usd",
                tags=list(usd_regime.tags),
                metrics={
                    "DXY": f"{inp.usd_index_broad:.1f}" if inp.usd_index_broad is not None else None,
                    "20d Chg": f"{inp.usd_chg_20d_pct:+.1f}%" if inp.usd_chg_20d_pct is not None else None,
                    "60d Chg": f"{inp.usd_chg_60d_pct:+.1f}%" if inp.usd_chg_60d_pct is not None else None,
                    "USD z": f"{inp.z_usd_level:+.1f}" if inp.z_usd_level is not None else None,
                    "60d z": f"{inp.z_usd_chg_60d:+.1f}" if inp.z_usd_chg_60d is not None else None,
                    "Strength": f"{inp.usd_strength_score:+.2f}" if inp.usd_strength_score is not None else None,
                    "YoY Chg": f"{inp.usd_yoy_chg_pct:+.1f}%" if inp.usd_yoy_chg_pct is not None else None,
                    "200d MA Dist": f"{inp.usd_200d_ma_dist_pct:+.1f}%" if inp.usd_200d_ma_dist_pct is not None else None,
                    "90d RVol": f"{inp.usd_90d_rvol:.1f}%" if inp.usd_90d_rvol is not None else None,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to build USD regime: {e}")
            return None

    def _build_commodities():
        try:
            from lox.commodities.signals import build_commodities_state
            from lox.commodities.regime import classify_commodities_regime

            comm_state = build_commodities_state(settings=settings, start_date=start_date, refresh=refresh)
            comm_regime = classify_commodities_regime(comm_state.inputs)

            inp = comm_state.inputs
            _cp = inp.commodity_pressure_score if inp.commodity_pressure_score is not None else 0
            _comm_score = max(0, min(100, 50 + _cp * 15))
            return RegimeResult(
                name=comm_regime.name,
                label=comm_regime.label,
                description=comm_regime.description,
                score=_comm_score,
                domain="commodities",
                tags=list(comm_regime.tags) if hasattr(comm_regime, "tags") else [],
                metrics={
                    "Gold": f"${inp.gold:.0f}" if inp.gold is not None else None,
                    "Silver": f"${inp.silver:.2f}" if inp.silver is not None else None,
                    "WTI": f"${inp.wti:.1f}" if inp.wti is not None else None,
                    "Copper": f"${inp.copper:.1f}" if inp.copper is not None else None,
                    "Broad 60d": f"{inp.broad_ret_60d_pct:+.1f}%" if inp.broad_ret_60d_pct is not None else None,
                    "Gold 20d Ret": f"{inp.gold_ret_20d_pct:+.1f}%" if inp.gold_ret_20d_pct is not None else None,
                    "WTI 20d Ret": f"{inp.wti_ret_20d_pct:+.1f}%" if inp.wti_ret_20d_pct is not None else None,
                    "Copper 60d Ret": f"{inp.copper_ret_60d_pct:+.1f}%" if inp.copper_ret_60d_pct is not None else None,
                    "Pressure": f"{inp.commodity_pressure_score:+.2f}" if inp.commodity_pressure_score is not None else None,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to build commodities regime: {e}")
            return None

    # ── Submit all builders to thread pool ─────────────────────────────────
    builders = {
        "growth": _build_growth,
        "inflation": _build_inflation,
        "volatility": _build_volatility,
        "credit": _build_credit,
        "rates": _build_rates,
        "liquidity": _build_liquidity,
        "consumer": _build_consumer,
        "fiscal": _build_fiscal,
        "usd": _build_usd,
        "commodities": _build_commodities,
    }

    with ThreadPoolExecutor(max_workers=10, thread_name_prefix="regime") as pool:
        futures = {pool.submit(fn): name for name, fn in builders.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if result is not None:
                    setattr(state, name, result)
            except Exception as e:
                logger.warning(f"Regime builder '{name}' raised: {e}")

    _t_regimes = _time.monotonic()
    logger.info(f"[Perf] all regimes built in {(_t_regimes - _t_macro)*1000:.0f}ms (total {(_t_regimes - _t0)*1000:.0f}ms)")

    # ── Macro Quadrant (derived from Growth + Inflation) ──────────────────
    state.macro_quadrant = _compute_macro_quadrant(state.growth, state.inflation)

    # ═══════════════════════════════════════════════════════════════════════
    # Calculate weighted overall risk score
    # ═══════════════════════════════════════════════════════════════════════
    weighted_sum = 0.0
    weight_total = 0.0
    for domain in ALL_DOMAINS:
        regime = getattr(state, domain, None)
        if regime is not None:
            w = REGIME_WEIGHTS.get(regime.label, REGIME_WEIGHTS.get(domain.title(), 0))
            if w == 0:
                w = REGIME_WEIGHTS.get(domain.title(), 0.05)
            weighted_sum += regime.score * w
            weight_total += w

    if weight_total > 0:
        state.overall_risk_score = weighted_sum / weight_total
    else:
        state.overall_risk_score = 50.0

    if state.overall_risk_score >= 65:
        state.overall_category = "RISK-OFF"
    elif state.overall_risk_score >= 55:
        state.overall_category = "CAUTIOUS"
    elif state.overall_risk_score >= 45:
        state.overall_category = "NEUTRAL"
    elif state.overall_risk_score >= 35:
        state.overall_category = "RISK-ON"
    else:
        state.overall_category = "STRONG RISK-ON"

    # ── Save regime history ───────────────────────────────────────────────
    try:
        from lox.data.regime_history import save_regime_snapshot
        results_dict = {
            domain: getattr(state, domain)
            for domain in ALL_DOMAINS
            if getattr(state, domain) is not None
        }
        save_regime_snapshot(results_dict)
    except Exception:
        pass

    return state


def extract_ml_features(
    settings=None,
    start_date: str = "2020-01-01",
    refresh: bool = False,
) -> dict:
    """
    Extract flat ML feature dictionary from all regimes.

    This is the main entry point for ML training pipelines.
    """
    state = build_unified_regime_state(settings=settings, start_date=start_date, refresh=refresh)
    return state.to_feature_dict()
