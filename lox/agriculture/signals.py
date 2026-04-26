"""
Agriculture & food-inflation signal builder.

Tracks crop prices (corn, wheat, soybeans), input costs (natural gas,
diesel, fertilizer equities), and computes composite scores for
food-inflation regime classification.

Data sources:
  - FRED: Henry Hub natural gas (DHHNGSP), diesel (GASDESW),
          PPI fertilizers (WPU0652)
  - FMP/Alpaca: CORN, WEAT, SOYB, DBA ETFs; CF, NTR, MOS fertilizer equities
  - FMP COT: CFTC Commitments of Traders — net speculative positioning
  - USDA FAS PSD: WASDE supply/demand balances (stocks-to-use)
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from lox.agriculture.models import AgricultureInputs, AgricultureState
from lox.config import Settings
from lox.data.fred import FredClient
from lox.macro.transforms import (
    annualized_rate_from_levels,
    merge_series_daily,
    seasonal_zscore,
    yoy_from_index_level,
    zscore,
)

logger = logging.getLogger(__name__)

# ── FRED series ────────────────────────────────────────────────────────────
AG_FRED_SERIES: Dict[str, str] = {
    "NATGAS": "DHHNGSP",       # Henry Hub Natural Gas Spot Price (daily)
    "DIESEL": "GASDESW",       # US No 2 Diesel Retail Price (weekly)
    "PPI_FERT": "WPU0652",    # PPI: Agricultural chemicals & fertilizers (monthly)
}

# ── Food CPI / PPI FRED series (monthly, forward-filled onto daily grid) ──
FOOD_CPI_FRED_SERIES: Dict[str, str] = {
    "CPI_FOOD": "CPIUFDSL",           # CPI Food (all items), SA index
    "CPI_FOOD_HOME": "CUSR0000SAF11",  # Food at home (grocery), SA
    "CPI_FOOD_AWAY": "CUSR0000SEFV",   # Food away from home (restaurant), SA
    "CPI_CEREALS": "CUSR0000SAF111",   # Cereals & bakery, SA
    "CPI_MEATS": "CUSR0000SAF112",     # Meats/poultry/fish/eggs, SA
    "CPI_DAIRY": "CUSR0000SEFJ",       # Dairy, SA
    "CPI_FRUITS_VEG": "CUSR0000SAF113",  # Fruits & vegetables, SA
    "PPI_FOOD_MFG": "PCU311311",       # PPI Food manufacturing, SA
}

# ── Protein prices (FRED, monthly avg retail $/unit) ──────────────────────
PROTEIN_FRED_SERIES: Dict[str, str] = {
    "BEEF_PRICE": "APU0000703112",     # Avg price ground beef $/lb
    "CHICKEN_PRICE": "APU0000706111",  # Avg price chicken breast $/lb
    "EGG_PRICE": "APU0000708111",      # Avg price eggs $/dozen
}

# ── ETF / equity tickers (fetched via FMP or Alpaca) ──────────────────────
CROP_TICKERS = ("CORN", "WEAT", "SOYB", "DBA")
FERT_TICKERS = ("CF", "NTR", "MOS")
SOFT_TICKERS = ("CANE", "JO", "NIB")  # sugar, coffee, cocoa ETFs

# ── 2022 peaks for analog comparison ──────────────────────────────────────
PEAKS_2022 = {
    "corn": 29.00,    # CORN ETF peak Apr 2022
    "natgas": 8.81,   # Henry Hub peak Aug 2022
    "wheat": 12.75,   # WEAT ETF peak Mar 2022
}

# Z-score lookback (3 years of trading days)
ZSCORE_WINDOW = 252 * 3

# ── CFTC COT agriculture symbols (FMP naming) ─────────────────────────────
AG_COT_SYMBOLS: Dict[str, str] = {
    "CORN": "CORN",
    "WHEAT": "WHEAT",
    "SOYBEANS": "SOYBEANS",
}

AG_COT_ALTS: Dict[str, list[str]] = {
    "CORN": ["corn", "corn - chicago"],
    "WHEAT": ["wheat", "wheat-srw", "wheat - chicago", "wheat-hrw"],
    "SOYBEANS": ["soybeans", "soybean", "soybeans - chicago"],
}


def _fetch_etf_daily(
    settings: Settings,
    tickers: list[str],
    start: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch daily closes for ETFs/equities via Alpaca (primary) or FMP."""
    if settings.alpaca_data_key or settings.alpaca_api_key:
        try:
            from lox.data.market import fetch_equity_daily_closes
            px = fetch_equity_daily_closes(
                settings=settings, symbols=tickers, start=start, refresh=refresh,
            )
            return px.sort_index().ffill()
        except Exception as e:
            logger.warning("Alpaca daily fetch failed for %s: %s", tickers, e)
    return pd.DataFrame()


def _fetch_live_quotes(settings: Settings) -> Dict[str, float]:
    """Fetch live FMP quotes for crop + fertilizer tickers."""
    try:
        from lox.altdata.fmp import fetch_realtime_quotes
        return fetch_realtime_quotes(
            settings=settings,
            tickers=list(CROP_TICKERS + FERT_TICKERS + SOFT_TICKERS),
        )
    except Exception as e:
        logger.warning("FMP live quotes failed: %s", e)
        return {}


def _safe_pct_return(series: pd.Series, periods: int) -> pd.Series:
    shifted = series.shift(periods)
    return ((series / shifted) - 1.0) * 100.0


def _fert_basket(df: pd.DataFrame) -> pd.Series | None:
    """Equal-weight index of available fertilizer equities (CF, NTR, MOS).

    Each stock is normalized to its first non-NaN value so they contribute
    equally regardless of price level.
    """
    available = [t for t in FERT_TICKERS if t in df.columns]
    if not available:
        return None
    normalized = pd.DataFrame()
    for t in available:
        s = pd.to_numeric(df[t], errors="coerce")
        first_valid = s.first_valid_index()
        if first_valid is not None and s.loc[first_valid] != 0:
            normalized[t] = s / s.loc[first_valid] * 100.0
    if normalized.empty:
        return None
    return normalized.mean(axis=1)


def _safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def fetch_ag_cot_data(
    settings: Settings,
) -> tuple[Dict[str, float], Dict[str, float], str]:
    """Fetch CFTC COT net speculative positioning for agriculture commodities.

    Uses the same FMP /api/v4/commitment_of_traders_report endpoint as the
    positioning module, but filtered to corn, wheat, soybeans.

    Returns:
        (net_spec, z_scores, report_date)
    """
    if not settings.fmp_api_key:
        return {}, {}, ""

    from lox.altdata.cache import cache_path, read_cache, write_cache
    from datetime import timedelta
    import requests

    cache_key = "agriculture_cot_data"
    p = cache_path(cache_key)
    cached = read_cache(p, max_age=timedelta(hours=24))
    if isinstance(cached, dict) and "net_spec" in cached:
        return cached.get("net_spec", {}), cached.get("z_scores", {}), cached.get("date", "")

    net_spec: Dict[str, float] = {}
    z_scores: Dict[str, float] = {}
    report_date = ""

    try:
        url = "https://financialmodelingprep.com/api/v4/commitment_of_traders_report"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list) or not data:
            return {}, {}, ""

        for sym_code, fmp_name in AG_COT_SYMBOLS.items():
            alts = [fmp_name.lower()] + AG_COT_ALTS.get(sym_code, [])
            rows = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                market = str(
                    row.get("short_name") or row.get("market_and_exchange_names") or ""
                ).lower()
                if any(alt in market for alt in alts):
                    rows.append(row)

            if not rows:
                continue

            rows.sort(
                key=lambda r: str(r.get("date") or r.get("report_date") or ""),
                reverse=True,
            )
            latest = rows[0]
            rpt_date = str(latest.get("date") or latest.get("report_date") or "")[:10]
            if rpt_date and (not report_date or rpt_date > report_date):
                report_date = rpt_date

            nc_long = _safe_float(
                latest.get("noncomm_positions_long_all") or latest.get("noncommercial_long")
            )
            nc_short = _safe_float(
                latest.get("noncomm_positions_short_all") or latest.get("noncommercial_short")
            )
            if nc_long is not None and nc_short is not None:
                net = nc_long - nc_short
                net_spec[sym_code] = net

                historical_nets = []
                for r in rows[:52]:
                    nl = _safe_float(
                        r.get("noncomm_positions_long_all") or r.get("noncommercial_long")
                    )
                    ns = _safe_float(
                        r.get("noncomm_positions_short_all") or r.get("noncommercial_short")
                    )
                    if nl is not None and ns is not None:
                        historical_nets.append(nl - ns)

                if len(historical_nets) >= 10:
                    arr = np.array(historical_nets)
                    mean = float(np.mean(arr))
                    std = float(np.std(arr, ddof=1))
                    if std > 0:
                        z_scores[sym_code] = (net - mean) / std

        if net_spec:
            write_cache(p, {"net_spec": net_spec, "z_scores": z_scores, "date": report_date})

    except Exception as e:
        logger.warning("Failed to fetch ag COT data: %s", e)

    return net_spec, z_scores, report_date


def fetch_ag_wasde(settings: Settings) -> Dict[str, Any]:
    """Fetch WASDE supply/demand balances for corn, wheat, soybeans.

    Returns dict with keys like 'corn_stu_pct', 'wheat_ending_stocks', etc.
    """
    api_key = settings.usda_fas_api_key
    if not api_key:
        return {}

    try:
        from lox.data.usda import fetch_all_wasde
        balances = fetch_all_wasde(api_key)
    except Exception as e:
        logger.warning("WASDE fetch failed: %s", e)
        return {}

    result: Dict[str, Any] = {}
    for comm, bal in balances.items():
        prefix = "soy" if comm == "soybeans" else comm
        result[f"{prefix}_stu_pct"] = bal.stocks_to_use_pct
        result[f"{prefix}_ending_stocks"] = bal.ending_stocks
        result["market_year"] = bal.market_year
    return result


def fetch_ag_crop_reports(settings: Settings) -> dict[str, Any]:
    """Fetch NASS crop reports (plantings, progress, condition).

    Returns flat dict with keys like 'corn_planted_acres_m', 'corn_pct_planted', etc.
    """
    api_key = settings.usda_nass_api_key
    if not api_key:
        return {}

    try:
        from lox.data.usda_nass import fetch_all_crop_reports
        reports = fetch_all_crop_reports(api_key)
    except Exception as e:
        logger.warning("NASS crop reports fetch failed: %s", e)
        return {}

    result: Dict[str, Any] = {}

    for comm, planting in reports.plantings.items():
        prefix = "soy" if comm == "SOYBEANS" else comm.lower()
        if planting.planted_acres_m is not None:
            result[f"{prefix}_planted_acres_m"] = planting.planted_acres_m
        if planting.yoy_change_pct is not None:
            result[f"{prefix}_planted_yoy_pct"] = planting.yoy_change_pct

    # Progress — all 3 crops
    for comm, prog in reports.progress.items():
        prefix = "soy" if comm == "SOYBEANS" else comm.lower()
        if prog.pct_planted is not None:
            result[f"{prefix}_pct_planted"] = prog.pct_planted
        if prog.five_yr_avg_pct_planted is not None and prog.pct_planted is not None:
            result[f"{prefix}_pct_planted_vs_avg"] = prog.pct_planted - prog.five_yr_avg_pct_planted
        if prog.pct_emerged is not None:
            result[f"{prefix}_pct_emerged"] = prog.pct_emerged
            if prog.five_yr_avg_pct_emerged is not None:
                result[f"{prefix}_pct_emerged_vs_avg"] = prog.pct_emerged - prog.five_yr_avg_pct_emerged
        if prog.pct_silking is not None:
            result[f"{prefix}_pct_silking"] = prog.pct_silking
        if prog.week_ending:
            result["crop_report_week"] = prog.week_ending

    # Condition — all 3 crops
    for comm, cond in reports.condition.items():
        prefix = "soy" if comm == "SOYBEANS" else comm.lower()
        if cond.pct_good_excellent is not None:
            result[f"{prefix}_condition_ge"] = cond.pct_good_excellent
        if cond.five_yr_avg_ge is not None and cond.pct_good_excellent is not None:
            result[f"{prefix}_condition_ge_vs_avg"] = cond.pct_good_excellent - cond.five_yr_avg_ge

    # Composite metrics
    ge_vals = [v for k, v in result.items() if k.endswith("_condition_ge") and v is not None]
    if ge_vals:
        result["crop_condition_composite"] = sum(ge_vals) / len(ge_vals)

    delay_count = sum(
        1 for k, v in result.items()
        if k.endswith("_pct_planted_vs_avg") and v is not None and v < -5
    )
    result["planting_delay_count"] = delay_count

    return result


def build_agriculture_dataset(
    settings: Settings,
    start_date: str = "2018-01-01",
    refresh: bool = False,
) -> pd.DataFrame:
    """Build merged daily dataset of crop prices, input costs, fertilizer basket."""
    if not settings.FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY in environment / .env")

    fred = FredClient(api_key=settings.FRED_API_KEY)
    frames: Dict[str, pd.DataFrame] = {}

    # ── FRED series ────────────────────────────────────────────────────
    all_fred = {**AG_FRED_SERIES, **FOOD_CPI_FRED_SERIES, **PROTEIN_FRED_SERIES}
    for name, sid in all_fred.items():
        try:
            df = fred.fetch_series(series_id=sid, start_date=start_date, refresh=refresh)
            if df is not None and not df.empty:
                frames[name] = df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.warning("Failed to load FRED %s (%s): %s", name, sid, e)

    # ── ETF / equity daily closes (Alpaca) ─────────────────────────────
    all_tickers = list(CROP_TICKERS + FERT_TICKERS + SOFT_TICKERS)
    px = _fetch_etf_daily(settings, all_tickers, start_date, refresh)
    for t in all_tickers:
        if t in px.columns and px[t].dropna().shape[0] > 30:
            s = pd.to_numeric(px[t], errors="coerce").dropna()
            frames[t] = pd.DataFrame({"date": s.index, "value": s.values}).sort_values("date").reset_index(drop=True)

    if not frames:
        raise RuntimeError("No agriculture data loaded — check API keys")

    # ── Merge onto daily grid ──────────────────────────────────────────
    max_date = max(df["date"].max() for df in frames.values())
    base = pd.DataFrame({
        "date": pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(max_date), freq="D"),
    })
    merged = merge_series_daily(base, frames, ffill=True)

    # Align to business days for return calculations
    idx = merged.set_index("date")
    b = idx.asfreq("B").ffill()

    # ── Build fertilizer basket ────────────────────────────────────────
    fert = _fert_basket(b)
    if fert is not None:
        b["FERT_BASKET"] = fert

    # ── Returns ────────────────────────────────────────────────────────
    for col, periods_label in [
        ("CORN", [(20, "20D"), (60, "60D")]),
        ("WEAT", [(20, "20D")]),
        ("SOYB", [(20, "20D"), (60, "60D")]),
        ("NATGAS", [(20, "20D"), (60, "60D")]),
        ("DIESEL", [(60, "60D")]),
        ("FERT_BASKET", [(20, "20D"), (60, "60D")]),
    ]:
        if col not in b.columns:
            continue
        s = pd.to_numeric(b[col], errors="coerce")
        for periods, lbl in periods_label:
            b[f"{col}_RET_{lbl}_PCT"] = _safe_pct_return(s, periods)

    # PPI fertilizer: use level change since it's an index
    if "PPI_FERT" in b.columns:
        ppi = pd.to_numeric(b["PPI_FERT"], errors="coerce")
        b["PPI_FERT_RET_60D_PCT"] = _safe_pct_return(ppi, 60)

    # ── Soft commodity returns ──────────────────────────────────────────
    for soft in SOFT_TICKERS:
        if soft in b.columns:
            s = pd.to_numeric(b[soft], errors="coerce")
            b[f"{soft}_RET_20D_PCT"] = _safe_pct_return(s, 20)

    # ── CPI / PPI YoY and momentum ─────────────────────────────────────
    cpi_food_cols = ["CPI_FOOD", "CPI_FOOD_HOME", "CPI_FOOD_AWAY",
                     "CPI_CEREALS", "CPI_MEATS", "CPI_DAIRY", "CPI_FRUITS_VEG"]
    for col in cpi_food_cols:
        if col in b.columns:
            lvl = pd.to_numeric(b[col], errors="coerce")
            b[f"{col}_YOY"] = yoy_from_index_level(lvl)

    if "PPI_FOOD_MFG" in b.columns:
        lvl = pd.to_numeric(b["PPI_FOOD_MFG"], errors="coerce")
        b["PPI_FOOD_MFG_YOY"] = yoy_from_index_level(lvl)

    if "CPI_FOOD" in b.columns:
        lvl = pd.to_numeric(b["CPI_FOOD"], errors="coerce")
        b["CPI_FOOD_3M_ANN"] = annualized_rate_from_levels(lvl, months=3)
        b["CPI_FOOD_6M_ANN"] = annualized_rate_from_levels(lvl, months=6)

    # Acceleration: 3m annualized - YoY
    if "CPI_FOOD_3M_ANN" in b.columns and "CPI_FOOD_YOY" in b.columns:
        b["FOOD_ACCEL"] = b["CPI_FOOD_3M_ANN"] - b["CPI_FOOD_YOY"]

    # Pipeline spreads
    if "CPI_FOOD_HOME_YOY" in b.columns and "PPI_FOOD_MFG_YOY" in b.columns:
        b["FARM_TO_RETAIL_SPREAD"] = b["CPI_FOOD_HOME_YOY"] - b["PPI_FOOD_MFG_YOY"]
    if "CPI_FOOD_HOME_YOY" in b.columns and "CPI_FOOD_AWAY_YOY" in b.columns:
        b["GROCERY_RESTAURANT_GAP"] = b["CPI_FOOD_HOME_YOY"] - b["CPI_FOOD_AWAY_YOY"]

    # Protein YoY composite
    protein_yoys = []
    for pcol in ["BEEF_PRICE", "CHICKEN_PRICE", "EGG_PRICE"]:
        if pcol in b.columns:
            lvl = pd.to_numeric(b[pcol], errors="coerce")
            yoy = yoy_from_index_level(lvl)
            b[f"{pcol}_YOY"] = yoy
            protein_yoys.append(yoy)
    if protein_yoys:
        b["PROTEIN_YOY_AVG"] = pd.concat(protein_yoys, axis=1).mean(axis=1)

    # Breadth: count of 5 sub-categories with YoY above 3yr rolling median
    breadth_cols = ["CPI_CEREALS_YOY", "CPI_MEATS_YOY", "CPI_DAIRY_YOY",
                    "CPI_FRUITS_VEG_YOY", "CPI_FOOD_AWAY_YOY"]
    available_breadth = [c for c in breadth_cols if c in b.columns]
    if available_breadth:
        breadth_sum = pd.Series(0, index=b.index, dtype=float)
        for bc in available_breadth:
            s = pd.to_numeric(b[bc], errors="coerce")
            med = s.rolling(ZSCORE_WINDOW, min_periods=252).median()
            breadth_sum += (s > med).astype(float)
        b["FOOD_BREADTH_COUNT"] = breadth_sum
        b["FOOD_BREADTH_PCT"] = breadth_sum / len(available_breadth) * 100.0

    # ── Merge derived columns back to daily grid ───────────────────────
    derived_cols = [c for c in b.columns if c not in idx.columns]
    for c in derived_cols:
        idx[c] = b[c].reindex(idx.index, method="ffill")
    merged = idx.reset_index()

    # ── Z-scores ───────────────────────────────────────────────────────
    win = ZSCORE_WINDOW
    z_map = {
        "Z_CORN_RET_20D": "CORN_RET_20D_PCT",
        "Z_CORN_RET_60D": "CORN_RET_60D_PCT",
        "Z_WHEAT_RET_20D": "WEAT_RET_20D_PCT",
        "Z_NATGAS_RET_20D": "NATGAS_RET_20D_PCT",
        "Z_NATGAS_RET_60D": "NATGAS_RET_60D_PCT",
        "Z_DIESEL_RET_60D": "DIESEL_RET_60D_PCT",
        "Z_FERT_BASKET_RET_20D": "FERT_BASKET_RET_20D_PCT",
        "Z_FERT_BASKET_RET_60D": "FERT_BASKET_RET_60D_PCT",
        "Z_PPI_FERT_60D": "PPI_FERT_RET_60D_PCT",
        "Z_CPI_FOOD_YOY": "CPI_FOOD_YOY",
        "Z_PPI_FOOD_MFG_YOY": "PPI_FOOD_MFG_YOY",
        "Z_PROTEIN_YOY": "PROTEIN_YOY_AVG",
    }
    for z_col, src_col in z_map.items():
        if src_col in merged.columns:
            merged[z_col] = zscore(pd.to_numeric(merged[src_col], errors="coerce"), window=win)

    # Soft commodity composite z-score
    soft_rets = [f"{s}_RET_20D_PCT" for s in SOFT_TICKERS if f"{s}_RET_20D_PCT" in merged.columns]
    if soft_rets:
        merged["SOFT_RET_20D"] = merged[soft_rets].mean(axis=1)
        merged["Z_SOFT_RET_20D"] = zscore(pd.to_numeric(merged["SOFT_RET_20D"], errors="coerce"), window=win)

    # Food breadth normalized z-score (for composite scoring)
    if "FOOD_BREADTH_PCT" in merged.columns:
        merged["Z_FOOD_BREADTH"] = zscore(pd.to_numeric(merged["FOOD_BREADTH_PCT"], errors="coerce"), window=win)

    # Pipeline (PPI-to-CPI) z-score
    if "FARM_TO_RETAIL_SPREAD" in merged.columns:
        merged["Z_PIPELINE"] = zscore(pd.to_numeric(merged["FARM_TO_RETAIL_SPREAD"], errors="coerce"), window=win)

    # ── Composite scores ───────────────────────────────────────────────
    # Input cost score: how much are ag production inputs rising?
    input_parts = []
    input_weights = []
    if "Z_NATGAS_RET_20D" in merged.columns:
        input_parts.append(merged["Z_NATGAS_RET_20D"] * 0.40)
        input_weights.append(0.40)
    if "Z_FERT_BASKET_RET_20D" in merged.columns:
        input_parts.append(merged["Z_FERT_BASKET_RET_20D"] * 0.30)
        input_weights.append(0.30)
    if "Z_DIESEL_RET_60D" in merged.columns:
        input_parts.append(merged["Z_DIESEL_RET_60D"] * 0.15)
        input_weights.append(0.15)
    if "Z_PPI_FERT_60D" in merged.columns:
        input_parts.append(merged["Z_PPI_FERT_60D"] * 0.15)
        input_weights.append(0.15)
    if input_parts:
        total_w = sum(input_weights)
        merged["INPUT_COST_SCORE"] = sum(input_parts) / total_w

    # Crop momentum score: are crop prices themselves rising?
    crop_parts = []
    crop_weights = []
    if "Z_CORN_RET_20D" in merged.columns:
        crop_parts.append(merged["Z_CORN_RET_20D"] * 0.45)
        crop_weights.append(0.45)
    if "Z_WHEAT_RET_20D" in merged.columns:
        crop_parts.append(merged["Z_WHEAT_RET_20D"] * 0.35)
        crop_weights.append(0.35)
    if "Z_CORN_RET_60D" in merged.columns:
        crop_parts.append(merged["Z_CORN_RET_60D"] * 0.20)
        crop_weights.append(0.20)
    if crop_parts:
        total_w = sum(crop_weights)
        merged["CROP_MOMENTUM_SCORE"] = sum(crop_parts) / total_w

    # Food inflation score: 6-component weighted z-score when CPI data available,
    # falls back to 2-component (input + crop) blend when CPI data unavailable.
    has_cpi = "Z_CPI_FOOD_YOY" in merged.columns
    if has_cpi:
        fi_parts = []
        fi_weights = []
        if "Z_CPI_FOOD_YOY" in merged.columns:
            fi_parts.append(merged["Z_CPI_FOOD_YOY"] * 0.25)
            fi_weights.append(0.25)
        if "Z_PIPELINE" in merged.columns:
            fi_parts.append(merged["Z_PIPELINE"] * 0.15)
            fi_weights.append(0.15)
        if "INPUT_COST_SCORE" in merged.columns:
            fi_parts.append(merged["INPUT_COST_SCORE"] * 0.20)
            fi_weights.append(0.20)
        if "CROP_MOMENTUM_SCORE" in merged.columns:
            fi_parts.append(merged["CROP_MOMENTUM_SCORE"] * 0.15)
            fi_weights.append(0.15)
        if "Z_FOOD_BREADTH" in merged.columns:
            fi_parts.append(merged["Z_FOOD_BREADTH"] * 0.15)
            fi_weights.append(0.15)
        if "Z_PROTEIN_YOY" in merged.columns:
            fi_parts.append(merged["Z_PROTEIN_YOY"] * 0.10)
            fi_weights.append(0.10)
        if fi_parts:
            total_w = sum(fi_weights)
            merged["FOOD_INFLATION_SCORE"] = sum(fi_parts) / total_w
    else:
        # Fallback: original 2-component blend
        if "INPUT_COST_SCORE" in merged.columns and "CROP_MOMENTUM_SCORE" in merged.columns:
            merged["FOOD_INFLATION_SCORE"] = 0.55 * merged["INPUT_COST_SCORE"] + 0.45 * merged["CROP_MOMENTUM_SCORE"]
        elif "INPUT_COST_SCORE" in merged.columns:
            merged["FOOD_INFLATION_SCORE"] = merged["INPUT_COST_SCORE"]
        elif "CROP_MOMENTUM_SCORE" in merged.columns:
            merged["FOOD_INFLATION_SCORE"] = merged["CROP_MOMENTUM_SCORE"]

    # ── Divergence signals ─────────────────────────────────────────────
    if "Z_FERT_BASKET_RET_60D" in merged.columns and "Z_CORN_RET_60D" in merged.columns:
        merged["FERT_CORN_DIVERGENCE"] = merged["Z_FERT_BASKET_RET_60D"] - merged["Z_CORN_RET_60D"]

    if "NATGAS" in merged.columns and "CORN" in merged.columns:
        ng = pd.to_numeric(merged["NATGAS"], errors="coerce")
        corn = pd.to_numeric(merged["CORN"], errors="coerce")
        ratio = ng / corn.replace(0, float("nan"))
        merged["NATGAS_CORN_RATIO"] = ratio
        merged["Z_NATGAS_CORN_RATIO"] = zscore(ratio, window=win)

    # ── Flags ──────────────────────────────────────────────────────────
    if "INPUT_COST_SCORE" in merged.columns:
        merged["INPUT_SHOCK"] = merged["INPUT_COST_SCORE"] > 2.0
    if "Z_NATGAS_RET_20D" in merged.columns and "INPUT_COST_SCORE" in merged.columns:
        merged["INPUT_SHOCK"] = merged.get("INPUT_SHOCK", False) | (merged["Z_NATGAS_RET_20D"] > 2.5)

    if "CROP_MOMENTUM_SCORE" in merged.columns:
        merged["CROP_SURGE"] = merged["CROP_MOMENTUM_SCORE"] > 1.5

    if "FERT_CORN_DIVERGENCE" in merged.columns:
        merged["COST_PASS_THROUGH_LAG"] = merged["FERT_CORN_DIVERGENCE"] > 1.0

    # New food CPI flags
    if "FOOD_BREADTH_COUNT" in merged.columns and "CPI_FOOD_YOY" in merged.columns:
        merged["BROAD_FOOD_INFLATION"] = (merged["FOOD_BREADTH_COUNT"] >= 4) & (merged["CPI_FOOD_YOY"] > 4.0)
    if "FOOD_ACCEL" in merged.columns:
        merged["FOOD_ACCEL_FLAG"] = merged["FOOD_ACCEL"] > 1.5
    if "GROCERY_RESTAURANT_GAP" in merged.columns:
        merged["GROCERY_SHOCK"] = merged["GROCERY_RESTAURANT_GAP"] > 3.0
    if "Z_PROTEIN_YOY" in merged.columns:
        merged["PROTEIN_SPIKE"] = merged["Z_PROTEIN_YOY"] > 2.0

    # ── Seasonal z-scores ───────────────────────────────────────────────
    # Compare returns to same-calendar-period history to strip planting/
    # harvest seasonality. Requires DatetimeIndex.
    if "date" in merged.columns:
        ts = merged.set_index("date")
    else:
        ts = merged
    sz_map = {
        "SZ_CORN_RET_20D": "CORN_RET_20D_PCT",
        "SZ_WHEAT_RET_20D": "WEAT_RET_20D_PCT",
        "SZ_NATGAS_RET_20D": "NATGAS_RET_20D_PCT",
    }
    for sz_col, src_col in sz_map.items():
        if src_col in ts.columns:
            merged[sz_col] = seasonal_zscore(
                pd.to_numeric(ts[src_col], errors="coerce"),
                min_years=3,
                bin_days=21,
            ).values if isinstance(ts.index, pd.DatetimeIndex) else np.nan

    return merged


def build_agriculture_state(
    settings: Settings,
    start_date: str = "2018-01-01",
    refresh: bool = False,
) -> AgricultureState:
    """Build current agriculture state from latest data point."""
    df = build_agriculture_dataset(settings=settings, start_date=start_date, refresh=refresh)

    # Find last row with at least one core series populated
    core_cols = [c for c in ["CORN", "NATGAS", "WEAT"] if c in df.columns]
    if not core_cols:
        raise RuntimeError("No core agriculture data available")
    last = df.dropna(subset=core_cols[:1]).iloc[-1]

    def _f(col):
        return float(last[col]) if col in last and pd.notna(last.get(col)) else None

    def _b(col):
        return bool(last[col]) if col in last and pd.notna(last.get(col)) else None

    # 2022 peak comparisons
    corn_val = _f("CORN")
    natgas_val = _f("NATGAS")
    fert_val = _f("FERT_BASKET")

    # ── CFTC COT positioning ──────────────────────────────────────────
    cot_net, cot_z, cot_date = fetch_ag_cot_data(settings)

    # ── WASDE supply/demand ────────────────────────────────────────────
    wasde = fetch_ag_wasde(settings)

    # ── NASS crop reports (plantings, progress, condition) ────────────
    nass = fetch_ag_crop_reports(settings)

    inp = AgricultureInputs(
        corn=corn_val,
        wheat=_f("WEAT"),
        soybeans=_f("SOYB"),
        ag_broad=_f("DBA"),
        natgas=natgas_val,
        diesel=_f("DIESEL"),
        ppi_fertilizer=_f("PPI_FERT"),
        fert_basket_level=fert_val,
        corn_ret_20d_pct=_f("CORN_RET_20D_PCT"),
        corn_ret_60d_pct=_f("CORN_RET_60D_PCT"),
        wheat_ret_20d_pct=_f("WEAT_RET_20D_PCT"),
        natgas_ret_20d_pct=_f("NATGAS_RET_20D_PCT"),
        natgas_ret_60d_pct=_f("NATGAS_RET_60D_PCT"),
        diesel_ret_60d_pct=_f("DIESEL_RET_60D_PCT"),
        fert_basket_ret_20d_pct=_f("FERT_BASKET_RET_20D_PCT"),
        fert_basket_ret_60d_pct=_f("FERT_BASKET_RET_60D_PCT"),
        z_corn_ret_20d=_f("Z_CORN_RET_20D"),
        z_corn_ret_60d=_f("Z_CORN_RET_60D"),
        z_wheat_ret_20d=_f("Z_WHEAT_RET_20D"),
        z_natgas_ret_20d=_f("Z_NATGAS_RET_20D"),
        z_natgas_ret_60d=_f("Z_NATGAS_RET_60D"),
        z_diesel_ret_60d=_f("Z_DIESEL_RET_60D"),
        z_fert_basket_ret_20d=_f("Z_FERT_BASKET_RET_20D"),
        z_fert_basket_ret_60d=_f("Z_FERT_BASKET_RET_60D"),
        z_ppi_fert_60d=_f("Z_PPI_FERT_60D"),
        sz_corn_ret_20d=_f("SZ_CORN_RET_20D"),
        sz_wheat_ret_20d=_f("SZ_WHEAT_RET_20D"),
        sz_natgas_ret_20d=_f("SZ_NATGAS_RET_20D"),
        input_cost_score=_f("INPUT_COST_SCORE"),
        crop_momentum_score=_f("CROP_MOMENTUM_SCORE"),
        food_inflation_score=_f("FOOD_INFLATION_SCORE"),
        # CPI Food Components
        cpi_food_yoy=_f("CPI_FOOD_YOY"),
        cpi_food_home_yoy=_f("CPI_FOOD_HOME_YOY"),
        cpi_food_away_yoy=_f("CPI_FOOD_AWAY_YOY"),
        cpi_cereals_yoy=_f("CPI_CEREALS_YOY"),
        cpi_meats_yoy=_f("CPI_MEATS_YOY"),
        cpi_dairy_yoy=_f("CPI_DAIRY_YOY"),
        cpi_fruits_veg_yoy=_f("CPI_FRUITS_VEG_YOY"),
        # CPI Food Momentum
        cpi_food_3m_ann=_f("CPI_FOOD_3M_ANN"),
        cpi_food_6m_ann=_f("CPI_FOOD_6M_ANN"),
        cpi_food_accel=_f("FOOD_ACCEL"),
        # PPI Food
        ppi_food_mfg_yoy=_f("PPI_FOOD_MFG_YOY"),
        # Pipeline
        farm_to_retail_spread=_f("FARM_TO_RETAIL_SPREAD"),
        grocery_restaurant_gap=_f("GROCERY_RESTAURANT_GAP"),
        # Breadth
        food_breadth_count=int(last["FOOD_BREADTH_COUNT"]) if "FOOD_BREADTH_COUNT" in last and pd.notna(last.get("FOOD_BREADTH_COUNT")) else None,
        food_breadth_pct=_f("FOOD_BREADTH_PCT"),
        # Protein
        beef_price=_f("BEEF_PRICE"),
        chicken_price=_f("CHICKEN_PRICE"),
        egg_price=_f("EGG_PRICE"),
        protein_yoy_avg=_f("PROTEIN_YOY_AVG"),
        protein_z=_f("Z_PROTEIN_YOY"),
        # Soft commodities
        sugar_price=_f("CANE"),
        coffee_price=_f("JO"),
        cocoa_price=_f("NIB"),
        soft_ret_20d=_f("SOFT_RET_20D"),
        soft_z=_f("Z_SOFT_RET_20D"),
        # Divergence / cross-signals
        fert_corn_divergence=_f("FERT_CORN_DIVERGENCE"),
        natgas_corn_ratio_z=_f("Z_NATGAS_CORN_RATIO"),
        input_shock=_b("INPUT_SHOCK"),
        crop_surge=_b("CROP_SURGE"),
        cost_pass_through_lag=_b("COST_PASS_THROUGH_LAG"),
        broad_food_inflation=_b("BROAD_FOOD_INFLATION"),
        food_accel_flag=_b("FOOD_ACCEL_FLAG"),
        grocery_shock=_b("GROCERY_SHOCK"),
        protein_spike=_b("PROTEIN_SPIKE"),
        corn_pct_of_2022_peak=(corn_val / PEAKS_2022["corn"] * 100) if corn_val else None,
        natgas_pct_of_2022_peak=(natgas_val / PEAKS_2022["natgas"] * 100) if natgas_val else None,
        fert_pct_of_2022_peak=None,
        # COT positioning
        cot_corn_net=cot_net.get("CORN"),
        cot_corn_z=cot_z.get("CORN"),
        cot_wheat_net=cot_net.get("WHEAT"),
        cot_wheat_z=cot_z.get("WHEAT"),
        cot_soybeans_net=cot_net.get("SOYBEANS"),
        cot_soybeans_z=cot_z.get("SOYBEANS"),
        cot_date=cot_date or None,
        # WASDE supply/demand
        wasde_corn_stu_pct=wasde.get("corn_stu_pct"),
        wasde_wheat_stu_pct=wasde.get("wheat_stu_pct"),
        wasde_soy_stu_pct=wasde.get("soy_stu_pct"),
        wasde_corn_ending_stocks=wasde.get("corn_ending_stocks"),
        wasde_wheat_ending_stocks=wasde.get("wheat_ending_stocks"),
        wasde_soy_ending_stocks=wasde.get("soy_ending_stocks"),
        wasde_market_year=wasde.get("market_year"),
        # NASS crop reports
        corn_planted_acres_m=nass.get("corn_planted_acres_m"),
        corn_planted_yoy_pct=nass.get("corn_planted_yoy_pct"),
        soy_planted_acres_m=nass.get("soy_planted_acres_m"),
        soy_planted_yoy_pct=nass.get("soy_planted_yoy_pct"),
        wheat_planted_acres_m=nass.get("wheat_planted_acres_m"),
        wheat_planted_yoy_pct=nass.get("wheat_planted_yoy_pct"),
        corn_pct_planted=nass.get("corn_pct_planted"),
        corn_pct_planted_vs_avg=nass.get("corn_pct_planted_vs_avg"),
        corn_condition_ge=nass.get("corn_condition_ge"),
        corn_condition_ge_vs_avg=nass.get("corn_condition_ge_vs_avg"),
        crop_report_week=nass.get("crop_report_week"),
        # Soybean progress & condition
        soy_pct_planted=nass.get("soy_pct_planted"),
        soy_pct_planted_vs_avg=nass.get("soy_pct_planted_vs_avg"),
        soy_pct_emerged=nass.get("soy_pct_emerged"),
        soy_condition_ge=nass.get("soy_condition_ge"),
        soy_condition_ge_vs_avg=nass.get("soy_condition_ge_vs_avg"),
        # Wheat progress & condition
        wheat_pct_planted=nass.get("wheat_pct_planted"),
        wheat_pct_planted_vs_avg=nass.get("wheat_pct_planted_vs_avg"),
        wheat_condition_ge=nass.get("wheat_condition_ge"),
        wheat_condition_ge_vs_avg=nass.get("wheat_condition_ge_vs_avg"),
        # Corn growth stages
        corn_pct_emerged=nass.get("corn_pct_emerged"),
        corn_pct_emerged_vs_avg=nass.get("corn_pct_emerged_vs_avg"),
        corn_pct_silking=nass.get("corn_pct_silking"),
        # Aggregate crop health
        crop_condition_composite=nass.get("crop_condition_composite"),
        planting_delay_count=nass.get("planting_delay_count"),
        components={
            "zscore_window_days": float(ZSCORE_WINDOW),
            "natgas_input_weight": 0.40,
            "fert_input_weight": 0.30,
            "diesel_input_weight": 0.15,
            "ppi_input_weight": 0.15,
            "corn_crop_weight": 0.45,
            "wheat_crop_weight": 0.35,
            "corn_60d_crop_weight": 0.20,
        },
    )

    return AgricultureState(
        asof=str(pd.to_datetime(last["date"]).date()),
        start_date=start_date,
        inputs=inp,
        notes=(
            "Agriculture regime: crop prices + input costs z-scored vs 3yr history. "
            "Enhanced with CFTC COT positioning, WASDE supply/demand, seasonal z-scores, "
            "and NASS crop reports (prospective plantings, crop progress, crop condition)."
        ),
    )
