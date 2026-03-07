from __future__ import annotations

from datetime import date, timedelta

import typer
from rich import print
from rich.panel import Panel

from lox.config import load_settings
from lox.fiscal.regime import classify_fiscal_regime, classify_fiscal_regime_skeleton
from lox.fiscal.mc_calibration import calibrate_fiscal_mc
from lox.fiscal.scoring import score_fiscal_regime
from lox.fiscal.signals import build_fiscal_deficit_page_data, build_fiscal_state
from lox.utils.formatting import fmt_usd_from_millions


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (module-level for reuse)
# ─────────────────────────────────────────────────────────────────────────────

def _band(
    x: float | None,
    *,
    low: float,
    high: float,
    low_label: str,
    mid_label: str,
    high_label: str,
    fmt: str = "{:.2f}",
    units: str = "",
) -> str:
    if not isinstance(x, (int, float)):
        return "n/a"
    v = float(x)
    label = mid_label
    if v <= low:
        label = low_label
    elif v >= high:
        label = high_label
    u = f" {units}".rstrip()
    return f"{fmt.format(v)}{u} → {label}"


def _deficit_level_ctx(deficit_pct_gdp: float | None) -> str:
    return _band(
        deficit_pct_gdp,
        low=3.0,
        high=6.0,
        low_label="Contractionary: private sector starved of NFA",
        mid_label="Moderate private sector NFA injection",
        high_label="Strong NFA injection to private sector",
        fmt="{:.1f}",
        units="% GDP",
    )


def _deficit_impulse_ctx(impulse_pct_gdp: float | None) -> str:
    return _band(
        impulse_pct_gdp,
        low=-0.75,
        high=0.75,
        low_label="Fiscal drag: private surplus shrinking (bearish)",
        mid_label="Neutral fiscal impulse",
        high_label="Fiscal thrust: private surplus growing (supportive)",
        fmt="{:+.2f}",
        units="% GDP",
    )


def _duration_share_ctx(long_share: float | None) -> str:
    if not isinstance(long_share, (int, float)):
        return "n/a"
    pct = 100.0 * float(long_share)
    label = "Balanced"
    if pct >= 40.0:
        label = "Long-tilted → stress watch"
    elif pct <= 25.0:
        label = "Bill/coupon-tilted (less duration risk)"
    return f"{pct:.1f}% → {label}"


def _tga_z_ctx(z: float | None) -> str:
    if not isinstance(z, (int, float)):
        return "n/a"
    v = float(z)
    if abs(v) < 0.75:
        label = "Neutral / normal"
    elif v < 0:
        label = "NFA injection (reserves up, supportive)"
    else:
        label = "Private sector NFA drain (reserves down)"
    return f"{v:+.2f} → {label}"


def _tga_level_ctx(z_level: float | None) -> str:
    if not isinstance(z_level, (int, float)):
        return "n/a"
    v = float(z_level)
    if v <= -0.75:
        label = "Low → reserves elevated (supportive)"
    elif v >= 0.75:
        label = "High → draining private sector reserves"
    else:
        label = "Normal vs recent history"
    return f"{v:+.2f} → {label}"


def _private_balance_ctx(pct_gdp: float | None) -> str:
    if not isinstance(pct_gdp, (int, float)):
        return "n/a"
    v = float(pct_gdp)
    if v <= -2.0:
        label = "Deep private deficit → unsustainable, bearish"
    elif v <= 0.0:
        label = "Private sector deficit → fragile"
    elif v <= 3.0:
        label = "Modest private surplus"
    else:
        label = "Strong private surplus → supportive"
    return f"{v:+.1f}% GDP → {label}"


def _private_impulse_ctx(impulse: float | None) -> str:
    if not isinstance(impulse, (int, float)):
        return "n/a"
    v = float(impulse)
    if v <= -1.0:
        label = "Sharp drag on private sector (bearish)"
    elif v <= -0.25:
        label = "Private surplus shrinking (watch)"
    elif v <= 0.25:
        label = "Stable"
    elif v <= 1.0:
        label = "Private surplus growing (supportive)"
    else:
        label = "Strong fiscal thrust to private sector"
    return f"{v:+.2f}% GDP → {label}"


def _auction_tail_ctx(tail_bps: float | None, *, is_proxy: bool = True) -> str:
    if not isinstance(tail_bps, (int, float)):
        return "n/a"
    v = float(tail_bps)
    if v < 1.0:
        label = "Through / strong"
    elif v < 3.0:
        label = "Normal"
    elif v < 5.0:
        label = "Elevated → watch"
    else:
        label = "Wide tail → stress"
    suffix = " (proxy: high−median)" if is_proxy else ""
    return f"{v:.1f}bp → {label}{suffix}"


def _dealer_take_ctx(pct: float | None) -> str:
    if not isinstance(pct, (int, float)):
        return "n/a"
    v = float(pct)
    if v < 10.0:
        label = "Minimal dealer absorption"
    elif v < 20.0:
        label = "Normal intermediation"
    elif v < 35.0:
        label = "Elevated → watch"
    else:
        label = "Dealers forced buyers → stress"
    return f"{v:.1f}% → {label}"


def _btc_ctx(btc: float | None) -> str:
    if not isinstance(btc, (int, float)):
        return "n/a"
    v = float(btc)
    if v >= 2.8:
        label = "Strong"
    elif v >= 2.3:
        label = "Normal"
    elif v >= 2.0:
        label = "Soft → watch"
    else:
        label = "Weak → stress"
    return f"{v:.2f}x → {label}"


# ─────────────────────────────────────────────────────────────────────────────
# Upcoming long-end auction schedule
# ─────────────────────────────────────────────────────────────────────────────

_LONG_TENOR_PATTERNS = {
    "10-Year": ("Note", "10-Year"),
    "20-Year": ("Bond", "20-Year"),
    "30-Year": ("Bond", "30-Year"),
}


def _fetch_upcoming_from_treasurydirect() -> list[dict]:
    """Best-effort fetch of announced-but-not-yet-auctioned securities from TreasuryDirect."""
    import json
    import urllib.request

    url = "https://www.treasurydirect.gov/TA_WS/securities/upcoming?format=json"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
    except Exception:
        return []

    today = date.today()
    results: list[dict] = []
    for item in data:
        auction_str = (item.get("auctionDate") or "")[:10]
        sec_type = item.get("securityType") or item.get("type") or ""
        sec_term = item.get("securityTerm") or item.get("term") or ""
        if not auction_str:
            continue
        try:
            auction_dt = date.fromisoformat(auction_str)
        except Exception:
            continue
        if auction_dt < today:
            continue
        for tenor, (type_match, term_match) in _LONG_TENOR_PATTERNS.items():
            if term_match.lower() in sec_term.lower() and type_match.lower() in sec_type.lower():
                offering_raw = item.get("offeringAmount") or ""
                offering_b = None
                try:
                    offering_b = float(offering_raw) / 1_000_000_000
                except (ValueError, TypeError):
                    pass
                results.append({
                    "tenor": tenor,
                    "date": auction_dt,
                    "offering_b": offering_b,
                    "source": "announced",
                    "reopening": (item.get("reopening") or "").strip().lower() == "yes",
                })
    return results


_DEFAULT_CADENCE_DAYS = {"10-Year": 28, "20-Year": 91, "30-Year": 28}


def _estimate_from_cadence(long_auctions: list[dict]) -> list[dict]:
    """
    For each long-end tenor not yet announced, estimate the next auction date
    from the observed cadence in recent auctions (or a known default cadence).
    """
    from collections import defaultdict

    by_tenor: dict[str, list[date]] = defaultdict(list)
    for ra in long_auctions:
        term = ra.get("term", "")
        d = ra.get("date", "")
        if term not in _LONG_TENOR_PATTERNS or not d:
            continue
        try:
            by_tenor[term].append(date.fromisoformat(str(d)))
        except Exception:
            continue

    today = date.today()
    estimates: list[dict] = []
    for tenor in _LONG_TENOR_PATTERNS:
        dates = sorted(by_tenor.get(tenor, []))
        if not dates:
            continue
        if len(dates) >= 2:
            gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            meaningful_gaps = [g for g in gaps if g >= 10] or gaps
            cadence = sorted(meaningful_gaps)[len(meaningful_gaps) // 2]
        else:
            cadence = _DEFAULT_CADENCE_DAYS.get(tenor, 28)
        last = dates[-1]
        est = last + timedelta(days=cadence)
        while est <= today:
            est += timedelta(days=cadence)
        estimates.append({
            "tenor": tenor,
            "date": est,
            "offering_b": None,
            "source": "estimated",
            "reopening": False,
        })
    return estimates


def _get_upcoming_long_end(long_auctions: list[dict]) -> list[dict]:
    """
    Return upcoming long-end auctions: announced ones from TreasuryDirect,
    plus cadence-estimated ones for tenors not yet announced.
    """
    announced = _fetch_upcoming_from_treasurydirect()
    announced_tenors = {a["tenor"] for a in announced}

    estimated = _estimate_from_cadence(long_auctions)
    # Only include estimates for tenors that haven't been announced
    for est in estimated:
        if est["tenor"] not in announced_tenors:
            announced.append(est)

    announced.sort(key=lambda x: x["date"])
    return announced


# ─────────────────────────────────────────────────────────────────────────────
# Auction trend sparklines
# ─────────────────────────────────────────────────────────────────────────────

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float]) -> str:
    """Render a list of floats as a Unicode sparkline."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0
    return "".join(_SPARK_CHARS[min(len(_SPARK_CHARS) - 1, int((v - lo) / span * (len(_SPARK_CHARS) - 1)))] for v in values)


def _trend_label(values: list[float], higher_is_worse: bool = True) -> str:
    """Compare the average of the recent half vs the older half."""
    if len(values) < 4:
        return ""
    mid = len(values) // 2
    old_avg = sum(values[:mid]) / mid
    new_avg = sum(values[mid:]) / (len(values) - mid)
    delta = new_avg - old_avg
    threshold = (max(values) - min(values)) * 0.1 if max(values) != min(values) else 0.01
    if abs(delta) < threshold:
        return "[dim]stable[/dim]"
    if higher_is_worse:
        return "[red]worsening[/red]" if delta > 0 else "[green]improving[/green]"
    return "[green]improving[/green]" if delta > 0 else "[red]weakening[/red]"


def _show_auction_trend(console, long_auctions: list[dict]) -> None:
    """Print a sparkline trend summary for long-end auctions."""
    # Reverse so oldest is first (sparkline reads left-to-right = old-to-new)
    ordered = list(reversed(long_auctions))

    tails = [float(a["tail_bps"]) for a in ordered if isinstance(a.get("tail_bps"), (int, float))]
    btcs = [float(a["btc"]) for a in ordered if isinstance(a.get("btc"), (int, float))]
    dealers = [float(a["dealer_take_pct"]) for a in ordered if isinstance(a.get("dealer_take_pct"), (int, float))]

    parts: list[str] = []
    if len(tails) >= 4:
        parts.append(f"Tail {_sparkline(tails)} ({_trend_label(tails, higher_is_worse=True)})")
    if len(btcs) >= 4:
        parts.append(f"BTC {_sparkline(btcs)} ({_trend_label(btcs, higher_is_worse=False)})")
    if len(dealers) >= 4:
        parts.append(f"Dealer {_sparkline(dealers)} ({_trend_label(dealers, higher_is_worse=True)})")

    if parts:
        n = len(tails) or len(btcs) or len(dealers)
        console.print(f"\n[dim]Long-end trend (last {n}):[/dim]  {'  │  '.join(parts)}")


# ─────────────────────────────────────────────────────────────────────────────
# Core implementation (callable directly)
# ─────────────────────────────────────────────────────────────────────────────

def _run_fiscal_snapshot(
    lookback_years: int = 5,
    refresh: bool = False,
    full: bool = False,
    llm: bool = False,
    ticker: str = "",
    features: bool = False,
    json_out: bool = False,
    delta: str = "",
    alert: bool = False,
    calendar: bool = False,
    trades: bool = False,
):
    """Shared implementation for fiscal snapshot."""
    from rich.console import Console
    from lox.cli_commands.shared.regime_display import render_regime_panel
    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        show_alert_output, show_calendar_output, show_trades_output,
    )
    
    console = Console()
    settings = load_settings()

    if full:
        # Full snapshot path - raw dataclass output for debugging
        state = build_fiscal_state(settings=settings, start_date="2011-01-01", refresh=refresh)
        print(state)
        regime = classify_fiscal_regime(state.inputs)
        print("\nFISCAL REGIME")
        print(regime)
        return

    # Standard Panel output
    d = build_fiscal_deficit_page_data(settings=settings, lookback_years=lookback_years, refresh=refresh)

    deficit_12m = float(d["deficit_12m"])
    # Format into $B/$T for readability
    dollars = deficit_12m * 1_000_000.0
    if abs(dollars) >= 1_000_000_000_000:
        disp = f"${dollars/1_000_000_000_000:,.2f}T"
    elif abs(dollars) >= 1_000_000_000:
        disp = f"${dollars/1_000_000_000:,.0f}B"
    else:
        disp = f"${dollars/1_000_000:,.0f}M"

    gdp = d.get("gdp") if isinstance(d.get("gdp"), dict) else None
    gdp_millions = gdp.get("gdp_millions") if gdp else None

    d30 = d.get("deficit_12m_30d_ago") if isinstance(d.get("deficit_12m_30d_ago"), dict) else None
    d1y = d.get("deficit_12m_1y_ago") if isinstance(d.get("deficit_12m_1y_ago"), dict) else None
    d30_val = fmt_usd_from_millions(d30.get("deficit_12m") if d30 else None)
    d30_asof = d30.get("asof") if d30 else None
    d1y_val = fmt_usd_from_millions(d1y.get("deficit_12m") if d1y else None)
    d1y_asof = d1y.get("asof") if d1y else None
    d_yoy = fmt_usd_from_millions(d.get("deficit_12m_delta_yoy"))
    impulse = d.get("deficit_impulse_pct_gdp")
    impulse_disp = f"{float(impulse):+.2f}%" if isinstance(impulse, (int, float)) else "n/a"
    gdp_asof = gdp.get("asof") if gdp else None
    deficit_pct_gdp = d.get("deficit_pct_gdp")
    deficit_pct_gdp_disp = f"{float(deficit_pct_gdp):.1f}%" if isinstance(deficit_pct_gdp, (int, float)) else "n/a"

    # Net issuance from MSPD
    net = d.get("net_issuance") if isinstance(d.get("net_issuance"), dict) else None
    bills = fmt_usd_from_millions(net.get("bills") if net else None)
    coupons = fmt_usd_from_millions(net.get("coupons") if net else None)
    long = fmt_usd_from_millions(net.get("long") if net else None)
    long_share = net.get("long_duration_share") if net else None
    long_share_disp = f"{100.0*float(long_share):.1f}%" if isinstance(long_share, (int, float)) else "n/a"
    net_total_m = None
    if net and all(isinstance(net.get(k), (int, float)) for k in ("bills", "coupons", "long")):
        net_total_m = float(net.get("bills")) + float(net.get("coupons")) + float(net.get("long"))  # type: ignore[arg-type]
    net_total_disp = fmt_usd_from_millions(net_total_m)

    # TGA behavior
    tga = d.get("tga") if isinstance(d.get("tga"), dict) else None
    tga_asof = tga.get("tga_asof") if tga else None
    tga_level = fmt_usd_from_millions(tga.get("tga_level") if tga else None)
    tga_z_level = tga.get("tga_z_level") if tga else None
    tga_z_level_disp = f"{float(tga_z_level):.2f}" if isinstance(tga_z_level, (int, float)) else "n/a"
    tga_d_4w = fmt_usd_from_millions(tga.get("tga_d_4w") if tga else None)
    tga_d_13w = fmt_usd_from_millions(tga.get("tga_d_13w") if tga else None)
    tga_z = tga.get("tga_z_d_4w") if tga else None
    tga_z_disp = f"{float(tga_z):.2f}" if isinstance(tga_z, (int, float)) else "n/a"

    # Auctions
    auctions = d.get("auctions") if isinstance(d.get("auctions"), dict) else None
    auction_asof = auctions.get("asof") if auctions else None
    tail_bps = auctions.get("tail_bps") if auctions else None
    dealer_take = auctions.get("dealer_take_pct") if auctions else None
    tail_disp = f"{float(tail_bps):.1f}bp" if isinstance(tail_bps, (int, float)) else "n/a"
    dealer_disp = f"{float(dealer_take):.1f}%" if isinstance(dealer_take, (int, float)) else "n/a"

    # Per-tenor auction detail
    by_tenor = auctions.get("by_tenor") if auctions else None
    recent_auctions = auctions.get("recent") if auctions else None

    # Use worst-tenor metrics for regime classification (more sensitive to stress)
    worst_tail = tail_bps
    worst_dealer = dealer_take
    if by_tenor and isinstance(by_tenor.get("worst"), dict):
        worst = by_tenor["worst"]
        if isinstance(worst.get("tail_bps"), (int, float)):
            worst_tail = max(float(worst["tail_bps"]), float(tail_bps or 0))
        if isinstance(worst.get("dealer_take_pct"), (int, float)):
            worst_dealer = max(float(worst["dealer_take_pct"]), float(dealer_take or 0))

    # TGA interpretation (MMT: TGA changes directly alter private sector reserves)
    tga_interp = "n/a"
    if isinstance(tga_z, (int, float)) and isinstance(tga.get("tga_d_4w") if tga else None, (int, float)):
        d4 = float(tga.get("tga_d_4w"))  # type: ignore[arg-type]
        z4 = float(tga_z)
        if abs(z4) < 0.75:
            tga_interp = "Neutral: TGA changes not draining or adding to private reserves."
        elif d4 < 0:
            tga_interp = "NFA injection: TGA drawdown adds reserves to private sector."
        else:
            tga_interp = "NFA drain: TGA build-up removes reserves from private sector."

    # MMT sectoral balance
    private_balance_pct_gdp = d.get("private_balance_pct_gdp")
    private_balance_impulse = d.get("private_balance_impulse")

    regime = classify_fiscal_regime_skeleton(
        deficit_12m=deficit_12m,
        gdp_millions=float(gdp_millions) if isinstance(gdp_millions, (int, float)) else None,
        deficit_impulse_pct_gdp=float(d["deficit_impulse_pct_gdp"])
        if isinstance(d.get("deficit_impulse_pct_gdp"), (int, float))
        else None,
        long_duration_issuance_share=float(net.get("long_duration_share"))
        if net and isinstance(net.get("long_duration_share"), (int, float))
        else None,
        tga_z_d_4w=float(tga.get("tga_z_d_4w")) if tga and isinstance(tga.get("tga_z_d_4w"), (int, float)) else None,
        auction_tail_bps=float(worst_tail) if isinstance(worst_tail, (int, float)) else None,
        dealer_take_pct=float(worst_dealer) if isinstance(worst_dealer, (int, float)) else None,
    )

    series_used = d.get("series_used") if isinstance(d.get("series_used"), dict) else {}
    fred_series = series_used.get("fred") if isinstance(series_used.get("fred"), list) else []
    fiscaldata_series = series_used.get("fiscaldata") if isinstance(series_used.get("fiscaldata"), list) else []
    fred_disp = ", ".join(str(x) for x in fred_series) if fred_series else "n/a"
    fiscaldata_disp = ", ".join(str(x) for x in fiscaldata_series) if fiscaldata_series else "n/a"

    # Build snapshot and features for output flags
    # Extract worst-tenor metrics for snapshot / features
    _front = by_tenor.get("front", {}) if by_tenor else {}
    _back = by_tenor.get("back", {}) if by_tenor else {}

    snapshot_data = {
        "deficit_12m": d.get("deficit_12m"),
        "deficit_pct_gdp": deficit_pct_gdp,
        "deficit_impulse_pct_gdp": impulse,
        "private_balance_pct_gdp": private_balance_pct_gdp,
        "private_balance_impulse": private_balance_impulse,
        "net_issuance_bills": net.get("bills") if net else None,
        "net_issuance_coupons": net.get("coupons") if net else None,
        "net_issuance_long": net.get("long") if net else None,
        "long_duration_share": long_share,
        "tga_level": tga.get("tga_level") if tga else None,
        "tga_z_d_4w": tga_z,
        "auction_tail_bps": tail_bps,
        "dealer_take_pct": dealer_take,
        "tail_front_bps": _front.get("tail_bps"),
        "tail_back_bps": _back.get("tail_bps"),
        "dealer_front_pct": _front.get("dealer_take_pct"),
        "dealer_back_pct": _back.get("dealer_take_pct"),
        "regime": regime.label or regime.name,
    }

    feature_dict = {
        "deficit_12m_millions": d.get("deficit_12m"),
        "deficit_pct_gdp": deficit_pct_gdp,
        "deficit_impulse_pct_gdp": impulse,
        "private_balance_pct_gdp": private_balance_pct_gdp,
        "private_balance_impulse": private_balance_impulse,
        "long_duration_share": long_share,
        "tga_z_d_4w": tga_z,
        "auction_tail_bps": worst_tail,
        "dealer_take_pct": worst_dealer,
        "tail_front_bps": _front.get("tail_bps"),
        "tail_back_bps": _back.get("tail_bps"),
        "dealer_front_pct": _front.get("dealer_take_pct"),
        "dealer_back_pct": _back.get("dealer_take_pct"),
    }

    # Handle --features and --json flags
    if handle_output_flags(
        domain="fiscal",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=regime.label or regime.name,
        regime_description=regime.description,
        asof=d.get("asof"),
        output_json=json_out,
        output_features=features,
    ):
        return

    # Handle --alert flag (silent unless extreme)
    if alert:
        show_alert_output("fiscal", regime.label or regime.name, snapshot_data, regime.description)
        return

    # Handle --calendar flag
    if calendar:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label or regime.name}", title="US Fiscal", border_style="cyan"))
        show_calendar_output("fiscal")
        return

    # Handle --trades flag
    if trades:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label or regime.name}", title="US Fiscal", border_style="cyan"))
        show_trades_output("fiscal", regime.label or regime.name)
        return

    # Handle --delta flag
    if delta:
        from lox.cli_commands.shared.labs_utils import get_delta_metrics
        
        delta_days = parse_delta_period(delta)
        
        # Define metrics to track: "Display Name:snapshot_key:unit"
        metric_keys = [
            "Deficit % GDP:deficit_pct_gdp:%",
            "Deficit Impulse:deficit_impulse_pct_gdp:%",
            "Long Duration Share:long_duration_share:",
            "TGA z(Δ4w):tga_z_d_4w:",
            "Auction Tail:auction_tail_bps:bp",
            "Dealer Take:dealer_take_pct:%",
        ]
        
        metrics_for_delta, prev_regime = get_delta_metrics("fiscal", snapshot_data, metric_keys, delta_days)
        show_delta_summary("fiscal", regime.label or regime.name, prev_regime, metrics_for_delta, delta_days)
        
        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox labs fiscal` daily to build history.[/dim]")
        return

    # ── FPI scoring engine (Phase 2 upgrade) ──────────────────────────────
    # Build full FiscalInputs via the state path so FPI has z-scores.
    mc_impact = None
    try:
        fiscal_state = build_fiscal_state(settings=settings, start_date="2011-01-01", refresh=refresh)
        scorecard = score_fiscal_regime(fiscal_state.inputs)
        fpi_score = scorecard.fpi
        fpi_label = scorecard.regime_label
        fpi_desc = scorecard.regime_description

        # Sub-score breakdown for the panel
        sub_scores = [
            {
                "name": s.name,
                "score": round(s.score, 1),
                "weight": f"{s.weight*100:.0f}%",
            }
            for s in scorecard.sub_scores
        ]

        # Calibrated MC impact
        mc_params = calibrate_fiscal_mc(scorecard)
        mc_impact = mc_params.description
    except Exception:
        # Graceful fallback to skeleton regime if scorer fails
        fpi_score = 80 if "dominance" in regime.name else (60 if "contraction" in regime.name else 40)
        fpi_label = regime.label or regime.name
        fpi_desc = regime.description
        sub_scores = None
        mc_impact = None

    # Sectoral balance display values
    pb_disp = f"{float(private_balance_pct_gdp):+.1f}%" if isinstance(private_balance_pct_gdp, (int, float)) else "n/a"
    pb_imp_disp = f"{float(private_balance_impulse):+.2f}%" if isinstance(private_balance_impulse, (int, float)) else "n/a"

    # ── Compute directional changes ──────────────────────────────────────
    def _chg_arrow(delta: float | None, *, invert: bool = False, units: str = "", fmt: str = "{:+.1f}") -> str:
        """Color-coded change string. invert=True means rising is bad."""
        if delta is None:
            return ""
        bad_up = invert
        if abs(delta) < 0.01:
            return "[dim]— flat[/dim]"
        color = ("red" if delta > 0 else "green") if bad_up else ("green" if delta > 0 else "red")
        arrow = "▲" if delta > 0 else "▼"
        return f"[{color}]{arrow} {fmt.format(delta)}{units}[/{color}]"

    # Deficit 30d change (in $B, positive = wider deficit = more NFA to private sector)
    deficit_30d_chg = None
    if d30 and isinstance(d30.get("deficit_12m"), (int, float)):
        deficit_30d_chg = (deficit_12m - float(d30["deficit_12m"])) * 1_000_000 / 1e9  # → $B

    # Deficit 1y change  
    deficit_1y_chg = None
    if d1y and isinstance(d1y.get("deficit_12m"), (int, float)):
        deficit_1y_chg = (deficit_12m - float(d1y["deficit_12m"])) * 1_000_000 / 1e9

    # Deficit % GDP change (30d)
    deficit_pct_30d_chg = None
    if (d30 and isinstance(d30.get("deficit_12m"), (int, float))
            and isinstance(gdp_millions, (int, float)) and float(gdp_millions) != 0):
        prev_pct = 100.0 * float(d30["deficit_12m"]) / float(gdp_millions)
        deficit_pct_30d_chg = float(deficit_pct_gdp) - prev_pct if isinstance(deficit_pct_gdp, (int, float)) else None

    # TGA 4w dollar change for display
    tga_4w_val = tga.get("tga_d_4w") if tga else None
    tga_13w_val = tga.get("tga_d_13w") if tga else None

    # Short context labels (full explanation lives in --full mode)
    def _pb_short(v):
        if not isinstance(v, (int, float)): return ""
        v = float(v)
        if v <= -2: return "deep deficit — fragile"
        if v <= 0: return "private deficit"
        if v <= 3: return "modest surplus"
        return "strong surplus"

    def _deficit_short(v):
        if not isinstance(v, (int, float)): return ""
        v = float(v)
        if v < 3: return "contractionary"
        if v < 6: return "moderate NFA"
        return "strong NFA"

    def _impulse_short(v):
        if not isinstance(v, (int, float)): return ""
        v = float(v)
        if v <= -1: return "sharp drag"
        if v <= -0.25: return "shrinking"
        if v <= 0.25: return "neutral"
        if v <= 1: return "growing"
        return "strong thrust"

    def _tga_short(z):
        if not isinstance(z, (int, float)): return ""
        v = float(z)
        if v <= -0.75: return "low → supportive"
        if v >= 0.75: return "high → draining"
        return "normal range"

    metrics = [
        {"name": "── Sectoral Balance", "value": "", "context": ""},
        {
            "name": "Private balance",
            "value": pb_disp,
            "change": _chg_arrow(private_balance_impulse, invert=True, units="pp YoY"),
            "context": _pb_short(private_balance_pct_gdp),
        },
        {"name": "── Gov't Spending", "value": "", "context": ""},
        {
            "name": "Deficit 12m",
            "value": disp,
            "change": _chg_arrow(deficit_30d_chg, invert=False, units="B 30d", fmt="{:+,.0f}") if deficit_30d_chg is not None else "",
            "context": f"YoY Δ: {d_yoy}" if d_yoy and d_yoy != "n/a" else "",
        },
        {
            "name": "Deficit (% GDP)",
            "value": deficit_pct_gdp_disp,
            "change": _chg_arrow(deficit_pct_30d_chg, invert=False, units="pp 30d") if deficit_pct_30d_chg is not None else "",
            "context": _deficit_short(deficit_pct_gdp),
        },
        {
            "name": "Impulse (% GDP)",
            "value": impulse_disp,
            "context": _impulse_short(impulse),
        },
        {
            "name": "Long issuance",
            "value": long_share_disp,
            "context": "bill-heavy" if isinstance(long_share, (int, float)) and float(long_share) < 0.25 else
                       "long-heavy → watch" if isinstance(long_share, (int, float)) and float(long_share) >= 0.40 else
                       "balanced",
        },
        {"name": "── Reserve Drain", "value": "", "context": ""},
        {
            "name": "TGA level",
            "value": tga_level,
            "change": _chg_arrow(
                float(tga_4w_val) * 1_000_000 / 1e9 if isinstance(tga_4w_val, (int, float)) else None,
                invert=True, units="B 4w", fmt="{:+,.0f}",
            ) if isinstance(tga_4w_val, (int, float)) else "",
            "context": _tga_short(tga_z_level),
        },
    ]

    # Per-tenor auction quality — compact: one row per tenor bucket
    metrics.append({"name": "── Auctions", "value": "", "context": ""})
    _tenor_labels = {"front": "2Y-5Y", "back": "7Y-30Y"}

    def _auction_signal(tail, dealer, btc):
        flags = 0
        if isinstance(tail, (int, float)) and float(tail) >= 3: flags += 1
        if isinstance(dealer, (int, float)) and float(dealer) >= 20: flags += 1
        if isinstance(btc, (int, float)) and float(btc) < 2.3: flags += 1
        if flags >= 2: return "[red]stress[/red]"
        if flags == 1: return "[yellow]watch[/yellow]"
        return "[green]ok[/green]"

    if by_tenor:
        for bucket in ("front", "back"):
            b = by_tenor.get(bucket, {})
            if not isinstance(b, dict) or not b.get("n"):
                continue
            prefix = _tenor_labels.get(bucket, bucket)
            b_tail = b.get("tail_bps")
            b_dealer = b.get("dealer_take_pct")
            b_btc = b.get("btc")
            sig = _auction_signal(b_tail, b_dealer, b_btc)
            t_s = f"{float(b_tail):.1f}" if isinstance(b_tail, (int, float)) else "—"
            d_s = f"{float(b_dealer):.0f}" if isinstance(b_dealer, (int, float)) else "—"
            c_s = f"{float(b_btc):.1f}" if isinstance(b_btc, (int, float)) else "—"
            metrics.append({
                "name": prefix,
                "value": f"{t_s}bp/{d_s}%/{c_s}x",
                "context": sig,
            })
    else:
        metrics.append({"name": "All tenors", "value": f"{tail_disp}/{dealer_disp}", "context": ""})

    # Append MC impact to description if available
    full_desc = fpi_desc
    if mc_impact:
        full_desc = f"{fpi_desc}\n[dim]MC impact: {mc_impact}[/dim]"

    from lox.regimes.trend import get_domain_trend
    trend = get_domain_trend("fiscal", fpi_score, fpi_label)

    print(render_regime_panel(
        domain="Fiscal",
        asof=d.get("asof", ""),
        regime_label=fpi_label,
        score=fpi_score,
        percentile=None,
        description=full_desc,
        metrics=metrics,
        sub_scores=sub_scores,
        trend=trend,
    ))

    # Recent individual auctions detail table (10Y + 30Y only — the tenors
    # that matter most for duration supply / term-premium stress)
    if recent_auctions:
        from rich.table import Table as RichTable
        _LONG_TENORS = {"10-Year", "30-Year", "20-Year"}
        long_auctions = [ra for ra in recent_auctions if ra.get("term") in _LONG_TENORS]
        at = RichTable(
            title="Recent Long-End Auctions (10Y / 20Y / 30Y)",
            show_header=True,
            header_style="bold yellow",
            box=None,
            padding=(0, 1),
        )
        at.add_column("Date", style="dim")
        at.add_column("Term")
        at.add_column("Tail", justify="right")
        at.add_column("Dealer %", justify="right")
        at.add_column("BTC", justify="right")
        at.add_column("Signal", style="dim")
        for ra in long_auctions:
            t = ra.get("tail_bps")
            d_pct = ra.get("dealer_take_pct")
            btc_v = ra.get("btc")
            # Color-code the signal
            stress_flags = 0
            if isinstance(t, (int, float)) and float(t) >= 3.0:
                stress_flags += 1
            if isinstance(d_pct, (int, float)) and float(d_pct) >= 20.0:
                stress_flags += 1
            if isinstance(btc_v, (int, float)) and float(btc_v) < 2.3:
                stress_flags += 1
            if stress_flags >= 2:
                signal = "[red]STRESS[/red]"
            elif stress_flags == 1:
                signal = "[yellow]WATCH[/yellow]"
            else:
                signal = "[green]OK[/green]"
            at.add_row(
                str(ra.get("date", "")),
                str(ra.get("term", "")),
                f"{float(t):.1f}bp" if isinstance(t, (int, float)) else "—",
                f"{float(d_pct):.1f}%" if isinstance(d_pct, (int, float)) else "—",
                f"{float(btc_v):.2f}x" if isinstance(btc_v, (int, float)) else "—",
                signal,
            )
        console.print()
        console.print(at)

        # ── Sparkline trend summary ──────────────────────────────────────
        if len(long_auctions) >= 4:
            _show_auction_trend(console, long_auctions)

        # ── Upcoming long-end auctions ────────────────────────────────────
        upcoming = _get_upcoming_long_end(long_auctions)
        if upcoming:
            from rich.table import Table as RichTable
            ut = RichTable(
                title="Upcoming Long-End Auctions",
                show_header=True,
                header_style="bold cyan",
                box=None,
                padding=(0, 1),
            )
            ut.add_column("Date", style="dim")
            ut.add_column("Term")
            ut.add_column("Offering")
            ut.add_column("Type", style="dim")
            ut.add_column("Source", style="dim")
            for ua in upcoming:
                dt = ua["date"]
                days_away = (dt - date.today()).days
                date_str = f"{dt.isoformat()}  ({days_away}d)"
                offering = f"${ua['offering_b']:.0f}B" if ua.get("offering_b") else "—"
                reopen = "Reopen" if ua.get("reopening") else "New"
                src = "[green]announced[/green]" if ua["source"] == "announced" else "[dim]est. from cadence[/dim]"
                ut.add_row(date_str, ua["tenor"], offering, reopen, src)
            console.print()
            console.print(ut)

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis

        snapshot_data = {
            "deficit_12m": d.get("deficit_12m"),
            "deficit_pct_gdp": deficit_pct_gdp,
            "deficit_impulse_pct_gdp": impulse,
            "private_balance_pct_gdp": private_balance_pct_gdp,
            "private_balance_impulse": private_balance_impulse,
            "net_issuance_bills": net.get("bills") if net else None,
            "net_issuance_coupons": net.get("coupons") if net else None,
            "net_issuance_long": net.get("long") if net else None,
            "long_duration_share": long_share,
            "tga_level": tga.get("tga_level") if tga else None,
            "tga_z_d_4w": tga_z,
            "tga_d_4w": tga.get("tga_d_4w") if tga else None,
            "auction_tail_bps_blended": tail_bps,
            "dealer_take_pct_blended": dealer_take,
            "auction_by_tenor": by_tenor,
            "recent_auctions": recent_auctions,
        }
        print_llm_regime_analysis(
            settings=settings,
            domain="fiscal",
            snapshot=snapshot_data,
            regime_label=regime.label or regime.name,
            regime_description=regime.description,
            ticker=ticker,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI Registration
# ─────────────────────────────────────────────────────────────────────────────

def fiscal_snapshot(**kwargs) -> None:
    """Entry point for `lox regime fiscal` (no subcommand)."""
    _run_fiscal_snapshot(**kwargs)


def register(fiscal_app: typer.Typer) -> None:
    @fiscal_app.callback(invoke_without_command=True)
    def fiscal_default(
        ctx: typer.Context,
        llm: bool = typer.Option(False, "--llm", help="Chat with LLM analyst"),
        ticker: str = typer.Option("", "--ticker", "-t", help="Ticker for focused chat (used with --llm)"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme (for cron/monitoring)"),
        calendar: bool = typer.Option(False, "--calendar", help="Show upcoming events that could shift this regime"),
        trades: bool = typer.Option(False, "--trades", help="Show quick trade expressions for current regime"),
    ):
        """US fiscal regime (deficits, issuance mix, auctions, TGA)"""
        if ctx.invoked_subcommand is None:
            _run_fiscal_snapshot(llm=llm, ticker=ticker, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

    @fiscal_app.command("snapshot")
    def fiscal_snapshot(
        lookback_years: int = typer.Option(
            5,
            "--lookback-years",
            help="How many years of history to load (enough for rolling 12m deficit).",
        ),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
        full: bool = typer.Option(
            False,
            "--full",
            help="Print the full fiscal state (TGA/interest placeholders) and the richer regime classifier.",
        ),
        llm: bool = typer.Option(False, "--llm", help="Chat with LLM analyst"),
        ticker: str = typer.Option("", "--ticker", "-t", help="Ticker for focused chat (used with --llm)"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme (for cron/monitoring)"),
        calendar: bool = typer.Option(False, "--calendar", help="Show upcoming events that could shift this regime"),
        trades: bool = typer.Option(False, "--trades", help="Show quick trade expressions for current regime"),
    ):
        """
        Print fiscal regime snapshot.

        Default behavior is intentionally simple and uniform with other regimes:
        it prints the rolling 12m deficit plus a skeleton regime label.
        Use --full for the richer snapshot + classifier.
        """
        _run_fiscal_snapshot(lookback_years=lookback_years, refresh=refresh, full=full, llm=llm, ticker=ticker, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

    @fiscal_app.command("outlook")
    def fiscal_outlook(
        lookback_years: int = typer.Option(5, "--lookback-years", help="How many years of history to load."),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh downloads"),
        llm_model: str = typer.Option("", "--llm-model", help="Override OPENAI_MODEL (optional)"),
        llm_temperature: float = typer.Option(0.2, "--llm-temperature", help="LLM temperature (0..2)"),
    ):
        """
        Ask an LLM to summarize the current fiscal snapshot and implications for trading.
        Grounded in the same quantitative snapshot used by `fiscal snapshot`.
        """
        settings = load_settings()
        d = build_fiscal_deficit_page_data(settings=settings, lookback_years=lookback_years, refresh=refresh)

        gdp = d.get("gdp") if isinstance(d.get("gdp"), dict) else None
        gdp_millions = gdp.get("gdp_millions") if gdp else None
        net = d.get("net_issuance") if isinstance(d.get("net_issuance"), dict) else None
        tga = d.get("tga") if isinstance(d.get("tga"), dict) else None
        auctions = d.get("auctions") if isinstance(d.get("auctions"), dict) else None

        deficit_12m = float(d["deficit_12m"]) if isinstance(d.get("deficit_12m"), (int, float)) else None
        regime = classify_fiscal_regime_skeleton(
            deficit_12m=deficit_12m,
            gdp_millions=float(gdp_millions) if isinstance(gdp_millions, (int, float)) else None,
            deficit_impulse_pct_gdp=float(d["deficit_impulse_pct_gdp"])
            if isinstance(d.get("deficit_impulse_pct_gdp"), (int, float))
            else None,
            long_duration_issuance_share=float(net.get("long_duration_share"))
            if net and isinstance(net.get("long_duration_share"), (int, float))
            else None,
            tga_z_d_4w=float(tga.get("tga_z_d_4w")) if tga and isinstance(tga.get("tga_z_d_4w"), (int, float)) else None,
            auction_tail_bps=float(auctions.get("tail_bps"))
            if auctions and isinstance(auctions.get("tail_bps"), (int, float))
            else None,
            dealer_take_pct=float(auctions.get("dealer_take_pct"))
            if auctions and isinstance(auctions.get("dealer_take_pct"), (int, float))
            else None,
        )

        from lox.llm.outlooks.fiscal_outlook import llm_fiscal_outlook

        text = llm_fiscal_outlook(
            settings=settings,
            fiscal_snapshot=d,
            fiscal_regime=regime,
            model=llm_model.strip() or None,
            temperature=float(llm_temperature),
        )
        print(text)
