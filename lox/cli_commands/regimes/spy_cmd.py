"""CLI command for the SPY Options Flow regime.

Single-fetch, 3-signal microstructure dashboard:
  GEX, Put/Call Ratio, 25-delta Skew.

All data derived from one options chain fetch.
"""
from __future__ import annotations

from rich import print

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


# ── Formatting helpers ────────────────────────────────────────────────────

def _fmt(x: object, fmt: str = "{:.1f}") -> str:
    return fmt.format(float(x)) if isinstance(x, (int, float)) else "n/a"

def _fmt_sign(x: object, fmt: str = "{:+.1f}") -> str:
    return fmt.format(float(x)) if isinstance(x, (int, float)) else "n/a"

def _fmt_pct_diff(spot, level) -> str:
    if isinstance(spot, (int, float)) and isinstance(level, (int, float)) and spot > 0:
        pct = (level - spot) / spot * 100
        return f"{level:.0f} ({pct:+.1f}% from spot)"
    return "n/a"


# ── Context helpers ───────────────────────────────────────────────────────

def _gex_ctx(gex_bn) -> str:
    if not isinstance(gex_bn, (int, float)):
        return "aggregate dealer gamma exposure"
    if gex_bn > 5.0:
        return "very positive — dealers long gamma, market pinning/stabilizing"
    if gex_bn > 2.0:
        return "positive — supportive, vol-suppressing"
    if gex_bn > 0:
        return "mildly positive — mild stabilization"
    if gex_bn > -2.0:
        return "negative — dealers short gamma, vol-amplifying"
    if gex_bn > -5.0:
        return "significantly negative — mechanical selling risk, wider moves"
    return "deeply negative — extreme vol amplification, tail risk elevated"


def _pcr_ctx(pcr) -> str:
    if not isinstance(pcr, (int, float)):
        return "OI-weighted equity put/call ratio"
    if pcr > 1.2:
        return "heavy put buying — institutional hedging or fear"
    if pcr > 1.0:
        return "elevated protection — above-average put demand"
    if pcr > 0.8:
        return "neutral — balanced options flow"
    if pcr > 0.6:
        return "call-skewed — bullish positioning"
    return "extreme call bias — speculative froth"


def _skew_ctx(skew) -> str:
    if not isinstance(skew, (int, float)):
        return "25-delta risk reversal (put IV − call IV)"
    if skew > 8:
        return "very steep — extreme downside protection demand"
    if skew > 5:
        return "elevated — meaningful put premium, hedging demand"
    if skew > 2:
        return "normal — standard demand for downside protection"
    if skew > 0:
        return "mild — balanced skew"
    return "inverted — calls richer than puts, unusual complacency"


# ── Core implementation ───────────────────────────────────────────────────

def spy_snapshot(
    *,
    refresh: bool = False,
    json_out: bool = False,
    llm: bool = False,
) -> None:
    """Entry point for `lox regime spy`."""
    from datetime import datetime, timezone
    from rich.console import Console
    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, save_snapshot,
    )

    settings = load_settings()
    console = Console()
    asof = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── 1. Fetch spot price ──────────────────────────────────────────────
    spot: float | None = None
    try:
        from lox.altdata.fmp import fetch_realtime_quotes
        quotes = fetch_realtime_quotes(settings=settings, tickers=["SPY"])
        spot = quotes.get("SPY")
    except Exception as e:
        console.print(f"[dim]Spot price unavailable: {e}[/dim]")

    # ── 2. Fetch options chain (single fetch) ────────────────────────────
    try:
        from lox.positioning.data import _fetch_options_chain
        chain = _fetch_options_chain(settings, "SPY")
    except Exception as e:
        console.print(f"[red]Options chain fetch failed:[/red] {e}")
        return

    if not chain:
        console.print("[red]No options chain data available for SPY.[/red]")
        return

    # ── 3. Compute 3 signals ─────────────────────────────────────────────
    from lox.positioning.data import (
        compute_gex_from_chain,
        compute_pcr_from_chain,
        compute_skew_from_chain,
    )

    gex_bn, flip_level = compute_gex_from_chain(chain, spot or 0)
    pcr = compute_pcr_from_chain(chain)
    skew_25d = compute_skew_from_chain(chain, spot or 0)

    # ── 4. Classify ──────────────────────────────────────────────────────
    from lox.regimes.spy import classify_spy_regime

    regime = classify_spy_regime(
        gex_bn=gex_bn,
        pcr=pcr,
        skew_25d=skew_25d,
        spot=spot,
        flip_level=flip_level,
    )

    # ── 5. Snapshot for delta tracking ───────────────────────────────────
    snapshot_data = {
        "spot": spot,
        "gex_bn": gex_bn,
        "gex_flip": flip_level,
        "pcr": pcr,
        "skew_25d": skew_25d,
        "regime": regime.label,
    }

    feature_dict = {
        "gex_bn": gex_bn,
        "pcr": pcr,
        "skew_25d": skew_25d,
    }

    save_snapshot("spy", snapshot_data, regime.label)

    if handle_output_flags(
        domain="spy",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=regime.label,
        regime_description=regime.description,
        asof=asof,
        output_json=json_out,
        output_features=False,
    ):
        return

    # ── 6. Build metrics ─────────────────────────────────────────────────
    metrics = [
        {"name": "SPY Spot", "value": _fmt(spot, "${:.2f}"), "context": "current price"},
        {"name": "GEX Total", "value": _fmt_sign(gex_bn, "{:+.2f}") + " $bn", "context": _gex_ctx(gex_bn)},
        {"name": "GEX Flip Level", "value": _fmt_pct_diff(spot, flip_level), "context": "price where dealer gamma flips negative"},
        {"name": "Put/Call Ratio (OI)", "value": _fmt(pcr, "{:.2f}"), "context": _pcr_ctx(pcr)},
        {"name": "25d Skew (vol pts)", "value": _fmt_sign(skew_25d, "{:+.1f}"), "context": _skew_ctx(skew_25d)},
    ]

    # ── 7. Sub-scores for transparency ───────────────────────────────────
    from lox.regimes.spy import _gex_subscore, _pcr_subscore, _skew_subscore, W_GEX, W_PCR, W_SKEW

    sub_scores = []
    if gex_bn is not None:
        sub_scores.append({"name": "GEX", "score": _gex_subscore(gex_bn), "weight": W_GEX})
    if pcr is not None:
        sub_scores.append({"name": "Put/Call", "score": _pcr_subscore(pcr), "weight": W_PCR})
    if skew_25d is not None:
        sub_scores.append({"name": "Skew", "score": _skew_subscore(skew_25d), "weight": W_SKEW})

    # ── 8. Trend ─────────────────────────────────────────────────────────
    trend = None
    try:
        from lox.regimes.trend import get_domain_trend
        trend = get_domain_trend("spy", regime.score, regime.label)
    except Exception:
        pass

    # ── 9. Render ────────────────────────────────────────────────────────
    panel = render_regime_panel(
        domain="SPY Options Flow",
        asof=asof,
        regime_label=regime.label,
        score=regime.score,
        description=regime.description,
        metrics=metrics,
        sub_scores=sub_scores,
        trend=trend,
    )
    print(panel)

    # ── 10. LLM chat ────────────────────────────────────────────────────
    if llm:
        from lox.cli_commands.shared.regime_chat import start_regime_chat
        start_regime_chat(
            domain="spy",
            regime_label=regime.label,
            regime_description=regime.description,
            score=regime.score,
            metrics=snapshot_data,
        )
