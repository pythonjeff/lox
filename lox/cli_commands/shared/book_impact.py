"""
Book Impact — correlate open positions with current regime state.

Factor-based approach: classify each ticker by its macro factor exposures
(equity beta, duration, credit, commodities, etc.), then score those
exposures against the current regime direction.  Adjusts for position
directionality (puts = inverted, shorts = inverted).

Usage:
    from lox.cli_commands.shared.book_impact import analyze_book_impact, render_book_impact

    impacts = analyze_book_impact(settings, regime_state)
    render_book_impact(console, impacts)
"""
from __future__ import annotations

import logging
import re
from copy import copy
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DomainImpact:
    """Single domain's impact on a position."""
    domain: str
    regime_label: str
    regime_score: float
    direction: str           # "risk_on", "risk_off", "rising", "falling", etc.
    raw_alignment: str       # "benefit" or "hurt"
    effective_signal: str    # "TAILWIND" or "HEADWIND"
    rationale: str


@dataclass
class PositionImpact:
    """Aggregate regime impact on a single position."""
    symbol: str
    underlying: str
    position_type: str       # "equity", "call", "put", "short_equity"
    qty: float
    unrealized_pl: float
    unrealized_plpc: float
    market_value: float
    impacts: list[DomainImpact] = field(default_factory=list)

    @property
    def headwind_count(self) -> int:
        return sum(1 for i in self.impacts if i.effective_signal == "HEADWIND")

    @property
    def tailwind_count(self) -> int:
        return sum(1 for i in self.impacts if i.effective_signal == "TAILWIND")

    @property
    def net_signal(self) -> str:
        if not self.impacts:
            return "NEUTRAL"
        if self.headwind_count > self.tailwind_count:
            return "HEADWIND"
        if self.tailwind_count > self.headwind_count:
            return "TAILWIND"
        return "MIXED"


# ── Factor exposure model ────────────────────────────────────────────────────
# Each ticker gets factor loadings: positive = benefits when factor rises,
# negative = hurts when factor rises.
#
# Factors:
#   equity_beta  — sensitivity to broad equity market direction
#   duration     — sensitivity to rate moves (positive = benefits from falling rates)
#   credit       — sensitivity to credit spreads (positive = benefits from tight spreads)
#   commodities  — sensitivity to commodity prices
#   gold         — sensitivity to gold / precious metals
#   vol          — sensitivity to volatility (positive = benefits from vol spike)
#   inflation    — sensitivity to inflation expectations

FACTOR_EXPOSURES: dict[str, dict[str, float]] = {
    # ── Broad equity ─────────────────────────────────────────────────────
    "SPY":  {"equity_beta": 1.0, "duration": -0.2},
    "QQQ":  {"equity_beta": 1.3, "duration": -0.3},
    "IWM":  {"equity_beta": 1.2, "credit": 0.3},
    "DIA":  {"equity_beta": 0.9},
    "RSP":  {"equity_beta": 1.0},
    # Inverse / leveraged
    "SH":   {"equity_beta": -1.0},
    "SPXU": {"equity_beta": -3.0},
    "SQQQ": {"equity_beta": -3.0, "duration": 0.5},
    "TQQQ": {"equity_beta": 3.0, "duration": -0.5},
    "SPXL": {"equity_beta": 3.0},
    "UPRO": {"equity_beta": 3.0},

    # ── Sector ETFs ──────────────────────────────────────────────────────
    "XLF":  {"equity_beta": 1.0, "duration": -0.5, "credit": 0.4},
    "KRE":  {"equity_beta": 1.1, "duration": -0.7, "credit": 0.5},
    "XLE":  {"equity_beta": 0.8, "commodities": 0.8, "inflation": 0.5},
    "XOP":  {"equity_beta": 1.0, "commodities": 0.9},
    "OIH":  {"equity_beta": 0.9, "commodities": 0.8},
    "XLU":  {"equity_beta": 0.4, "duration": 0.6},
    "XLP":  {"equity_beta": 0.5, "duration": 0.3},
    "XLV":  {"equity_beta": 0.6},
    "XLY":  {"equity_beta": 1.2, "credit": 0.2},
    "XLI":  {"equity_beta": 1.0, "commodities": -0.2},
    "XLB":  {"equity_beta": 1.0, "commodities": 0.5, "inflation": 0.3},
    "XLC":  {"equity_beta": 1.1},
    "XLRE": {"equity_beta": 0.7, "duration": 0.7, "credit": 0.3},
    "XLK":  {"equity_beta": 1.2, "duration": -0.2},
    "XME":  {"equity_beta": 1.0, "commodities": 0.7, "inflation": 0.3},

    # ── Retail / consumer ────────────────────────────────────────────────
    "XRT":  {"equity_beta": 1.1, "credit": 0.2},
    "XHB":  {"equity_beta": 1.0, "duration": 0.6},
    "ITB":  {"equity_beta": 1.0, "duration": 0.6},

    # ── Defense / aerospace ──────────────────────────────────────────────
    "XAR":  {"equity_beta": 0.8},
    "ITA":  {"equity_beta": 0.8},

    # ── Bonds / duration ─────────────────────────────────────────────────
    "TLT":  {"duration": 1.0, "equity_beta": -0.2},
    "IEF":  {"duration": 0.6, "equity_beta": -0.1},
    "SHY":  {"duration": 0.2},
    "TBT":  {"duration": -1.0},
    "TMV":  {"duration": -1.5},
    "TMF":  {"duration": 1.5},
    "BND":  {"duration": 0.5, "credit": 0.1},
    "AGG":  {"duration": 0.5, "credit": 0.1},
    "TIP":  {"duration": 0.4, "inflation": 0.5},
    "GOVT": {"duration": 0.5},
    "SGOV": {"duration": 0.05},
    "BIL":  {"duration": 0.02},

    # ── Credit ───────────────────────────────────────────────────────────
    "HYG":  {"credit": 0.8, "equity_beta": 0.4, "duration": 0.2},
    "JNK":  {"credit": 0.8, "equity_beta": 0.4, "duration": 0.2},
    "LQD":  {"credit": 0.4, "duration": 0.5},
    "BKLN": {"credit": 0.6},

    # ── Commodities ──────────────────────────────────────────────────────
    "GLD":  {"gold": 1.0, "inflation": 0.4, "equity_beta": -0.1},
    "GLDM": {"gold": 1.0, "inflation": 0.4, "equity_beta": -0.1},
    "SLV":  {"gold": 0.7, "commodities": 0.3, "inflation": 0.3},
    "GDX":  {"gold": 0.9, "equity_beta": 0.3},
    "GDXJ": {"gold": 1.0, "equity_beta": 0.4},
    "SIL":  {"gold": 0.8, "equity_beta": 0.3},
    "USO":  {"commodities": 1.0, "inflation": 0.4},
    "DBC":  {"commodities": 0.8, "inflation": 0.4},
    "PDBC": {"commodities": 0.8, "inflation": 0.4},
    "COPX": {"commodities": 0.6, "equity_beta": 0.5},

    # ── Volatility ───────────────────────────────────────────────────────
    "VXX":  {"vol": 1.0, "equity_beta": -0.8},
    "VIXY": {"vol": 1.0, "equity_beta": -0.8},
    "UVXY": {"vol": 1.5, "equity_beta": -1.2},
    "SVXY": {"vol": -1.0, "equity_beta": 0.6},
    "USMV": {"equity_beta": 0.6, "vol": -0.2},

    # ── USD / FX ─────────────────────────────────────────────────────────
    "UUP":  {"gold": -0.4, "commodities": -0.2},
    "EEM":  {"equity_beta": 0.8, "commodities": 0.3},
    "EFA":  {"equity_beta": 0.8},

    # ── Crypto-adjacent ──────────────────────────────────────────────────
    "COIN": {"equity_beta": 1.5, "credit": 0.3},
    "MARA": {"equity_beta": 1.8},
    "RIOT": {"equity_beta": 1.8},
    "BITO": {"equity_beta": 1.2},
    "GBTC": {"equity_beta": 1.2},

    # ── Airlines / transports ────────────────────────────────────────────
    "JETS": {"equity_beta": 1.0, "commodities": -0.5},
    "XTN":  {"equity_beta": 1.0, "commodities": -0.3},

    # ── REITs ────────────────────────────────────────────────────────────
    "VNQ":  {"equity_beta": 0.7, "duration": 0.7, "credit": 0.3},
    "XLRE": {"equity_beta": 0.7, "duration": 0.7, "credit": 0.3},

    # ── Low-vol / quality ────────────────────────────────────────────────
    "ARKK": {"equity_beta": 1.5, "duration": -0.4},
}

# Sector → default factor exposures (for stocks not in FACTOR_EXPOSURES)
SECTOR_DEFAULTS: dict[str, dict[str, float]] = {
    "technology":             {"equity_beta": 1.2, "duration": -0.2},
    "consumer cyclical":      {"equity_beta": 1.1, "credit": 0.2},
    "consumer defensive":     {"equity_beta": 0.5, "duration": 0.3},
    "financial services":     {"equity_beta": 1.0, "duration": -0.5, "credit": 0.4},
    "healthcare":             {"equity_beta": 0.6},
    "industrials":            {"equity_beta": 1.0, "commodities": -0.2},
    "energy":                 {"equity_beta": 0.8, "commodities": 0.7, "inflation": 0.4},
    "utilities":              {"equity_beta": 0.4, "duration": 0.6},
    "real estate":            {"equity_beta": 0.7, "duration": 0.7},
    "basic materials":        {"equity_beta": 1.0, "commodities": 0.5, "inflation": 0.3},
    "communication services": {"equity_beta": 1.1},
}

# Cache: ticker → sector string (populated lazily via FMP)
_sector_cache: dict[str, str | None] = {}


def _get_factor_exposures(settings: Any, ticker: str) -> dict[str, float] | None:
    """Get factor exposures for a ticker.

    1. Check explicit FACTOR_EXPOSURES table
    2. Look up sector via FMP and use SECTOR_DEFAULTS
    3. Return None if completely unknown
    """
    if ticker in FACTOR_EXPOSURES:
        return FACTOR_EXPOSURES[ticker]

    # Try sector lookup
    if ticker in _sector_cache:
        sector = _sector_cache[ticker]
        if sector:
            return SECTOR_DEFAULTS.get(sector.lower())
        return None

    try:
        from lox.altdata.fmp import fetch_profile
        profile = fetch_profile(settings=settings, ticker=ticker)
        if profile and profile.sector:
            _sector_cache[ticker] = profile.sector
            return SECTOR_DEFAULTS.get(profile.sector.lower())
    except Exception:
        pass
    _sector_cache[ticker] = None
    return None


# ── Regime → factor direction mapping ────────────────────────────────────────
# For each domain, map the regime label to which factors it pushes and in
# which direction.  Positive = factor is rising/favorable, negative = falling.
#
# Example: growth "Accelerating" → equity_beta +1 (good for equities),
#          duration -1 (bad for bonds, rates likely rising)

def _classify_regime_direction(domain: str, label: str) -> dict[str, float]:
    """Map a regime domain + label into factor directions.

    Returns dict of {factor: direction} where:
      +1 = factor rising / risk-on / favorable
      -1 = factor falling / risk-off / unfavorable
    """
    low = label.lower()

    if domain == "volatility":
        if any(k in low for k in ("shock", "stress", "crisis")):
            return {"equity_beta": -1, "vol": +1, "credit": -1, "duration": +1}
        if any(k in low for k in ("elevated", "fragile", "moderate")):
            return {"equity_beta": -0.5, "vol": +0.5, "credit": -0.3}
        if any(k in low for k in ("normal", "baseline", "low", "complacent")):
            return {"equity_beta": +0.5, "vol": -0.5, "credit": +0.3}
        return {}

    if domain == "growth":
        if any(k in low for k in ("contraction", "recession")):
            return {"equity_beta": -1, "duration": +1, "credit": -1, "commodities": -0.5}
        if any(k in low for k in ("slowing", "decelerat", "weaken")):
            return {"equity_beta": -0.5, "duration": +0.5, "credit": -0.3}
        if any(k in low for k in ("stable",)):
            return {"equity_beta": +0.3, "duration": -0.2}
        if any(k in low for k in ("accelerat", "boom", "expand", "strong", "robust")):
            return {"equity_beta": +1, "duration": -0.5, "credit": +0.5, "commodities": +0.3}
        if any(k in low for k in ("recovery", "early cycle")):
            return {"equity_beta": +1, "duration": -0.3, "credit": +0.7}
        return {}

    if domain == "rates":
        if any(k in low for k in ("shock higher", "bear steepener", "bear flattener",
                                   "rising", "hawkish")):
            return {"duration": -1, "equity_beta": -0.3, "credit": -0.2}
        if any(k in low for k in ("shock lower", "bull steepener", "bull flattener",
                                   "falling", "easing")):
            return {"duration": +1, "equity_beta": +0.2, "credit": +0.2}
        if any(k in low for k in ("steep curve", "steepening")):
            return {"duration": -0.5, "equity_beta": +0.3}
        if any(k in low for k in ("inverted", "flattening")):
            return {"duration": +0.3, "equity_beta": -0.3}
        if "neutral" in low:
            return {}
        return {}

    if domain == "inflation":
        if "stagflat" in low:
            return {"inflation": +1, "equity_beta": -0.7, "commodities": +0.5,
                    "gold": +0.5, "duration": -0.5}
        if any(k in low for k in ("hot", "above target", "elevated", "rising")):
            return {"inflation": +1, "duration": -0.5, "commodities": +0.5,
                    "gold": +0.3}
        if any(k in low for k in ("at target",)):
            return {}
        if any(k in low for k in ("below target", "deflat", "falling", "disinflat")):
            return {"inflation": -1, "duration": +0.5, "equity_beta": +0.3,
                    "gold": -0.2}
        return {}

    if domain == "commodities":
        if any(k in low for k in ("energy shock", "reflation", "rally", "rising")):
            return {"commodities": +1, "inflation": +0.5, "gold": +0.3,
                    "equity_beta": -0.2}
        if any(k in low for k in ("disinflation", "falling", "neutral")):
            return {"commodities": -0.5, "inflation": -0.3}
        return {}

    if domain == "credit":
        if any(k in low for k in ("tight", "benign", "compress")):
            return {"credit": +0.5, "equity_beta": +0.3}
        if any(k in low for k in ("widen", "stress", "blow")):
            return {"credit": -1, "equity_beta": -0.5, "duration": +0.3}
        return {}

    if domain == "liquidity":
        if any(k in low for k in ("abundant", "loose", "easing")):
            return {"equity_beta": +0.5, "credit": +0.3}
        if any(k in low for k in ("tight", "stress", "drain")):
            return {"equity_beta": -0.5, "credit": -0.5}
        return {}

    return {}


# ── Core functions ───────────────────────────────────────────────────────────

def parse_position_type(symbol: str, qty: float) -> tuple[str, str]:
    """Parse a position into (underlying_ticker, position_type).

    Returns:
        (underlying, type) where type is one of:
        "equity", "short_equity", "call", "put"
    """
    # Options symbols are OCC format: AAPL250117C00200000 (15-21 chars)
    if len(symbol) > 10:
        underlying = ""
        for i, c in enumerate(symbol):
            if c.isdigit():
                underlying = symbol[:i]
                break
        if not underlying:
            underlying = symbol[:4]

        date_start = len(underlying)
        option_part = symbol[date_start:]
        if "C" in option_part:
            return underlying.upper(), "call"
        elif "P" in option_part:
            return underlying.upper(), "put"
        return underlying.upper(), "call"

    if qty < 0:
        return symbol.upper(), "short_equity"
    return symbol.upper(), "equity"


def compute_effective_signal(raw_alignment: str, position_type: str) -> str:
    """Compute TAILWIND or HEADWIND adjusted for position directionality."""
    inverted = position_type in ("put", "short_equity")
    if raw_alignment == "benefit":
        return "HEADWIND" if inverted else "TAILWIND"
    elif raw_alignment == "hurt":
        return "TAILWIND" if inverted else "HEADWIND"
    return "NEUTRAL"


def _score_position_vs_regime(
    factor_exposures: dict[str, float],
    regime_direction: dict[str, float],
) -> float:
    """Dot-product of factor exposures × regime direction.

    Positive = regime is a tailwind for this position (long).
    Negative = regime is a headwind.
    """
    score = 0.0
    for factor, exposure in factor_exposures.items():
        direction = regime_direction.get(factor, 0.0)
        score += exposure * direction
    return score


def analyze_book_impact(settings: Any, regime_state: Any) -> list[PositionImpact]:
    """Analyze open positions against current regime state.

    Uses factor-based scoring: each ticker's factor exposures are dot-product'd
    against the regime's factor direction to produce a tailwind/headwind signal.

    Returns list of PositionImpact sorted by headwind count (most exposed first).
    """
    from lox.cli_commands.research.portfolio_cmd import _fetch_positions

    positions = _fetch_positions(settings)
    if not positions:
        return []

    # Map domain names to regime objects
    domain_regimes: dict[str, Any] = {}
    for domain in ("volatility", "rates", "growth", "inflation",
                    "commodities", "credit", "liquidity"):
        regime = getattr(regime_state, domain, None)
        if regime is not None:
            domain_regimes[domain] = regime

    # Threshold: |score| must exceed this to count as a signal
    SIGNAL_THRESHOLD = 0.3

    results: list[PositionImpact] = []
    for pos in positions:
        underlying, pos_type = parse_position_type(pos["symbol"], pos["qty"])

        impact = PositionImpact(
            symbol=pos["symbol"],
            underlying=underlying,
            position_type=pos_type,
            qty=pos["qty"],
            unrealized_pl=pos["unrealized_pl"],
            unrealized_plpc=pos["unrealized_plpc"],
            market_value=pos["market_value"],
        )

        exposures = _get_factor_exposures(settings, underlying)
        if exposures is None:
            results.append(impact)
            continue

        for domain, regime in domain_regimes.items():
            direction = _classify_regime_direction(domain, regime.label)
            if not direction:
                continue

            score = _score_position_vs_regime(exposures, direction)

            if abs(score) < SIGNAL_THRESHOLD:
                continue

            # Determine raw alignment (for the underlying, not position)
            raw_alignment = "benefit" if score > 0 else "hurt"
            effective = compute_effective_signal(raw_alignment, pos_type)

            # Build a concise rationale from the dominant factor
            dominant_factor = max(
                direction.keys(),
                key=lambda f: abs(exposures.get(f, 0) * direction.get(f, 0)),
            )
            rationale = f"{underlying} {dominant_factor} exposure vs {domain}"

            impact.impacts.append(DomainImpact(
                domain=domain,
                regime_label=regime.label,
                regime_score=regime.score,
                direction=f"{'↑' if score > 0 else '↓'} {dominant_factor}",
                raw_alignment=raw_alignment,
                effective_signal=effective,
                rationale=rationale,
            ))

        results.append(impact)

    # Sort: most headwinds first, then by market value
    results.sort(key=lambda r: (-r.headwind_count, -abs(r.market_value)))
    return results


def render_book_impact(console: Console, impacts: list[PositionImpact]) -> None:
    """Render the book impact table to console.

    Shows ALL positions.  Correlated ones get TAILWIND/HEADWIND signals;
    uncorrelated ones show a dim dash.
    """
    if not impacts:
        console.print("[dim]No open positions found.[/dim]")
        return

    matched = [p for p in impacts if p.impacts]
    hw_positions = sum(1 for p in matched if p.net_signal == "HEADWIND")
    tw_positions = sum(1 for p in matched if p.net_signal == "TAILWIND")

    table = Table(
        title="[bold]Book Impact — Position × Regime[/bold]",
        box=None,
        padding=(0, 1),
        header_style="bold",
    )
    table.add_column("Position", style="bold", min_width=8, no_wrap=True)
    table.add_column("Type", min_width=5, no_wrap=True)
    table.add_column("P/L %", justify="right", min_width=7)
    table.add_column("Net", min_width=10)
    table.add_column("Regime Signals", ratio=2)

    type_colors = {"equity": "blue", "call": "green", "put": "red", "short_equity": "magenta"}

    for p in impacts:
        display_sym = p.symbol if len(p.symbol) <= 12 else f"{p.underlying} {p.position_type[0].upper()}"

        pl_color = "green" if p.unrealized_plpc >= 0 else "red"
        pl_str = f"[{pl_color}]{p.unrealized_plpc:+.1f}%[/{pl_color}]"

        tc = type_colors.get(p.position_type, "white")
        type_str = f"[{tc}]{p.position_type.upper()}[/{tc}]"

        if p.impacts:
            net_color = "green" if p.net_signal == "TAILWIND" else "red" if p.net_signal == "HEADWIND" else "yellow"
            net_str = f"[{net_color}]{p.net_signal}[/{net_color}]"

            signals = []
            for di in p.impacts:
                sig_color = "green" if di.effective_signal == "TAILWIND" else "red"
                signals.append(f"[{sig_color}]{di.effective_signal}[/{sig_color}] {di.domain}")
            signals_str = " · ".join(signals)
        else:
            net_str = "[dim]—[/dim]"
            signals_str = "[dim]—[/dim]"

        table.add_row(display_sym, type_str, pl_str, net_str, signals_str)

    console.print()
    console.print(table)

    summary_parts = [f"[bold]{len(impacts)}[/bold] positions"]
    if tw_positions:
        summary_parts.append(f"[green]{tw_positions} tailwind[/green]")
    if hw_positions:
        summary_parts.append(f"[red]{hw_positions} headwind[/red]")
    uncorrelated = len(impacts) - len(matched)
    if uncorrelated:
        summary_parts.append(f"[dim]{uncorrelated} uncorrelated[/dim]")
    console.print("  " + "  |  ".join(summary_parts))


def show_book_impact(domain: str | None = None) -> None:
    """One-liner for `lox regime <pillar> --book`.

    Builds unified regime state, fetches Alpaca positions, scores them,
    optionally filters to a single domain, and prints the table.
    """
    from rich.console import Console as _Console

    from lox.config import load_settings

    console = _Console()
    settings = load_settings()

    try:
        from lox.regimes import build_unified_regime_state

        state = build_unified_regime_state(settings=settings)
    except Exception as exc:
        console.print(f"[dim]Book impact unavailable: {exc}[/dim]")
        return

    impacts = analyze_book_impact(settings, state)
    if not impacts:
        console.print("[dim]No open positions found.[/dim]")
        return

    if domain:
        display: list[PositionImpact] = []
        for p in impacts:
            domain_hits = [di for di in p.impacts if di.domain == domain]
            p_copy = copy(p)
            p_copy.impacts = domain_hits
            display.append(p_copy)
        render_book_impact(console, display)
    else:
        render_book_impact(console, impacts)


def format_book_impact_for_llm(impacts: list[PositionImpact]) -> str:
    """Format book impact as markdown for LLM system prompt injection."""
    if not impacts:
        return ""

    matched = [p for p in impacts if p.impacts]
    if not matched:
        return ""

    lines = [
        "## Open Positions (Book Impact)",
        "The portfolio has the following regime-correlated positions.",
        "When discussing trades, reference whether the current regime supports or threatens them.",
        "",
    ]

    for p in matched:
        pos_label = f"{p.underlying} ({p.position_type}, {p.unrealized_plpc:+.1f}%)"
        signal_parts = []
        for di in p.impacts:
            signal_parts.append(f"{di.effective_signal} from {di.domain}")
        signals = ", ".join(signal_parts)
        lines.append(f"- **{pos_label}** — {signals}")

    unmatched = [p for p in impacts if not p.impacts]
    if unmatched:
        lines.append("")
        tickers = ", ".join(p.underlying for p in unmatched[:10])
        lines.append(f"Uncorrelated positions (no regime mapping): {tickers}")

    return "\n".join(lines)
