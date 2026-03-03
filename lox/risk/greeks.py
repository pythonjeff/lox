"""
Portfolio Greeks aggregation engine.

Fetches all open positions from Alpaca, enriches option positions with
per-contract Greeks from the option chain, and aggregates to give a
portfolio-level Greeks summary.

Greek conventions:
  - Equity delta = qty (1 delta per share)
  - Option position delta = per_contract_delta × qty × 100
  - Same for gamma, theta, vega
  - Negative qty = short position (sign flips naturally)

Author: Lox Capital Research
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from lox.config import Settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PositionGreeks:
    """Greeks for a single position (equity or option)."""
    symbol: str             # raw symbol (OCC for options, ticker for equity)
    display_name: str       # human-readable (e.g., "AAPL 200C 3/21")
    underlying: str
    position_type: str      # "equity", "short_equity", "call", "put"
    qty: float
    delta: float            # position-level (aggregated)
    gamma: float
    theta: float            # $ per day
    vega: float
    iv: float | None
    current_price: float
    market_value: float
    unrealized_pl: float


@dataclass
class UnderlyingExposure:
    """Aggregated exposure for a single underlying ticker."""
    underlying: str
    equity_delta: float
    options_delta: float
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float


@dataclass
class PortfolioGreeks:
    """Full portfolio Greeks summary."""
    positions: list[PositionGreeks]
    by_underlying: list[UnderlyingExposure]
    net_delta: float
    net_gamma: float
    net_theta: float        # daily $ decay across portfolio
    net_vega: float
    account_equity: float
    buying_power: float
    options_bp: float | None
    risk_signals: list[str] = field(default_factory=list)
    asof: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "asof": self.asof,
            "account": {
                "equity": self.account_equity,
                "buying_power": self.buying_power,
                "options_bp": self.options_bp,
            },
            "portfolio_greeks": {
                "net_delta": round(self.net_delta, 2),
                "net_gamma": round(self.net_gamma, 4),
                "net_theta": round(self.net_theta, 2),
                "net_vega": round(self.net_vega, 2),
            },
            "by_underlying": [
                {
                    "underlying": u.underlying,
                    "equity_delta": round(u.equity_delta, 2),
                    "options_delta": round(u.options_delta, 2),
                    "net_delta": round(u.net_delta, 2),
                    "net_gamma": round(u.net_gamma, 4),
                    "net_theta": round(u.net_theta, 2),
                    "net_vega": round(u.net_vega, 2),
                }
                for u in self.by_underlying
            ],
            "positions": [
                {
                    "symbol": p.symbol,
                    "display_name": p.display_name,
                    "underlying": p.underlying,
                    "position_type": p.position_type,
                    "qty": p.qty,
                    "delta": round(p.delta, 2),
                    "gamma": round(p.gamma, 4),
                    "theta": round(p.theta, 2),
                    "vega": round(p.vega, 2),
                    "iv": round(p.iv, 4) if p.iv is not None else None,
                    "market_value": round(p.market_value, 2),
                    "unrealized_pl": round(p.unrealized_pl, 2),
                }
                for p in self.positions
            ],
            "risk_signals": self.risk_signals,
        }


# ─────────────────────────────────────────────────────────────────────
# OCC symbol parsing
# ─────────────────────────────────────────────────────────────────────

_OCC_RE = re.compile(
    r"^([A-Z]+)"         # underlying (1+ uppercase letters)
    r"(\d{6})"           # date YYMMDD
    r"([CP])"            # call or put
    r"(\d{8})$"          # strike × 1000 (8 digits)
)


def parse_occ_symbol(symbol: str) -> dict[str, Any] | None:
    """Parse an OCC option symbol into its components.

    Example: AAPL250321C00200000 → {underlying: AAPL, expiry: 2025-03-21,
             opt_type: call, strike: 200.0}
    """
    m = _OCC_RE.match(symbol.upper())
    if not m:
        return None

    underlying, date_str, cp, strike_raw = m.groups()
    try:
        yy, mm, dd = int(date_str[:2]), int(date_str[2:4]), int(date_str[4:])
        expiry = date(2000 + yy, mm, dd)
    except ValueError:
        return None

    return {
        "underlying": underlying,
        "expiry": expiry,
        "opt_type": "call" if cp == "C" else "put",
        "strike": int(strike_raw) / 1000.0,
    }


def _display_name(symbol: str, parsed: dict[str, Any] | None) -> str:
    """Build a human-readable display name for a position."""
    if parsed is None:
        return symbol

    strike = parsed["strike"]
    strike_str = f"{strike:.0f}" if strike == int(strike) else f"{strike:.1f}"
    cp = "C" if parsed["opt_type"] == "call" else "P"
    exp = parsed["expiry"]
    return f"{parsed['underlying']} {strike_str}{cp} {exp.month}/{exp.day}"


# ─────────────────────────────────────────────────────────────────────
# Safe float helper
# ─────────────────────────────────────────────────────────────────────

def _safe_float(x: Any) -> float:
    try:
        return float(x) if x is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


# ─────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────

def compute_portfolio_greeks(settings: Settings) -> PortfolioGreeks:
    """Fetch all open positions, enrich with Greeks, and aggregate.

    Returns a PortfolioGreeks with per-position, per-underlying, and
    portfolio-level Greek exposure.
    """
    from lox.data.alpaca import make_clients, fetch_option_chain

    trading, data_client = make_clients(settings)

    # ── Account info ──────────────────────────────────────────────────
    acct = trading.get_account()
    equity = _safe_float(getattr(acct, "equity", None))
    bp = _safe_float(getattr(acct, "buying_power", None))
    opt_bp_raw = getattr(acct, "options_buying_power", None)
    opt_bp = float(opt_bp_raw) if opt_bp_raw is not None else None

    # ── Fetch all positions ───────────────────────────────────────────
    raw_positions = trading.get_all_positions()

    if not raw_positions:
        return PortfolioGreeks(
            positions=[], by_underlying=[], net_delta=0, net_gamma=0,
            net_theta=0, net_vega=0, account_equity=equity,
            buying_power=bp, options_bp=opt_bp,
            risk_signals=["No open positions."],
            asof=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )

    # ── Separate equity vs option positions ───────────────────────────
    equity_positions: list[dict[str, Any]] = []
    option_positions: list[dict[str, Any]] = []

    for p in raw_positions:
        sym = str(getattr(p, "symbol", ""))
        qty = _safe_float(getattr(p, "qty", None))
        pos_data = {
            "symbol": sym,
            "qty": qty,
            "current_price": _safe_float(getattr(p, "current_price", None)),
            "avg_entry_price": _safe_float(getattr(p, "avg_entry_price", None)),
            "market_value": _safe_float(getattr(p, "market_value", None)),
            "unrealized_pl": _safe_float(getattr(p, "unrealized_pl", None)),
            "unrealized_plpc": _safe_float(getattr(p, "unrealized_plpc", None)),
            "asset_class": str(getattr(p, "asset_class", "")),
        }

        parsed = parse_occ_symbol(sym)
        if parsed is not None:
            pos_data["parsed"] = parsed
            option_positions.append(pos_data)
        else:
            equity_positions.append(pos_data)

    # ── Build equity PositionGreeks (delta = qty, no other Greeks) ────
    all_positions: list[PositionGreeks] = []

    for pos in equity_positions:
        qty = pos["qty"]
        ptype = "short_equity" if qty < 0 else "equity"
        all_positions.append(PositionGreeks(
            symbol=pos["symbol"],
            display_name=f"{pos['symbol']} (equity)",
            underlying=pos["symbol"].upper(),
            position_type=ptype,
            qty=qty,
            delta=qty,  # 1 delta per share
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            iv=None,
            current_price=pos["current_price"],
            market_value=pos["market_value"],
            unrealized_pl=pos["unrealized_pl"],
        ))

    # ── Fetch option chain Greeks per underlying ──────────────────────
    # Group option positions by underlying
    by_ul: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pos in option_positions:
        ul = pos["parsed"]["underlying"]
        by_ul[ul].append(pos)

    feed = settings.alpaca_options_feed or None

    for underlying, positions in by_ul.items():
        # Fetch the full chain for this underlying
        chain_greeks: dict[str, dict[str, float | None]] = {}
        try:
            chain = fetch_option_chain(data_client, underlying, feed=feed)
            if chain:
                for sym, snap in chain.items():
                    greeks = getattr(snap, "greeks", None)
                    chain_greeks[sym] = {
                        "delta": _safe_float(getattr(greeks, "delta", None)) if greeks else 0.0,
                        "gamma": _safe_float(getattr(greeks, "gamma", None)) if greeks else 0.0,
                        "theta": _safe_float(getattr(greeks, "theta", None)) if greeks else 0.0,
                        "vega": _safe_float(getattr(greeks, "vega", None)) if greeks else 0.0,
                        "iv": (
                            float(getattr(snap, "implied_volatility", None) or getattr(snap, "iv", None) or 0)
                            if getattr(snap, "implied_volatility", None) or getattr(snap, "iv", None)
                            else None
                        ),
                    }
        except Exception as e:
            logger.warning(f"Failed to fetch option chain for {underlying}: {e}")

        # Match each position to its chain entry
        for pos in positions:
            sym = pos["symbol"]
            qty = pos["qty"]
            parsed = pos["parsed"]

            greeks = chain_greeks.get(sym, {})
            contract_delta = greeks.get("delta", 0.0) or 0.0
            contract_gamma = greeks.get("gamma", 0.0) or 0.0
            contract_theta = greeks.get("theta", 0.0) or 0.0
            contract_vega = greeks.get("vega", 0.0) or 0.0
            iv = greeks.get("iv")

            # Position-level = per-contract × qty × 100 (contract multiplier)
            multiplier = qty * 100
            all_positions.append(PositionGreeks(
                symbol=sym,
                display_name=_display_name(sym, parsed),
                underlying=underlying,
                position_type=parsed["opt_type"],
                qty=qty,
                delta=contract_delta * multiplier,
                gamma=contract_gamma * multiplier,
                theta=contract_theta * multiplier,
                vega=contract_vega * multiplier,
                iv=iv,
                current_price=pos["current_price"],
                market_value=pos["market_value"],
                unrealized_pl=pos["unrealized_pl"],
            ))

    # ── Aggregate by underlying ───────────────────────────────────────
    ul_map: dict[str, UnderlyingExposure] = {}
    for pg in all_positions:
        ul = pg.underlying
        if ul not in ul_map:
            ul_map[ul] = UnderlyingExposure(
                underlying=ul,
                equity_delta=0.0,
                options_delta=0.0,
                net_delta=0.0,
                net_gamma=0.0,
                net_theta=0.0,
                net_vega=0.0,
            )
        exp = ul_map[ul]

        if pg.position_type in ("equity", "short_equity"):
            exp.equity_delta += pg.delta
        else:
            exp.options_delta += pg.delta

        exp.net_delta = exp.equity_delta + exp.options_delta
        exp.net_gamma += pg.gamma
        exp.net_theta += pg.theta
        exp.net_vega += pg.vega

    by_underlying = sorted(
        ul_map.values(),
        key=lambda x: abs(x.net_delta),
        reverse=True,
    )

    # ── Portfolio totals ──────────────────────────────────────────────
    net_delta = sum(p.delta for p in all_positions)
    net_gamma = sum(p.gamma for p in all_positions)
    net_theta = sum(p.theta for p in all_positions)
    net_vega = sum(p.vega for p in all_positions)

    # ── Risk signals ──────────────────────────────────────────────────
    signals = _generate_risk_signals(
        net_delta, net_gamma, net_theta, net_vega, equity,
    )

    return PortfolioGreeks(
        positions=all_positions,
        by_underlying=by_underlying,
        net_delta=net_delta,
        net_gamma=net_gamma,
        net_theta=net_theta,
        net_vega=net_vega,
        account_equity=equity,
        buying_power=bp,
        options_bp=opt_bp,
        risk_signals=signals,
        asof=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


# ─────────────────────────────────────────────────────────────────────
# Risk signal generation
# ─────────────────────────────────────────────────────────────────────

def _generate_risk_signals(
    net_delta: float,
    net_gamma: float,
    net_theta: float,
    net_vega: float,
    equity: float,
) -> list[str]:
    """Generate human-readable risk warnings based on portfolio Greeks."""
    signals: list[str] = []

    # Delta exposure
    if net_delta > 200:
        signals.append(f"[yellow]⚠[/yellow]  Net long delta ({net_delta:+.0f}) — vulnerable to market pullback")
    elif net_delta < -200:
        signals.append(f"[yellow]⚠[/yellow]  Net short delta ({net_delta:+.0f}) — exposed to market rally")
    elif abs(net_delta) < 50:
        signals.append(f"[green]✓[/green]  Delta near neutral ({net_delta:+.0f}) — well hedged directionally")
    else:
        direction = "long" if net_delta > 0 else "short"
        signals.append(f"[dim]→[/dim]  Net {direction} delta ({net_delta:+.0f}) — moderate directional exposure")

    # Gamma
    if net_gamma < -5:
        signals.append(f"[red]⚠[/red]  Short gamma ({net_gamma:+.2f}) — accelerating losses on big moves")
    elif net_gamma > 5:
        signals.append(f"[green]✓[/green]  Long gamma ({net_gamma:+.2f}) — convexity in your favor")

    # Theta
    if net_theta < -50:
        signals.append(f"[yellow]⚠[/yellow]  Significant theta decay (${net_theta:+.0f}/day) — time working against you")
    elif net_theta > 50:
        signals.append(f"[green]✓[/green]  Theta positive (${net_theta:+.0f}/day) — earning from time decay")
    elif net_theta != 0:
        signals.append(f"[dim]→[/dim]  Daily theta: ${net_theta:+.1f}/day")

    # Vega
    if net_vega > 100:
        signals.append(f"[yellow]⚠[/yellow]  Long vega ({net_vega:+.0f}) — benefits from vol expansion, hurts on crush")
    elif net_vega < -100:
        signals.append(f"[red]⚠[/red]  Short vega ({net_vega:+.0f}) — exposed to volatility spike")
    elif net_vega != 0:
        direction = "long" if net_vega > 0 else "short"
        signals.append(f"[dim]→[/dim]  Net {direction} vega ({net_vega:+.0f}) — modest vol exposure")

    # Delta as % of equity
    if equity > 0:
        delta_pct = abs(net_delta) / equity * 100
        if delta_pct > 1.0:
            signals.append(
                f"[yellow]⚠[/yellow]  Delta/equity ratio: {delta_pct:.1f}% — "
                f"${abs(net_delta):,.0f} directional per $1 move"
            )

    return signals
