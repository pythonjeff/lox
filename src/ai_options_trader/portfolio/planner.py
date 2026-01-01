from __future__ import annotations

from dataclasses import dataclass

from ai_options_trader.execution.alpaca import OrderPreview
from ai_options_trader.portfolio.model import HorizonForecast


@dataclass(frozen=True)
class PortfolioPlan:
    outlook: str  # "risk_on" | "risk_off" | "mixed" | "unknown"
    risk_budget_usd: float
    risk_used_usd: float
    target_weights: dict[str, float]  # weights over risk_used_usd (sum to 1)
    orders: list[OrderPreview]
    notes: list[str]


def _confidence(p: float | None) -> float:
    if p is None:
        return 0.0
    return max(0.0, min(1.0, abs(p - 0.5) * 2.0))


def decide_outlook(forecasts: list[HorizonForecast]) -> str:
    f = {x.horizon: x for x in forecasts}
    p3 = f.get("3m").prob_up if f.get("3m") else None
    p6 = f.get("6m").prob_up if f.get("6m") else None
    p12 = f.get("12m").prob_up if f.get("12m") else None

    if p3 is None or p6 is None or p12 is None:
        return "unknown"

    # Coarse regime decision with hysteresis
    if p6 > 0.55 and p12 > 0.55:
        return "risk_on"
    if p3 < 0.45 and p6 < 0.45:
        return "risk_off"
    return "mixed"


def plan_portfolio(
    *,
    forecasts: list[HorizonForecast],
    cash_usd: float,
    max_risk_pct_cash: float = 0.50,
    liquidity_tight: bool | None = None,
    usd_strong: bool | None = None,
    latest_prices: dict[str, float],
) -> PortfolioPlan:
    """
    Convert model forecasts + regimes into a macro-oriented ETF plan.

    v1 assumptions:
    - Long-only (buys only); "risk_off" means rotating into defensives + cash, not shorting.
    - Risk is proxied by cash deployed (gross notional for buys).
    """
    notes: list[str] = []
    risk_budget = max(0.0, float(cash_usd) * float(max_risk_pct_cash))

    outlook = decide_outlook(forecasts)
    notes.append(f"Equity outlook: {outlook}")

    # Confidence scaling using 6m + 12m
    f = {x.horizon: x for x in forecasts}
    conf = 0.0
    for h in ("6m", "12m"):
        if h in f:
            conf += _confidence(f[h].prob_up)
    conf = conf / 2.0 if conf > 0 else 0.0
    # Use between 25% and 100% of risk budget
    deploy_frac = min(1.0, max(0.25, conf))

    # Liquidity tight → reduce risk deployed
    if liquidity_tight is True:
        deploy_frac = min(deploy_frac, 0.50)
        notes.append("Liquidity regime is tight → reducing deployed risk.")

    risk_used = risk_budget * deploy_frac

    # Target macro allocations
    if outlook == "risk_on":
        weights = {"SPY": 0.45, "QQQ": 0.25, "IWM": 0.10, "GLD": 0.10, "TLT": 0.10}
    elif outlook == "risk_off":
        weights = {"TLT": 0.45, "GLD": 0.25, "UUP": 0.15, "LQD": 0.15}
    elif outlook == "mixed":
        weights = {"SPY": 0.25, "TLT": 0.25, "GLD": 0.20, "UUP": 0.15, "LQD": 0.15}
    else:
        weights = {"GLD": 0.34, "TLT": 0.33, "UUP": 0.33}
        notes.append("Model forecasts incomplete → defaulting to defensive mix.")

    # USD strong → tilt more to UUP (small, bounded)
    if usd_strong is True and "UUP" in weights:
        bump = 0.05
        weights["UUP"] = min(0.30, weights["UUP"] + bump)
        # Reduce SPY first, else TLT
        if "SPY" in weights and weights["SPY"] > bump:
            weights["SPY"] -= bump
        elif "TLT" in weights and weights["TLT"] > bump:
            weights["TLT"] -= bump
        notes.append("USD regime strong → small tilt toward UUP.")

    # Normalize weights
    s = sum(weights.values())
    if s <= 0:
        weights = {"TLT": 0.5, "GLD": 0.5}
        s = 1.0
    weights = {k: float(v) / float(s) for k, v in weights.items()}

    # Build order previews (buy-only)
    orders: list[OrderPreview] = []
    for sym, w in weights.items():
        px = latest_prices.get(sym)
        if px is None or px <= 0:
            notes.append(f"Missing price for {sym}; skipping.")
            continue
        dollars = risk_used * w
        qty = int(dollars // px)
        if qty <= 0:
            continue
        orders.append(OrderPreview(symbol=sym, qty=qty, side="buy", order_type="market"))

    used_cash = 0.0
    for o in orders:
        px = latest_prices.get(o.symbol, 0.0)
        used_cash += float(o.qty) * float(px)
    # Guardrail
    used_cash = min(used_cash, risk_budget)

    return PortfolioPlan(
        outlook=outlook,
        risk_budget_usd=risk_budget,
        risk_used_usd=used_cash,
        target_weights=weights,
        orders=orders,
        notes=notes,
    )


