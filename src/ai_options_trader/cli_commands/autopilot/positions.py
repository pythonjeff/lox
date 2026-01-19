"""
Position analysis and review logic for autopilot.

Provides functions for:
- Stop-loss candidate identification
- Position health assessment
- P&L tracking and attribution

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class PositionStatus(str, Enum):
    """Position health classification."""
    HEALTHY = "healthy"           # Above water, on track
    WARNING = "warning"           # Down but within tolerance
    STOP_CANDIDATE = "stop"       # Below stop-loss threshold
    EXPIRED = "expired"           # Option expired/near expiry
    ILLIQUID = "illiquid"         # Wide spreads, low volume


@dataclass
class PositionSummary:
    """Analyzed position with health metrics."""
    symbol: str
    quantity: float
    avg_entry: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    status: PositionStatus
    
    # Option-specific
    is_option: bool = False
    dte: Optional[int] = None
    underlying: Optional[str] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None
    
    # Risk flags
    flags: List[str] = None
    
    def __post_init__(self):
        if self.flags is None:
            self.flags = []
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price * (100 if self.is_option else 1)
    
    @property
    def cost_basis(self) -> float:
        """Original cost basis."""
        return self.quantity * self.avg_entry * (100 if self.is_option else 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for display/logging."""
        return {
            "symbol": self.symbol,
            "qty": self.quantity,
            "entry": f"${self.avg_entry:.2f}",
            "current": f"${self.current_price:.2f}",
            "pnl": f"${self.unrealized_pnl:+,.0f}",
            "pnl_pct": f"{self.unrealized_pnl_pct:+.1%}",
            "status": self.status.value,
            "flags": self.flags,
        }


def to_float(x: Any) -> Optional[float]:
    """Safe float conversion."""
    try:
        return float(x) if x is not None else None
    except (ValueError, TypeError):
        return None


def analyze_positions(
    raw_positions: List[Any],
    stop_loss_pct: float = 0.30,
    warning_pct: float = 0.15,
    expiry_warning_days: int = 7,
) -> List[PositionSummary]:
    """
    Analyze portfolio positions and classify health status.
    
    Args:
        raw_positions: List of Alpaca position objects
        stop_loss_pct: Threshold for stop-loss flag (e.g., 0.30 = -30%)
        warning_pct: Threshold for warning flag
        expiry_warning_days: Flag options expiring within N days
    
    Returns:
        List of PositionSummary with health classifications
    """
    summaries: List[PositionSummary] = []
    
    for pos in raw_positions:
        symbol = str(getattr(pos, "symbol", "") or "").upper()
        qty = to_float(getattr(pos, "qty", None)) or 0.0
        avg_entry = to_float(getattr(pos, "avg_entry_price", None)) or 0.0
        current = to_float(getattr(pos, "current_price", None)) or 0.0
        upl = to_float(getattr(pos, "unrealized_pl", None)) or 0.0
        uplpc = to_float(getattr(pos, "unrealized_plpc", None)) or 0.0
        
        # Determine if option
        is_option = "/" in symbol or _looks_like_option(symbol)
        
        # Parse option details if applicable
        dte = None
        underlying = None
        strike = None
        opt_type = None
        
        if is_option:
            parsed = _parse_option_details(symbol)
            if parsed:
                underlying = parsed.get("underlying")
                strike = parsed.get("strike")
                opt_type = parsed.get("type")
                expiry = parsed.get("expiry")
                if expiry:
                    dte = (expiry - datetime.now()).days
        
        # Classify status
        flags: List[str] = []
        
        if uplpc <= -abs(stop_loss_pct):
            status = PositionStatus.STOP_CANDIDATE
            flags.append(f"DOWN {uplpc:.1%}")
        elif uplpc <= -abs(warning_pct):
            status = PositionStatus.WARNING
            flags.append(f"DOWN {uplpc:.1%}")
        elif is_option and dte is not None and dte <= expiry_warning_days:
            status = PositionStatus.EXPIRED
            flags.append(f"EXPIRING in {dte}d")
        else:
            status = PositionStatus.HEALTHY
        
        # Add to summaries
        summaries.append(PositionSummary(
            symbol=symbol,
            quantity=qty,
            avg_entry=avg_entry,
            current_price=current,
            unrealized_pnl=upl,
            unrealized_pnl_pct=uplpc,
            status=status,
            is_option=is_option,
            dte=dte,
            underlying=underlying,
            strike=strike,
            option_type=opt_type,
            flags=flags,
        ))
    
    return summaries


def get_stop_candidates(
    positions: List[PositionSummary],
) -> List[PositionSummary]:
    """Filter positions at or below stop-loss threshold."""
    return [p for p in positions if p.status == PositionStatus.STOP_CANDIDATE]


def get_expiring_options(
    positions: List[PositionSummary],
    days: int = 7,
) -> List[PositionSummary]:
    """Filter options expiring within N days."""
    return [
        p for p in positions
        if p.is_option and p.dte is not None and p.dte <= days
    ]


def compute_portfolio_pnl(
    positions: List[PositionSummary],
) -> Dict[str, float]:
    """
    Compute aggregate P&L metrics.
    
    Returns:
        Dict with total_pnl, total_cost, pnl_pct, n_winners, n_losers
    """
    total_pnl = sum(p.unrealized_pnl for p in positions)
    total_cost = sum(p.cost_basis for p in positions)
    
    winners = [p for p in positions if p.unrealized_pnl > 0]
    losers = [p for p in positions if p.unrealized_pnl < 0]
    
    return {
        "total_pnl": total_pnl,
        "total_cost": total_cost,
        "pnl_pct": total_pnl / total_cost if total_cost > 0 else 0.0,
        "n_winners": len(winners),
        "n_losers": len(losers),
        "n_total": len(positions),
    }


def _looks_like_option(symbol: str) -> bool:
    """Heuristic check if symbol looks like an option."""
    return (
        len(symbol) > 10 and
        any(c.isdigit() for c in symbol) and
        any(c in symbol.upper() for c in ["C", "P"])
    )


def _parse_option_details(symbol: str) -> Optional[Dict[str, Any]]:
    """Parse option symbol for details."""
    try:
        from ai_options_trader.utils.occ import parse_occ_option_full
        return parse_occ_option_full(symbol)
    except Exception:
        return None
