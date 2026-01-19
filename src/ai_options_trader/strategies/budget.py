"""
Budget allocation and capital management.

This module implements institutional-grade budget allocation logic for
multi-asset portfolios. Supports strict and flexible allocation modes
with configurable equity/options splits.

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class AllocationMode(str, Enum):
    """Portfolio allocation strategy."""
    AUTO = "auto"           # Dynamic based on regime/volatility
    EQUITY_100 = "equity100"  # 100% equity, no options
    SPLIT_50_50 = "50_50"   # 50% equity, 50% options
    SPLIT_70_30 = "70_30"   # 70% equity, 30% options
    BOTH = "both"           # Show both plans


class BudgetMode(str, Enum):
    """Budget enforcement strategy."""
    STRICT = "strict"  # Enforce max-premium and equity/options split
    FLEX = "flex"      # Allocate full cash across ≥min trades


@dataclass
class BudgetConstraints:
    """Capital allocation constraints."""
    total_cash: float
    equity_pct: float = 0.70          # Default 70/30 split
    max_premium_per_contract: float = 100.0
    max_new_trades: int = 3
    min_new_trades: int = 2
    shares_budget_per_idea: float = 100.0
    
    @property
    def equity_budget(self) -> float:
        """Capital allocated to equity positions."""
        return self.total_cash * self.equity_pct
    
    @property
    def options_budget(self) -> float:
        """Capital allocated to options positions."""
        return self.total_cash * (1.0 - self.equity_pct)
    
    @property
    def per_trade_budget(self) -> float:
        """Average budget per trade."""
        if self.max_new_trades <= 0:
            return 0.0
        return self.total_cash / self.max_new_trades


@dataclass
class AllocationPlan:
    """Computed allocation plan for a trading session."""
    mode: AllocationMode
    budget_mode: BudgetMode
    
    # Capital breakdown
    total_available: float
    equity_allocation: float
    options_allocation: float
    
    # Per-trade limits
    max_equity_per_trade: float
    max_option_premium: float
    
    # Trade counts
    target_equity_trades: int
    target_option_trades: int
    
    # Metadata
    constraints: BudgetConstraints
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/display."""
        return {
            "mode": self.mode.value,
            "budget_mode": self.budget_mode.value,
            "total": f"${self.total_available:,.0f}",
            "equity": f"${self.equity_allocation:,.0f}",
            "options": f"${self.options_allocation:,.0f}",
            "max_equity_per_trade": f"${self.max_equity_per_trade:,.0f}",
            "max_option_premium": f"${self.max_option_premium:,.0f}",
        }


def compute_allocation(
    cash: float,
    mode: AllocationMode = AllocationMode.AUTO,
    budget_mode: BudgetMode = BudgetMode.STRICT,
    max_premium: float = 100.0,
    max_trades: int = 3,
    min_trades: int = 2,
    shares_budget: float = 100.0,
    vix_level: Optional[float] = None,
) -> AllocationPlan:
    """
    Compute optimal capital allocation given constraints.
    
    Args:
        cash: Available trading capital
        mode: Allocation strategy (auto uses regime-aware logic)
        budget_mode: Enforcement strategy (strict vs flex)
        max_premium: Maximum premium per option contract
        max_trades: Maximum new positions to open
        min_trades: Minimum new positions (best effort)
        shares_budget: Budget per equity idea
        vix_level: Current VIX (for auto mode)
    
    Returns:
        AllocationPlan with computed budgets
    
    Example:
        >>> plan = compute_allocation(10000, mode=AllocationMode.SPLIT_70_30)
        >>> print(f"Equity: ${plan.equity_allocation:,.0f}")
        Equity: $7,000
    """
    constraints = BudgetConstraints(
        total_cash=cash,
        max_premium_per_contract=max_premium,
        max_new_trades=max(1, max_trades),
        min_new_trades=max(0, min(min_trades, max_trades)),
        shares_budget_per_idea=shares_budget,
    )
    
    notes: List[str] = []
    
    # Determine equity/options split
    if mode == AllocationMode.EQUITY_100:
        equity_pct = 1.0
        notes.append("100% equity allocation (no options)")
    elif mode == AllocationMode.SPLIT_50_50:
        equity_pct = 0.50
        notes.append("50/50 equity/options split")
    elif mode == AllocationMode.SPLIT_70_30:
        equity_pct = 0.70
        notes.append("70/30 equity/options split")
    elif mode == AllocationMode.AUTO:
        # Regime-aware: increase options allocation in high-vol
        if vix_level is not None and vix_level > 25:
            equity_pct = 0.50  # More hedging in high vol
            notes.append(f"Auto: 50/50 (VIX={vix_level:.1f} > 25)")
        elif vix_level is not None and vix_level < 15:
            equity_pct = 0.80  # Risk-on in low vol
            notes.append(f"Auto: 80/20 (VIX={vix_level:.1f} < 15)")
        else:
            equity_pct = 0.70  # Default
            notes.append("Auto: 70/30 (default)")
    else:
        equity_pct = 0.70
    
    constraints.equity_pct = equity_pct
    
    # Compute allocations
    equity_alloc = cash * equity_pct
    options_alloc = cash * (1.0 - equity_pct)
    
    # Distribute across trades
    if budget_mode == BudgetMode.STRICT:
        # Strict: respect max_premium and per-trade limits
        target_equity = max(1, int(equity_alloc / shares_budget)) if equity_pct > 0 else 0
        target_options = max(1, int(options_alloc / (max_premium * 100))) if equity_pct < 1 else 0
        
        # Cap at max trades
        target_equity = min(target_equity, max_trades)
        target_options = min(target_options, max_trades)
        
        max_equity_per = equity_alloc / max(1, target_equity)
        max_opt_premium = min(max_premium, options_alloc / (100 * max(1, target_options)))
    else:
        # Flex: allocate full cash, prefer options for convexity
        target_equity = min_trades
        target_options = max(min_trades, max_trades - target_equity)
        max_equity_per = cash / max(1, max_trades)
        max_opt_premium = cash / (100 * max(1, max_trades))
        notes.append("Flex mode: full cash allocation")
    
    return AllocationPlan(
        mode=mode,
        budget_mode=budget_mode,
        total_available=cash,
        equity_allocation=equity_alloc,
        options_allocation=options_alloc,
        max_equity_per_trade=max_equity_per,
        max_option_premium=max_opt_premium,
        target_equity_trades=target_equity,
        target_option_trades=target_options,
        constraints=constraints,
        notes=notes,
    )


def validate_trade_against_budget(
    plan: AllocationPlan,
    trade_cost: float,
    is_option: bool = False,
) -> tuple[bool, str]:
    """
    Check if a proposed trade fits within the allocation plan.
    
    Returns:
        (is_valid, reason)
    """
    if is_option:
        if trade_cost > plan.max_option_premium * 100:
            return False, f"Premium ${trade_cost:.0f} exceeds max ${plan.max_option_premium * 100:.0f}"
        if trade_cost > plan.options_allocation:
            return False, f"Premium ${trade_cost:.0f} exceeds options budget ${plan.options_allocation:.0f}"
    else:
        if trade_cost > plan.max_equity_per_trade:
            return False, f"Cost ${trade_cost:.0f} exceeds per-trade max ${plan.max_equity_per_trade:.0f}"
        if trade_cost > plan.equity_allocation:
            return False, f"Cost ${trade_cost:.0f} exceeds equity budget ${plan.equity_allocation:.0f}"
    
    return True, "OK"
