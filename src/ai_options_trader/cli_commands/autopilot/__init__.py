"""
Autopilot command package - modular trading automation.

Modules:
    positions: Position analysis and health classification
    execution: Safe trade execution with guards
    utils: Legacy utilities (deprecated, use positions module)

Author: Lox Capital Research
"""
from .positions import (
    PositionStatus,
    PositionSummary,
    analyze_positions,
    get_stop_candidates,
    get_expiring_options,
    compute_portfolio_pnl,
    to_float,
)

from .execution import (
    ExecutionMode,
    ExecutionGuard,
    TradeOrder,
    ExecutionResult,
    create_close_order,
)

# Legacy compatibility
from .utils import stop_candidates

__all__ = [
    # Position analysis
    "PositionStatus",
    "PositionSummary", 
    "analyze_positions",
    "get_stop_candidates",
    "get_expiring_options",
    "compute_portfolio_pnl",
    "to_float",
    # Execution
    "ExecutionMode",
    "ExecutionGuard",
    "TradeOrder",
    "ExecutionResult",
    "create_close_order",
    # Legacy
    "stop_candidates",
]
