"""
Trade execution logic for autopilot.

Provides safe, guarded execution with:
- Paper/live mode separation
- Confirmation prompts
- Order validation
- Execution logging

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Any, Callable
from enum import Enum


class ExecutionMode(str, Enum):
    """Trading execution mode."""
    DRY_RUN = "dry_run"     # No actual orders
    PAPER = "paper"         # Alpaca paper trading
    LIVE = "live"           # Live trading (guarded)


@dataclass
class TradeOrder:
    """Represents a trade to be executed."""
    symbol: str
    side: str               # "buy" or "sell"
    quantity: int
    order_type: str         # "market", "limit"
    limit_price: Optional[float] = None
    time_in_force: str = "day"
    
    # Metadata
    is_option: bool = False
    reason: str = ""
    source_sleeve: str = ""
    
    def __str__(self) -> str:
        px = f" @ ${self.limit_price:.2f}" if self.limit_price else ""
        return f"{self.side.upper()} {self.quantity}x {self.symbol}{px}"


@dataclass
class ExecutionResult:
    """Result of a trade execution attempt."""
    order: TradeOrder
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ExecutionGuard:
    """
    Safe execution wrapper with mode guards and confirmations.
    
    Usage:
        guard = ExecutionGuard(trading_client, mode=ExecutionMode.PAPER)
        result = guard.execute(order, confirm_fn=typer.confirm)
    """
    
    def __init__(
        self,
        trading_client: Any,
        mode: ExecutionMode = ExecutionMode.DRY_RUN,
        alpaca_paper: bool = True,
    ):
        self.trading = trading_client
        self.mode = mode
        self.alpaca_paper = alpaca_paper
        self._execution_log: List[ExecutionResult] = []
    
    @property
    def is_live(self) -> bool:
        """Check if we're in live trading mode."""
        return self.mode == ExecutionMode.LIVE and not self.alpaca_paper
    
    @property
    def requires_confirmation(self) -> bool:
        """Check if execution requires user confirmation."""
        return self.mode in (ExecutionMode.PAPER, ExecutionMode.LIVE)
    
    def validate_mode(self) -> tuple[bool, str]:
        """
        Validate execution mode against environment.
        
        Returns:
            (is_valid, error_message)
        """
        if self.mode == ExecutionMode.LIVE and self.alpaca_paper:
            return False, (
                "LIVE mode requested but ALPACA_PAPER=true. "
                "Set ALPACA_PAPER=false for live trading."
            )
        
        if self.mode == ExecutionMode.PAPER and not self.alpaca_paper:
            return False, (
                "PAPER mode requested but ALPACA_PAPER=false. "
                "Either set ALPACA_PAPER=true or use --live flag."
            )
        
        return True, ""
    
    def execute(
        self,
        order: TradeOrder,
        confirm_fn: Optional[Callable[[str], bool]] = None,
    ) -> ExecutionResult:
        """
        Execute a trade order with appropriate guards.
        
        Args:
            order: The trade order to execute
            confirm_fn: Confirmation callback (e.g., typer.confirm)
        
        Returns:
            ExecutionResult with success/failure details
        """
        # Validate mode
        valid, error = self.validate_mode()
        if not valid:
            result = ExecutionResult(order=order, success=False, message=error)
            self._execution_log.append(result)
            return result
        
        # Dry run - just log
        if self.mode == ExecutionMode.DRY_RUN:
            result = ExecutionResult(
                order=order,
                success=True,
                message=f"DRY RUN: {order}",
            )
            self._execution_log.append(result)
            return result
        
        # Confirmation required for paper/live
        if confirm_fn is not None:
            mode_label = "LIVE" if self.is_live else "PAPER"
            prompt = f"Execute {order}? [{mode_label}]"
            
            if not confirm_fn(prompt):
                result = ExecutionResult(
                    order=order,
                    success=False,
                    message="User cancelled",
                )
                self._execution_log.append(result)
                return result
        
        # Execute the order
        try:
            if order.is_option:
                response = self._submit_option_order(order)
            else:
                response = self._submit_equity_order(order)
            
            order_id = getattr(response, "id", str(response))
            result = ExecutionResult(
                order=order,
                success=True,
                order_id=order_id,
                message=f"Submitted: {order_id}",
            )
        except Exception as e:
            result = ExecutionResult(
                order=order,
                success=False,
                message=f"{type(e).__name__}: {e}",
            )
        
        self._execution_log.append(result)
        return result
    
    def _submit_option_order(self, order: TradeOrder) -> Any:
        """Submit option order via Alpaca."""
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        
        side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if order.time_in_force == "day" else TimeInForce.GTC
        
        req = LimitOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            time_in_force=tif,
            limit_price=order.limit_price,
            order_class=OrderClass.SIMPLE,
        )
        
        return self.trading.submit_order(req)
    
    def _submit_equity_order(self, order: TradeOrder) -> Any:
        """Submit equity order via Alpaca."""
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if order.time_in_force == "day" else TimeInForce.GTC
        
        if order.order_type == "limit" and order.limit_price:
            req = LimitOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
                limit_price=order.limit_price,
            )
        else:
            req = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
            )
        
        return self.trading.submit_order(req)
    
    def get_execution_log(self) -> List[ExecutionResult]:
        """Get all execution results from this session."""
        return self._execution_log.copy()
    
    def get_summary(self) -> dict:
        """Get execution summary statistics."""
        total = len(self._execution_log)
        successes = sum(1 for r in self._execution_log if r.success)
        failures = total - successes
        
        return {
            "total_orders": total,
            "successful": successes,
            "failed": failures,
            "mode": self.mode.value,
        }


def create_close_order(position: Any) -> TradeOrder:
    """
    Create an order to close an existing position.
    
    Args:
        position: PositionSummary or similar object with symbol, quantity
    
    Returns:
        TradeOrder configured to close the position
    """
    symbol = getattr(position, "symbol", str(position))
    qty = abs(getattr(position, "quantity", 1))
    is_option = "/" in symbol or len(symbol) > 12
    
    # Closing = sell if long, buy if short
    current_qty = getattr(position, "quantity", 1)
    side = "sell" if current_qty > 0 else "buy"
    
    return TradeOrder(
        symbol=symbol,
        side=side,
        quantity=int(qty),
        order_type="market",
        is_option=is_option,
        reason="close_position",
    )
