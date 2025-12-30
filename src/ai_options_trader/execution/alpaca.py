from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest


@dataclass(frozen=True)
class OrderPreview:
    symbol: str
    qty: int
    side: str  # "buy"|"sell"
    order_type: str  # "market"|"limit"
    limit_price: Optional[float] = None
    tif: str = "day"


def submit_option_order(
    *,
    trading: TradingClient,
    symbol: str,
    qty: int,
    side: str,
    limit_price: float | None = None,
    tif: str = "day",
):
    """
    Submit an options order to Alpaca.

    Note: For safety, call this only after explicit user confirmation.
    """
    side_enum = OrderSide.BUY if side.lower().startswith("b") else OrderSide.SELL
    tif_enum = TimeInForce.DAY if tif.lower() == "day" else TimeInForce.GTC

    if limit_price is None:
        req = MarketOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=tif_enum)
    else:
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=tif_enum,
            limit_price=round(float(limit_price), 2),
        )

    return trading.submit_order(order_data=req)




