"""Aster DEX trading client.

Web3 ECDSA-authenticated order execution: market, limit, stop orders.
Adapted from alpha-arena-clone/src/aster_data_fetcher.py (trading methods).
"""
from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

import requests

from lox.config import Settings

logger = logging.getLogger(__name__)

ASTER_BASE_URL = "https://fapi.asterdex.com"


class AsterClient:
    """Authenticated Aster DEX trading client."""

    def __init__(self, settings: Settings):
        self.base_url = ASTER_BASE_URL
        self.user_address = settings.ASTER_USER_ADDRESS
        self.signer_address = settings.ASTER_SIGNER_ADDRESS
        self.private_key = settings.ASTER_PRIVATE_KEY
        self._exchange_info_cache: dict | None = None
        self._symbol_filters_cache: dict[str, dict] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self.user_address and self.signer_address and self.private_key)

    # ------------------------------------------------------------------
    # Symbol helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_symbol(coin: str) -> str:
        """Convert coin to Aster symbol (e.g. BTC -> BTCUSDT)."""
        coin = coin.split("/")[0] if "/" in coin else coin
        coin = coin.split(":")[0] if ":" in coin else coin
        coin = coin.replace(" PERP", "").strip()
        return f"{coin.upper()}USDT"

    # ------------------------------------------------------------------
    # Auth / signing
    # ------------------------------------------------------------------

    def _require_auth(self) -> None:
        if not self.is_configured:
            raise ValueError(
                "Aster DEX not configured. Set ASTER_USER_ADDRESS, ASTER_SIGNER_ADDRESS, "
                "and ASTER_PRIVATE_KEY in your .env file."
            )

    @staticmethod
    def _trim_dict(d: dict) -> dict:
        """Convert all values to strings for signing."""
        for key in d:
            value = d[key]
            if isinstance(value, list):
                new_value = []
                for item in value:
                    if isinstance(item, dict):
                        new_value.append(json.dumps(AsterClient._trim_dict(item)))
                    else:
                        new_value.append(str(item))
                d[key] = json.dumps(new_value)
            elif isinstance(value, dict):
                d[key] = json.dumps(AsterClient._trim_dict(value))
            else:
                d[key] = str(value)
        return d

    def _sign(self, params: dict) -> dict:
        """Generate Web3 ECDSA signature for authenticated requests."""
        from eth_abi import encode
        from eth_account import Account
        from eth_account.messages import encode_defunct
        from web3 import Web3

        self._require_auth()

        params = {k: v for k, v in params.items() if v is not None}
        params["recvWindow"] = "50000"
        params["timestamp"] = str(int(round(time.time() * 1000)))

        nonce = math.trunc(time.time() * 1000000)

        params = self._trim_dict(params)
        json_str = json.dumps(params, sort_keys=True).replace(" ", "").replace("'", '"')

        encoded = encode(
            ["string", "address", "address", "uint256"],
            [json_str, self.user_address, self.signer_address, nonce],
        )

        keccak_hex = Web3.keccak(encoded).hex()
        signable_msg = encode_defunct(hexstr=keccak_hex)
        signed = Account.sign_message(signable_message=signable_msg, private_key=self.private_key)
        signature = "0x" + signed.signature.hex()

        params["nonce"] = str(nonce)
        params["user"] = self.user_address
        params["signer"] = self.signer_address
        params["signature"] = signature

        return params

    def _request(self, method: str, endpoint: str, params: dict | None = None) -> Any:
        """Make an authenticated request to the Aster API."""
        params = params or {}
        signed = self._sign(params)
        url = f"{self.base_url}{endpoint}"

        if method == "GET":
            resp = requests.get(url, params=signed, timeout=10)
        elif method == "POST":
            resp = requests.post(url, data=signed, timeout=10)
        elif method == "DELETE":
            resp = requests.delete(url, params=signed, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            logger.error(f"Aster API {resp.status_code}: {detail}")
            resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Exchange info / rounding
    # ------------------------------------------------------------------

    def _fetch_exchange_info(self) -> dict:
        if self._exchange_info_cache is not None:
            return self._exchange_info_cache
        try:
            resp = requests.get(f"{self.base_url}/fapi/v3/exchangeInfo", timeout=10)
            resp.raise_for_status()
            self._exchange_info_cache = resp.json()
            return self._exchange_info_cache
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            return {}

    def _get_symbol_filters(self, symbol: str) -> dict:
        if symbol in self._symbol_filters_cache:
            return self._symbol_filters_cache[symbol]

        info = self._fetch_exchange_info()
        for sym in info.get("symbols", []):
            if sym.get("symbol") == symbol:
                tick_size = step_size = None
                for f in sym.get("filters", []):
                    if f.get("filterType") == "PRICE_FILTER":
                        tick_size = float(f.get("tickSize", 0.01))
                    elif f.get("filterType") == "LOT_SIZE":
                        step_size = float(f.get("stepSize", 0.001))
                result = {"tickSize": tick_size or 0.01, "stepSize": step_size or 0.001}
                self._symbol_filters_cache[symbol] = result
                return result

        default = {"tickSize": 0.01, "stepSize": 0.001}
        self._symbol_filters_cache[symbol] = default
        return default

    @staticmethod
    def _decimals_for(step: float) -> int:
        if step >= 1:
            return 0
        elif step >= 0.1:
            return 1
        elif step >= 0.01:
            return 2
        elif step >= 0.001:
            return 3
        elif step >= 0.0001:
            return 4
        elif step >= 0.00001:
            return 5
        return 6

    def round_price(self, price: float, coin: str) -> float:
        symbol = self._to_symbol(coin)
        tick = self._get_symbol_filters(symbol)["tickSize"]
        rounded = round(price / tick) * tick
        return round(rounded, self._decimals_for(tick))

    def round_quantity(self, quantity: float, coin: str) -> float:
        symbol = self._to_symbol(coin)
        step = self._get_symbol_filters(symbol)["stepSize"]
        rounded = round(quantity / step) * step
        return round(rounded, self._decimals_for(step))

    # ------------------------------------------------------------------
    # Public market data (no auth)
    # ------------------------------------------------------------------

    def get_price(self, coin: str) -> float:
        """Get current mark price for a coin."""
        symbol = self._to_symbol(coin)
        try:
            resp = requests.get(
                f"{self.base_url}/fapi/v3/ticker/price",
                params={"symbol": symbol},
                timeout=5,
            )
            resp.raise_for_status()
            return float(resp.json()["price"])
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    def place_order(
        self,
        coin: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
        stop_price: float | None = None,
        leverage: int | None = None,
        reduce_only: bool = False,
    ) -> dict:
        """Place an order on Aster DEX."""
        self._require_auth()

        if leverage is not None:
            self.set_leverage(coin, leverage)

        symbol = self._to_symbol(coin)
        quantity = self.round_quantity(quantity, coin)
        if quantity <= 0:
            filters = self._get_symbol_filters(symbol)
            raise ValueError(f"Quantity too small for {coin}. Min step: {filters['stepSize']}")

        # Check minimum notional ($5)
        current_price = self.get_price(coin)
        if current_price > 0 and quantity * current_price < 5.0:
            raise ValueError(
                f"Order notional ${quantity * current_price:.2f} below $5 minimum. "
                f"Need at least {5.0 / current_price:.6f} {coin}."
            )

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "positionSide": "BOTH",
            "quantity": str(quantity),
        }

        if reduce_only:
            params["reduceOnly"] = "true"

        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price required for LIMIT orders")
            params["price"] = str(self.round_price(price, coin))
            params["timeInForce"] = "GTC"
        elif order_type.upper() in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
            if stop_price is None:
                raise ValueError(f"Stop price required for {order_type} orders")
            params["stopPrice"] = str(self.round_price(stop_price, coin))
            params["workingType"] = "MARK_PRICE"

        result = self._request("POST", "/fapi/v3/order", params)
        logger.info(f"Order placed: {side} {quantity} {coin} @ {order_type}")
        return result

    def close_position(self, coin: str) -> dict:
        """Close an open position by placing a reduce-only market order."""
        self._require_auth()
        positions = self.get_positions()
        symbol = self._to_symbol(coin)

        position = None
        for pos in positions:
            if pos["symbol"] == symbol:
                position = pos
                break

        if not position:
            return {"status": "no_position", "message": f"No open position for {coin}"}

        amt = float(position.get("positionAmt", 0))
        if amt == 0:
            return {"status": "already_closed"}

        side = "SELL" if amt > 0 else "BUY"
        return self.place_order(coin=coin, side=side, quantity=abs(amt), reduce_only=True)

    def get_positions(self) -> list:
        """Get current open positions."""
        self._require_auth()
        result = self._request("GET", "/fapi/v3/positionRisk", {})
        return [pos for pos in result if float(pos.get("positionAmt", 0)) != 0]

    def get_balance(self) -> list:
        """Get account balances."""
        self._require_auth()
        return self._request("GET", "/fapi/v3/balance", {})

    def get_account_state(self) -> dict:
        """Get comprehensive account state (balance, equity, unrealized PnL)."""
        self._require_auth()
        info = self._request("GET", "/fapi/v3/account", {})
        return {
            "totalWalletBalance": float(info.get("totalWalletBalance", 0)),
            "availableBalance": float(info.get("availableBalance", 0)),
            "totalUnrealizedProfit": float(info.get("totalUnrealizedProfit", 0)),
            "totalMarginBalance": float(info.get("totalMarginBalance", 0)),
        }

    def set_leverage(self, coin: str, leverage: int) -> dict:
        """Set leverage for a symbol (1-125x)."""
        self._require_auth()
        symbol = self._to_symbol(coin)
        return self._request("POST", "/fapi/v3/leverage", {"symbol": symbol, "leverage": str(leverage)})

    def cancel_order(self, coin: str, order_id: int) -> dict:
        """Cancel a specific order."""
        self._require_auth()
        symbol = self._to_symbol(coin)
        return self._request("DELETE", "/fapi/v3/order", {"symbol": symbol, "orderId": order_id})

    def get_open_orders(self, coin: str | None = None) -> list:
        """Get open orders, optionally filtered by coin."""
        self._require_auth()
        params = {}
        if coin:
            params["symbol"] = self._to_symbol(coin)
        return self._request("GET", "/fapi/v3/openOrders", params)
