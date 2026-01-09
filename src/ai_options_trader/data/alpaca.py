from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, TYPE_CHECKING

from ai_options_trader.config import Settings

if TYPE_CHECKING:
    # Only for type-checking; keep runtime imports lazy to avoid hard dependency
    # during test collection or constrained environments.
    from alpaca.data.historical import OptionHistoricalDataClient
    from alpaca.trading.client import TradingClient

@dataclass(frozen=True)
class OptionCandidate:
    symbol: str
    opt_type: str
    expiry: date
    strike: float
    dte_days: int
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    iv: float | None
    oi: int | None
    volume: int | None
    bid: float | None
    ask: float | None
    last: float | None

    @property
    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return None

def _first_attr(obj: object, paths: list[tuple[str, ...]]) -> object | None:
    for path in paths:
        cur: object | None = obj
        for p in path:
            if cur is None:
                break
            cur = getattr(cur, p, None)
        if cur is not None:
            return cur
    return None

def make_clients(settings: Settings):
    # Lazy import so core library can be imported without alpaca SDK installed.
    from alpaca.data.historical import OptionHistoricalDataClient
    from alpaca.trading.client import TradingClient

    trading = TradingClient(settings.alpaca_api_key, settings.alpaca_api_secret, paper=settings.alpaca_paper)
    data_key = settings.alpaca_data_key or settings.alpaca_api_key
    data_secret = settings.alpaca_data_secret or settings.alpaca_api_secret
    data = OptionHistoricalDataClient(data_key, data_secret)
    return trading, data

def fetch_option_chain(data_client: Any, ticker: str, *, feed: str | None = None) -> dict[str, Any]:
    from alpaca.data.requests import OptionChainRequest

    # Alpaca SDKs have changed request shapes over time; try to pass `feed` if supported.
    try:
        req = OptionChainRequest(underlying_symbol=ticker, feed=feed) if feed else OptionChainRequest(underlying_symbol=ticker)
    except TypeError:
        req = OptionChainRequest(underlying_symbol=ticker)
    return data_client.get_option_chain(request_params=req)

def to_candidates(chain: dict[str, Any], ticker: str) -> Iterable[OptionCandidate]:
    # Alpaca returns mapping: symbol -> snapshot-like object.
    for sym, snap in chain.items():
        greeks = getattr(snap, "greeks", None)
        quote = getattr(snap, "latest_quote", None)
        trade = getattr(snap, "latest_trade", None)
        daily_bar = getattr(snap, "daily_bar", None)

        delta = getattr(greeks, "delta", None) if greeks else None
        gamma = getattr(greeks, "gamma", None) if greeks else None
        theta = getattr(greeks, "theta", None) if greeks else None
        vega = getattr(greeks, "vega", None) if greeks else None
        iv = _first_attr(snap, [("implied_volatility",), ("iv",)])
        bid = getattr(quote, "bid_price", None) if quote else None
        ask = getattr(quote, "ask_price", None) if quote else None
        last = getattr(trade, "price", None) if trade else None

        oi = _first_attr(snap, [("open_interest",), ("oi",)])
        vol = _first_attr(snap, [("volume",), ("daily_bar", "volume")])

        # leave expiry/strike/type to selector via OCC parsing
        yield OptionCandidate(
            symbol=sym,
            opt_type="",
            expiry=date(1970, 1, 1),
            strike=0.0,
            dte_days=0,
            delta=float(delta) if delta is not None else None,
            gamma=float(gamma) if gamma is not None else None,
            theta=float(theta) if theta is not None else None,
            vega=float(vega) if vega is not None else None,
            iv=float(iv) if iv is not None else None,
            oi=int(oi) if oi is not None else None,
            volume=int(vol) if vol is not None else None,
            bid=float(bid) if bid is not None else None,
            ask=float(ask) if ask is not None else None,
            last=float(last) if last is not None else None,
        )
