"""Crypto perpetual futures data: OHLCV, OI, funding rates, technical indicators.

Uses CCXT for exchange-agnostic data fetching (OKX, Bybit, Binance).
Adapted from alpha-arena-clone/src/data_fetcher.py.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import ccxt
import pandas as pd
import pandas_ta as ta

from lox.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_COINS = ["BTC", "ETH", "SOL"]


class CryptoPerpsData:
    """Fetches crypto perpetual futures data via CCXT."""

    def __init__(self, settings: Settings):
        exchange_name = settings.CCXT_EXCHANGE or "okx"
        self.exchange = getattr(ccxt, exchange_name)(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )
        self.exchange_name = exchange_name
        self._markets_loaded = False

    def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            try:
                self.exchange.load_markets()
                self._markets_loaded = True
            except Exception:
                pass

    def _get_perp_symbol(self, symbol: str) -> str:
        """Convert symbol to perpetual futures format (e.g. BTC/USDT -> BTC/USDT:USDT)."""
        if ":" in symbol:
            return symbol
        if "/" not in symbol:
            symbol = f"{symbol}/USDT"
        return f"{symbol}:USDT"

    # ------------------------------------------------------------------
    # Raw data fetching
    # ------------------------------------------------------------------

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV candles for a perp symbol."""
        self._ensure_markets()
        perp_symbol = self._get_perp_symbol(symbol)
        try:
            ohlcv = self.exchange.fetch_ohlcv(perp_symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            raise RuntimeError(f"Error fetching {symbol} {timeframe} OHLCV: {e}") from e

    def fetch_open_interest(self, symbol: str) -> dict | None:
        """Fetch OI for a perp. Returns {'open_interest': float} or None."""
        try:
            perp_symbol = self._get_perp_symbol(symbol)
            if hasattr(self.exchange, "fetch_open_interest"):
                oi_data = self.exchange.fetch_open_interest(perp_symbol)
                return {
                    "open_interest": oi_data.get("openInterestAmount", 0),
                    "timestamp": oi_data.get("timestamp"),
                }
            return None
        except Exception as e:
            logger.debug(f"OI unavailable for {symbol}: {e}")
            return None

    def fetch_funding_rate(self, symbol: str) -> dict | None:
        """Fetch funding rate. Returns {'funding_rate': float} or None."""
        try:
            perp_symbol = self._get_perp_symbol(symbol)
            if hasattr(self.exchange, "fetch_funding_rate"):
                data = self.exchange.fetch_funding_rate(perp_symbol)
                return {
                    "funding_rate": data.get("fundingRate", 0),
                    "next_funding_time": data.get("fundingTimestamp"),
                }
            return None
        except Exception:
            return None

    def fetch_ticker(self, symbol: str) -> dict | None:
        """Fetch 24h ticker data (price, volume, change)."""
        try:
            perp_symbol = self._get_perp_symbol(symbol)
            ticker = self.exchange.fetch_ticker(perp_symbol)
            return {
                "last": ticker.get("last"),
                "volume_24h": ticker.get("baseVolume"),
                "quote_volume_24h": ticker.get("quoteVolume"),
                "change_pct_24h": ticker.get("percentage"),
                "high_24h": ticker.get("high"),
                "low_24h": ticker.get("low"),
            }
        except Exception as e:
            logger.debug(f"Ticker unavailable for {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI, MACD, BBands, EMAs, ATR, volume MA to an OHLCV dataframe."""
        df["rsi_7"] = ta.rsi(df["close"], length=7)
        df["rsi_14"] = ta.rsi(df["close"], length=14)

        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["macd"] = macd.iloc[:, 0]
            df["macd_signal"] = macd.iloc[:, 2]
            df["macd_histogram"] = macd.iloc[:, 1]
        else:
            df["macd"] = df["macd_signal"] = df["macd_histogram"] = 0

        bbands = ta.bbands(df["close"], length=20, std=2)
        if bbands is not None and not bbands.empty:
            df["bb_lower"] = bbands.iloc[:, 0]
            df["bb_middle"] = bbands.iloc[:, 1]
            df["bb_upper"] = bbands.iloc[:, 2]
        else:
            sma_20 = ta.sma(df["close"], length=20)
            df["bb_middle"] = sma_20
            df["bb_upper"] = sma_20 * 1.02 if sma_20 is not None else df["close"] * 1.02
            df["bb_lower"] = sma_20 * 0.98 if sma_20 is not None else df["close"] * 0.98

        df["ema_9"] = ta.ema(df["close"], length=9)
        df["ema_20"] = ta.ema(df["close"], length=20)
        df["ema_50"] = ta.ema(df["close"], length=50)

        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        df["volume_ma"] = ta.sma(df["volume"], length=20)

        df = df.ffill().bfill()
        return df

    # ------------------------------------------------------------------
    # Snapshot: structured data for one coin
    # ------------------------------------------------------------------

    def snapshot(
        self,
        coin: str,
        short_tf: str = "15m",
        long_tf: str = "4h",
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Full snapshot for one coin: OHLCV + indicators + OI + funding + ticker."""
        retry_delay = 2
        symbol = f"{coin}/USDT"

        for attempt in range(1, max_retries + 1):
            try:
                df_short = self.fetch_ohlcv(symbol, short_tf, limit=100)
                df_long = self.fetch_ohlcv(symbol, long_tf, limit=100)

                df_short = self.add_indicators(df_short)
                df_long = self.add_indicators(df_long)

                oi = self.fetch_open_interest(symbol)
                funding = self.fetch_funding_rate(symbol)
                ticker = self.fetch_ticker(symbol)

                latest_short = df_short.iloc[-1]
                latest_long = df_long.iloc[-1]

                return {
                    "coin": coin,
                    "price": float(latest_short["close"]),
                    "ticker": ticker,
                    "open_interest": oi,
                    "funding": funding,
                    "short_tf": {
                        "timeframe": short_tf,
                        "df": df_short,
                        "latest": {
                            "close": float(latest_short["close"]),
                            "rsi_7": float(latest_short["rsi_7"]),
                            "rsi_14": float(latest_short["rsi_14"]),
                            "macd": float(latest_short["macd"]),
                            "macd_signal": float(latest_short["macd_signal"]),
                            "ema_9": float(latest_short["ema_9"]),
                            "ema_20": float(latest_short["ema_20"]),
                            "ema_50": float(latest_short["ema_50"]),
                            "bb_upper": float(latest_short["bb_upper"]),
                            "bb_middle": float(latest_short["bb_middle"]),
                            "bb_lower": float(latest_short["bb_lower"]),
                        },
                    },
                    "long_tf": {
                        "timeframe": long_tf,
                        "df": df_long,
                        "latest": {
                            "close": float(latest_long["close"]),
                            "rsi_14": float(latest_long["rsi_14"]),
                            "macd": float(latest_long["macd"]),
                            "ema_20": float(latest_long["ema_20"]),
                            "ema_50": float(latest_long["ema_50"]),
                            "atr_14": float(latest_long["atr_14"]),
                            "volume": float(latest_long["volume"]),
                            "volume_ma": float(latest_long["volume_ma"]),
                        },
                    },
                }

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"{coin} fetch failed (attempt {attempt}/{max_retries}): {str(e)[:100]}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to fetch {coin} after {max_retries} attempts: {e}")
                    raise

    def multi_snapshot(
        self,
        coins: list[str] | None = None,
        short_tf: str = "15m",
        long_tf: str = "4h",
    ) -> dict[str, Any]:
        """Snapshot for multiple coins. Skips coins that fail."""
        coins = coins or DEFAULT_COINS
        results: dict[str, Any] = {}
        for coin in coins:
            try:
                results[coin] = self.snapshot(coin, short_tf=short_tf, long_tf=long_tf)
            except Exception:
                logger.warning(f"Skipping {coin} — unable to fetch data")
        return results

    # ------------------------------------------------------------------
    # LLM formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_for_llm(snap: dict[str, Any]) -> str:
        """Format a single-coin snapshot as text for LLM prompt injection."""
        coin = snap["coin"]
        price = snap["price"]
        s = snap["short_tf"]["latest"]
        lt = snap["long_tf"]["latest"]

        # Price precision based on magnitude
        if price >= 100:
            pd_ = 2
        elif price >= 1:
            pd_ = 4
        else:
            pd_ = 5

        oi_text = "N/A"
        if snap.get("open_interest") and snap["open_interest"].get("open_interest"):
            oi_text = f"{snap['open_interest']['open_interest']:,.2f}"

        fr_text = "N/A"
        if snap.get("funding") and snap["funding"].get("funding_rate") is not None:
            fr_text = f"{snap['funding']['funding_rate'] * 100:.4f}%"

        vol_text = "N/A"
        if snap.get("ticker") and snap["ticker"].get("volume_24h"):
            vol_text = f"{snap['ticker']['volume_24h']:,.0f}"

        def fmt_arr(series, decimals=2):
            return [round(float(x), decimals) for x in series]

        df_short = snap["short_tf"]["df"]
        df_long = snap["long_tf"]["df"]
        last10 = df_short.tail(10)

        lines = [
            f"=== {coin} PERP ===",
            f"Price: {price:.{pd_}f}  |  EMA20: {s['ema_20']:.{pd_}f}  |  RSI(7): {s['rsi_7']:.1f}  |  MACD: {s['macd']:.2f}",
            f"Open Interest: {oi_text}  |  Funding Rate: {fr_text}  |  24h Volume: {vol_text}",
            "",
            f"Short TF ({snap['short_tf']['timeframe']}) last 10 closes: {fmt_arr(last10['close'], pd_)}",
            f"  EMA20: {fmt_arr(last10['ema_20'], pd_)}",
            f"  RSI(7): {fmt_arr(last10['rsi_7'], 1)}",
            f"  MACD: {fmt_arr(last10['macd'], 3)}",
            "",
            f"Long TF ({snap['long_tf']['timeframe']}): EMA20={lt['ema_20']:.{pd_}f} vs EMA50={lt['ema_50']:.{pd_}f}",
            f"  ATR(14): {lt['atr_14']:.{pd_}f}  |  RSI(14): {lt['rsi_14']:.1f}",
            f"  Volume: {lt['volume']:.0f} vs MA: {lt['volume_ma']:.0f}",
            f"  MACD(10): {fmt_arr(df_long.tail(10)['macd'], 3)}",
            f"  RSI14(10): {fmt_arr(df_long.tail(10)['rsi_14'], 1)}",
        ]
        return "\n".join(lines)

    @staticmethod
    def format_multi_for_llm(snapshots: dict[str, Any]) -> str:
        """Format multiple coin snapshots for LLM."""
        parts = []
        for coin, snap in snapshots.items():
            parts.append(CryptoPerpsData.format_for_llm(snap))
        return "\n\n".join(parts)
