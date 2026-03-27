"""
Positions engine for the LOX FUND Dashboard.

Core position data assembly: Alpaca positions, batch quote fetching,
P&L calculation, days-open tracking, thesis generation, TWR computation.
"""

import os
import re
import csv
from datetime import datetime, timezone

from lox.config import load_settings
from lox.data.alpaca import make_clients
from lox.nav.store import default_nav_flows_path, default_nav_sheet_path, read_nav_sheet
from lox.nav.investors import default_investor_flows_path, read_investor_flows
from lox.utils.occ import parse_occ_option_symbol

from dashboard.cache import (
    POSITIONS_CACHE, POSITIONS_CACHE_LOCK, POSITIONS_CACHE_TTL,
    REGIME_CTX_CACHE, REGIME_CTX_CACHE_LOCK, REGIME_CTX_CACHE_TTL,
    BENCHMARK_CACHE, BENCHMARK_CACHE_LOCK, BENCHMARK_CACHE_TTL,
)
from dashboard.quotes import fetch_batch_option_quotes, fetch_batch_stock_quotes
from dashboard.trade_config import get_indicators_for_position
from dashboard.data_fetchers import (
    get_hy_oas, get_vix, get_10y_yield,
    get_sp500_return_since_inception, get_btc_return_since_inception,
    get_macro_hf_return_since_inception,
)
from dashboard.regime_utils import get_regime_label


# ── Option symbol parsing ──

def parse_option_symbol(symbol):
    """Parse option symbol and return option info dict or None."""
    try:
        sym_upper = symbol.upper()
        if '/' in sym_upper:
            parts = sym_upper.split('/')
            if len(parts) >= 2:
                underlying = parts[0]
                option_part = parts[1]
                m = re.match(r"^(\d{6})([CP])(\d{8})$", option_part)
                if m:
                    exp_str, opt_type_char, strike_str = m.groups()
                    year = int(exp_str[:2]) + 2000
                    month = int(exp_str[2:4])
                    day = int(exp_str[4:6])
                    from datetime import date
                    exp = date(year, month, day)
                    opt_type = 'C' if opt_type_char == 'C' else 'P'
                    strike = float(strike_str) / 1000.0
                    return {
                        "underlying": underlying,
                        "expiry": str(exp),
                        "strike": strike,
                        "opt_type": opt_type,
                    }
        else:
            m = re.match(r"^([A-Z]+)(\d{6}[CP]\d{8})$", sym_upper)
            if m:
                underlying = m.group(1)
                exp, opt_type, strike = parse_occ_option_symbol(sym_upper, underlying)
                opt_type_char = 'C' if opt_type.lower() == 'call' else 'P'
                return {
                    "underlying": underlying,
                    "expiry": str(exp),
                    "strike": strike,
                    "opt_type": opt_type_char,
                }
    except Exception:
        pass
    return None


# ── Position thesis generation ──

def generate_position_theory(position, regime_context, settings):
    """Generate macro-aware theory for a position using LLM (max 50 chars)."""
    try:
        if not settings or not hasattr(settings, 'openai_api_key') or not settings.openai_api_key:
            return simple_position_theory(position)

        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)

        symbol = position.get("symbol", "")
        qty = position.get("qty", 0)
        opt_info = position.get("opt_info")

        if opt_info:
            underlying = opt_info.get("underlying", symbol.split('/')[0] if '/' in symbol else symbol[:3])
            opt_type = opt_info.get("opt_type", opt_info.get("type", "")).upper()
            strike = opt_info.get("strike", "")
            expiry = opt_info.get("expiry", "")
            is_long = qty > 0
            is_call = opt_type in ['C', 'CALL']

            if is_long and is_call:
                direction = f"{underlying} MUST RISE"
                vol_need = "IV expansion helps"
            elif is_long and not is_call:
                direction = f"{underlying} MUST FALL"
                vol_need = "IV expansion helps"
            elif not is_long and is_call:
                direction = f"{underlying} MUST STAY DOWN or FALL"
                vol_need = "IV crush helps"
            else:
                direction = f"{underlying} MUST STAY UP or RISE"
                vol_need = "IV crush helps"

            pos_desc = f"{'Long' if is_long else 'Short'} {opt_type} on {underlying}"
        else:
            underlying = symbol
            is_long = qty > 0
            direction = f"{underlying} MUST {'RISE' if is_long else 'FALL'}"
            vol_need = "N/A"
            pos_desc = f"{'Long' if is_long else 'Short'} {underlying}"

        vix = regime_context.get("vix", "N/A")
        hy_spread = regime_context.get("hy_oas_bps", "N/A")
        yield_10y = regime_context.get("yield_10y", "N/A")
        regime = regime_context.get("regime", "UNKNOWN")

        prompt = f"""Position: {pos_desc}
REQUIREMENT: {direction} ({vol_need})
Current regime: {regime}
Macro: VIX {vix}, HY spreads {hy_spread}bp, 10Y {yield_10y}%

In 1 sentence (max 50 chars), what macro conditions would cause {direction.lower()}?
Be specific about regime, volatility, rates, credit, or sector catalysts.

Examples:
- Long PUT on FXI: "China stress → FXI breaks, vol expands"
- Long CALL on SOFI: "Growth rally → rates fall, tech outperforms"
- Long PUT on TAN: "Solar selloff → rates rise, sector rotation"

Theory:"""

        response = client.chat.completions.create(
            model=settings.openai_model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80,
        )

        theory = response.choices[0].message.content.strip()
        if len(theory) > 50:
            theory = theory[:47] + "..."
        return theory

    except Exception as e:
        print(f"[Theory] LLM error for {position.get('symbol', '?')}: {e}")
        return simple_position_theory(position)


def simple_position_theory(position):
    """Fallback rule-based thesis (no LLM)."""
    symbol = position.get("symbol", "")
    qty = position.get("qty", 0)
    opt_info = position.get("opt_info")

    if opt_info:
        underlying = opt_info.get("underlying", symbol.split('/')[0] if '/' in symbol else symbol[:3])
        opt_type = opt_info.get("opt_type", opt_info.get("type", "")).upper()
        is_long = qty > 0
        is_call = opt_type in ['C', 'CALL']

        if is_long and is_call:
            return f"↑ {underlying} rises, IV expands"
        elif is_long and not is_call:
            return f"↓ {underlying} falls, IV expands"
        elif not is_long and is_call:
            return f"↓ {underlying} stays down, IV crushes"
        else:
            return f"↑ {underlying} stays up, IV crushes"
    else:
        is_long = qty > 0
        return f"{'↑' if is_long else '↓'} Price {'appreciation' if is_long else 'decline'}"


# ── Account helpers ──

def get_nav_equity(account):
    """Get NAV equity — prefer live Alpaca equity, fallback to NAV sheet."""
    if account:
        try:
            account_equity = float(getattr(account, 'equity', 0.0) or 0.0)
            if account_equity > 0:
                return account_equity
        except Exception:
            pass
    try:
        nav_rows = read_nav_sheet()
        if nav_rows:
            nav = nav_rows[-1]
            nav_equity = float(nav.equity) if hasattr(nav, 'equity') and nav.equity is not None else 0.0
            if nav_equity > 0:
                return nav_equity
    except Exception as nav_error:
        print(f"Warning: Could not read NAV sheet: {nav_error}")
    return 0.0


def get_original_capital():
    """Get original capital from investor flows (preferred) or env var fallback."""
    try:
        flows = read_investor_flows(path=default_investor_flows_path())
        capital_sum = sum(float(f.amount) for f in flows if float(f.amount) > 0)
        if capital_sum > 0:
            return capital_sum
    except Exception as flow_error:
        print(f"Warning: Could not read investor flows: {flow_error}")
    return float(os.environ.get("FUND_TOTAL_CAPITAL", "0")) or 950.0


def get_regime_context(settings):
    """Get macro regime context for thesis generation. Cached for 5 minutes."""
    with REGIME_CTX_CACHE_LOCK:
        if REGIME_CTX_CACHE["data"] and REGIME_CTX_CACHE["timestamp"]:
            cache_age = (datetime.now(timezone.utc) - REGIME_CTX_CACHE["timestamp"]).total_seconds()
            if cache_age < REGIME_CTX_CACHE_TTL:
                return REGIME_CTX_CACHE["data"]

    try:
        hy_oas = get_hy_oas(settings)
        vix = get_vix(settings)
        yield_10y = get_10y_yield(settings)

        vix_val = vix.get("value") if vix else None
        hy_val = hy_oas.get("value") if hy_oas else None
        regime_label = get_regime_label(vix_val, hy_val)

        dxy_val = None
        try:
            import requests as _req
            url = "https://financialmodelingprep.com/api/v3/quote/DX-Y.NYB"
            resp = _req.get(url, params={"apikey": settings.FMP_API_KEY}, timeout=10)
            resp.raise_for_status()
            dxy_data = resp.json()
            if isinstance(dxy_data, list) and dxy_data and dxy_data[0].get("price"):
                dxy_val = float(dxy_data[0]["price"])
        except Exception as e:
            print(f"[RegimeCtx] DXY fetch error: {e}")

        result = {
            "vix": f"{vix_val:.1f}" if vix_val else "N/A",
            "hy_oas_bps": f"{hy_val:.0f}" if hy_val else "N/A",
            "yield_10y": f"{yield_10y.get('value'):.2f}%" if yield_10y and yield_10y.get('value') else "N/A",
            "dxy": f"{dxy_val:.1f}" if dxy_val else "N/A",
            "regime": regime_label,
        }

        with REGIME_CTX_CACHE_LOCK:
            REGIME_CTX_CACHE["data"] = result
            REGIME_CTX_CACHE["timestamp"] = datetime.now(timezone.utc)

        return result
    except Exception as e:
        print(f"[Positions] Could not fetch regime context: {e}")
        return {"vix": "N/A", "hy_oas_bps": "N/A", "yield_10y": "N/A", "dxy": "N/A", "regime": "UNKNOWN"}


def get_live_twr(live_equity: float) -> float | None:
    """
    Calculate LIVE Time-Weighted Return by chaining:
    - Historical TWR from nav_sheet.csv (up to last snapshot)
    - Live return since last snapshot (from Alpaca)
    """
    try:
        nav_sheet_path = default_nav_sheet_path()
        nav_flows_path = default_nav_flows_path()

        if not os.path.exists(nav_sheet_path):
            return None

        with open(nav_sheet_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return None

        latest = rows[-1]
        twr_cum_str = latest.get("twr_cum", "")
        last_equity_str = latest.get("equity", "")
        last_snapshot_ts = latest.get("ts", "")

        if not twr_cum_str or not last_equity_str:
            return None

        twr_cum = float(twr_cum_str)
        last_equity = float(last_equity_str)

        if last_equity <= 0 or live_equity <= 0:
            return None

        flows_since_snapshot = 0.0
        if os.path.exists(nav_flows_path) and last_snapshot_ts:
            try:
                snapshot_dt = datetime.fromisoformat(last_snapshot_ts.replace("Z", "+00:00"))
                with open(nav_flows_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        flow_ts = row.get("ts", "")
                        flow_amount = float(row.get("amount", 0) or 0)
                        if flow_ts and flow_amount != 0:
                            try:
                                flow_dt = datetime.fromisoformat(flow_ts.replace("Z", "+00:00"))
                                if flow_dt > snapshot_dt:
                                    flows_since_snapshot += flow_amount
                            except Exception:
                                pass
            except Exception as e:
                print(f"[TWR] Error reading flows: {e}")

        adjusted_equity = live_equity - flows_since_snapshot
        return_since_snapshot = (adjusted_equity - last_equity) / last_equity
        live_twr = (1 + twr_cum) * (1 + return_since_snapshot) - 1
        return live_twr * 100

    except Exception as e:
        print(f"[TWR] Live TWR calculation error: {e}")
        return None


# ── Main positions data function ──

def get_positions_data(force_refresh: bool = False):
    """
    Fetch positions and calculate P&L using conservative bid/ask marking.
    Uses short-lived cache (30s) to reduce API calls on rapid refreshes.
    """
    if not force_refresh:
        with POSITIONS_CACHE_LOCK:
            if POSITIONS_CACHE["data"] and POSITIONS_CACHE["timestamp"]:
                cache_age = (datetime.now(timezone.utc) - POSITIONS_CACHE["timestamp"]).total_seconds()
                if cache_age < POSITIONS_CACHE_TTL:
                    print(f"[Positions] Serving from cache (age: {cache_age:.1f}s)")
                    return POSITIONS_CACHE["data"]

    try:
        settings = load_settings()
        trading, data_client = make_clients(settings)
        account = trading.get_account()
        positions = trading.get_all_positions()

        nav_equity = get_nav_equity(account)
        original_capital = get_original_capital()
        total_pnl = nav_equity - original_capital
        regime_context = get_regime_context(settings)

        # Fetch FILL activities to compute "days open" per position.
        # Builds fill_map: symbol -> [(timestamp, qty, side), ...]
        from dateutil.parser import parse as _parse_dt
        fill_map = {}
        try:
            raw_fills = trading.get_activities(activity_types="FILL") or []
            if isinstance(raw_fills, dict):
                raw_fills = [raw_fills]
            print(f"[Days Open] Got {len(raw_fills) if isinstance(raw_fills, list) else 'non-list'} raw fills")
            if isinstance(raw_fills, list) and len(raw_fills) > 0:
                sample = raw_fills[0]
                if isinstance(sample, dict):
                    print(f"[Days Open] Sample fill (dict): symbol={sample.get('symbol')}, side={sample.get('side')}, qty={sample.get('qty')}, time={sample.get('transaction_time')}")
                else:
                    print(f"[Days Open] Sample fill (obj type={type(sample).__name__}): symbol={getattr(sample, 'symbol', '?')}, side={getattr(sample, 'side', '?')}, qty={getattr(sample, 'qty', '?')}")
            for f in (raw_fills if isinstance(raw_fills, list) else []):
                if isinstance(f, dict):
                    sym = f.get("symbol", "")
                    ts = f.get("transaction_time")
                    fqty = float(f.get("qty", 0) or 0)
                    side = str(f.get("side", "")).split(".")[-1].lower()
                else:
                    sym = str(getattr(f, "symbol", ""))
                    ts = getattr(f, "transaction_time", None)
                    fqty = float(getattr(f, "qty", 0) or 0)
                    side = str(getattr(f, "side", "")).split(".")[-1].lower()
                if not sym or not ts or fqty <= 0:
                    continue
                if isinstance(ts, str):
                    ts = _parse_dt(ts)
                fill_map.setdefault(sym, []).append((ts, fqty, side))
            for sym in fill_map:
                fill_map[sym].sort(key=lambda x: x[0], reverse=True)
            print(f"[Days Open] fill_map keys: {list(fill_map.keys())[:20]}")
        except Exception as e:
            import traceback
            print(f"[Days Open] Error fetching fills: {e}")
            traceback.print_exc()
            fill_map = {}

        # Pre-parse all positions and batch quote requests
        position_data = []
        option_symbols = []
        stock_symbols = []

        for p in positions:
            symbol = str(getattr(p, "symbol", "") or "")
            if not symbol:
                continue
            opt_info = parse_option_symbol(symbol)
            position_data.append({
                "raw": p,
                "symbol": symbol,
                "opt_info": opt_info,
                "qty": float(getattr(p, "qty", 0.0) or 0.0),
                "avg_entry": float(getattr(p, "avg_entry_price", 0.0) or 0.0),
                "current_price": float(getattr(p, "current_price", 0.0) or 0.0),
            })
            if opt_info:
                option_symbols.append(symbol)
            else:
                stock_symbols.append(symbol)

        # Batch fetch all quotes in 2 API calls
        print(f"[Positions] Batch fetching quotes: {len(option_symbols)} options, {len(stock_symbols)} stocks")
        option_quotes = fetch_batch_option_quotes(data_client, option_symbols)
        stock_quotes = fetch_batch_stock_quotes(data_client, stock_symbols, settings=settings)
        all_quotes = {**option_quotes, **stock_quotes}

        positions_list = []
        total_liquidation_value = 0.0

        for pd in position_data:
            try:
                symbol = pd["symbol"]
                qty = pd["qty"]
                avg_entry = pd["avg_entry"]
                current_price = pd["current_price"]
                opt_info = pd["opt_info"]

                quote = all_quotes.get(symbol)
                liquidation_price = current_price
                multiplier = 100 if opt_info else 1

                # Conservative mark: longs at bid, shorts at ask
                if quote:
                    if qty > 0:
                        liquidation_price = quote["bid"] if quote["bid"] > 0 else current_price
                    else:
                        liquidation_price = quote["ask"] if quote["ask"] > 0 else current_price

                entry_cost = avg_entry * multiplier * abs(qty) if avg_entry > 0 else 0
                liquidation_value = liquidation_price * multiplier * abs(qty)
                pnl = liquidation_value - entry_cost if qty > 0 else entry_cost - liquidation_value

                total_liquidation_value += liquidation_value

                # Days open — walk fills chronologically, track running position,
                # reset clock whenever position goes flat (qty == 0).
                # Reports days since the oldest fill of the current continuous hold.
                days_open = None
                try:
                    fills = fill_map.get(symbol)
                    print(f"[Days Open] Position {symbol}: direct lookup={'found' if fills else 'miss'}")
                    if not fills and opt_info:
                        underlying = opt_info.get("underlying", "")
                        fills = fill_map.get(underlying)
                        print(f"[Days Open] Position {symbol}: underlying lookup '{underlying}'={'found' if fills else 'miss'}")
                    if fills:
                        # Sort oldest-first for chronological walk
                        chrono = sorted(fills, key=lambda x: x[0])
                        running = 0.0
                        open_since = None
                        for ts, fqty, side in chrono:
                            if side == "buy":
                                if running == 0.0:
                                    open_since = ts  # new position started
                                running += fqty
                            elif side in ("sell", "sell_short"):
                                if running == 0.0:
                                    open_since = ts  # new short started
                                running -= fqty
                            # Position went flat → reset clock
                            if abs(running) < 1e-9:
                                running = 0.0
                                open_since = None
                        if open_since:
                            if not open_since.tzinfo:
                                open_since = open_since.replace(tzinfo=timezone.utc)
                            days_open = (datetime.now(timezone.utc) - open_since).days
                except Exception:
                    pass

                position_dict = {
                    "symbol": symbol,
                    "qty": qty,
                    "market_value": liquidation_value,
                    "pnl": pnl,
                    "pnl_pct": (pnl / entry_cost * 100) if entry_cost > 0 else 0.0,
                    "current_price": liquidation_price,
                    "opt_info": opt_info,
                    "bid_ask_mark": True,
                    "days_open": days_open,
                }

                thesis = simple_position_theory(position_dict)
                position_dict["thesis"] = thesis
                position_dict["indicators"] = get_indicators_for_position(position_dict)

                positions_list.append(position_dict)
            except Exception as pos_err:
                print(f"Warning: Error processing position: {pos_err}")
                continue

        positions_list.sort(key=lambda x: x["pnl"], reverse=True)

        cash_available = 0.0
        try:
            if account:
                cash_available = float(getattr(account, 'cash', 0.0) or 0.0)
        except Exception:
            pass

        liquidation_nav = cash_available + total_liquidation_value
        liquidation_pnl = liquidation_nav - original_capital

        # Live TWR (GIPS compliant)
        alpaca_equity = float(getattr(account, 'equity', 0.0) or 0.0) if account else 0.0
        display_nav = alpaca_equity if alpaca_equity > 0 else liquidation_nav
        display_pnl = display_nav - original_capital
        simple_return_pct = (display_pnl / original_capital * 100) if original_capital > 0 else 0.0
        live_twr_pct = get_live_twr(display_nav)
        return_pct = live_twr_pct if live_twr_pct is not None else simple_return_pct

        # Benchmark comparison (cached 5 min)
        sp500_return = None
        btc_return = None
        macro_hf_return = None
        with BENCHMARK_CACHE_LOCK:
            if BENCHMARK_CACHE["data"] and BENCHMARK_CACHE["timestamp"]:
                _bcache_age = (datetime.now(timezone.utc) - BENCHMARK_CACHE["timestamp"]).total_seconds()
                if _bcache_age < BENCHMARK_CACHE_TTL:
                    sp500_return = BENCHMARK_CACHE["data"].get("sp500")
                    btc_return = BENCHMARK_CACHE["data"].get("btc")
                    macro_hf_return = BENCHMARK_CACHE["data"].get("macro_hf")
        if sp500_return is None and btc_return is None:
            sp500_return = get_sp500_return_since_inception(settings)
            btc_return = get_btc_return_since_inception(settings)
            macro_hf_return = get_macro_hf_return_since_inception(settings)
            with BENCHMARK_CACHE_LOCK:
                BENCHMARK_CACHE["data"] = {"sp500": sp500_return, "btc": btc_return, "macro_hf": macro_hf_return}
                BENCHMARK_CACHE["timestamp"] = datetime.now(timezone.utc)
        alpha_sp500 = return_pct - sp500_return if sp500_return is not None else None
        alpha_btc = return_pct - btc_return if btc_return is not None else None
        alpha_macro_hf = return_pct - macro_hf_return if macro_hf_return is not None else None

        # AUM and investor count
        aum = original_capital
        investor_count = 0
        try:
            flows = read_investor_flows(path=default_investor_flows_path())
            investor_codes = set(f.code for f in flows if float(f.amount) > 0)
            investor_count = len(investor_codes)
        except Exception:
            pass

        result = {
            "positions": positions_list,
            "total_pnl": display_pnl,
            "total_value": total_liquidation_value,
            "nav_equity": display_nav,
            "liquidation_nav": liquidation_nav,
            "original_capital": original_capital,
            "aum": aum,
            "investor_count": investor_count,
            "return_pct": return_pct,
            "sp500_return": sp500_return,
            "btc_return": btc_return,
            "macro_hf_return": macro_hf_return,
            "alpha_sp500": alpha_sp500,
            "alpha_btc": alpha_btc,
            "alpha_macro_hf": alpha_macro_hf,
            "cash_available": cash_available,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mark_type": "equity",
            "cached": False,
        }

        with POSITIONS_CACHE_LOCK:
            POSITIONS_CACHE["data"] = {**result, "cached": True}
            POSITIONS_CACHE["timestamp"] = datetime.now(timezone.utc)

        return result
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error in get_positions_data: {error_msg}")
        traceback.print_exc()

        with POSITIONS_CACHE_LOCK:
            if POSITIONS_CACHE["data"]:
                stale = {**POSITIONS_CACHE["data"], "cached": True, "stale": True}
                print("[Positions] Serving stale cache as fallback after error")
                return stale

        nav_equity = 0.0
        original_capital = 950.0
        try:
            settings = load_settings()
            trading, _ = make_clients(settings)
            account = trading.get_account()
            if account:
                nav_equity = float(getattr(account, 'equity', 0.0) or 0.0)
        except Exception:
            pass

        try:
            flows = read_investor_flows(path=default_investor_flows_path())
            capital_sum = sum(float(f.amount) for f in flows if float(f.amount) > 0)
            if capital_sum > 0:
                original_capital = capital_sum
        except Exception:
            pass

        return {
            "error": error_msg,
            "positions": [],
            "total_pnl": nav_equity - original_capital,
            "total_value": 0.0,
            "nav_equity": nav_equity,
            "original_capital": original_capital,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
