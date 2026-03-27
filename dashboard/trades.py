"""Closed trade FIFO matching, performance metrics, and entry date tracking."""

from datetime import datetime, timezone, timedelta
import os
import re
import time

from lox.config import load_settings
from lox.data.alpaca import make_clients
from lox.utils.occ import parse_occ_option_symbol
from dashboard.cache import TRADES_CACHE, TRADES_CACHE_LOCK, TRADES_CACHE_TTL


def parse_option_symbol_display(sym: str) -> str:
    """Parse option symbol for display."""
    if '/' in sym:  # Crypto
        return sym
    if len(sym) <= 6:  # Stock/ETF
        return sym

    try:
        i = 0
        while i < len(sym) and not sym[i].isdigit():
            i += 1

        if i == 0 or i >= len(sym):
            return sym

        ticker = sym[:i]
        rest = sym[i:]

        if len(rest) >= 15:
            exp = f"{rest[2:4]}/{rest[4:6]}"
            opt_type = "C" if rest[6] == 'C' else "P"
            strike = int(rest[7:]) / 1000
            return f"{ticker} ${strike:.0f}{opt_type} {exp}"
    except (ValueError, IndexError):
        pass

    return sym


def get_closed_trades_data():
    """Fetch closed trades and calculate realized P&L using FIFO matching."""
    from collections import defaultdict

    try:
        settings = load_settings()
        trading, _ = make_clients(settings)
    except Exception as e:
        return {"error": str(e), "trades": [], "total_pnl": 0, "win_rate": 0}

    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500)
        orders = trading.get_orders(req) or []
    except Exception as e:
        return {"error": str(e), "trades": [], "total_pnl": 0, "win_rate": 0}

    # Group filled orders by symbol
    trades_by_symbol = defaultdict(lambda: {'buys': [], 'sells': []})

    for o in orders:
        status = str(getattr(o, 'status', '?')).split('.')[-1].lower()
        if 'filled' not in status:
            continue

        sym = getattr(o, 'symbol', '?')
        side = str(getattr(o, 'side', '?')).split('.')[-1].lower()
        filled_qty = float(getattr(o, 'filled_qty', 0) or 0)
        filled_price = getattr(o, 'filled_avg_price', 0)
        filled_at = getattr(o, 'filled_at', None)

        try:
            price = float(filled_price) if filled_price else 0
        except (ValueError, TypeError):
            price = 0

        if filled_qty <= 0 or price <= 0:
            continue

        # Determine if option (multiplier = 100)
        is_option = len(sym) > 10 and any(c.isdigit() for c in sym[-8:]) and '/' not in sym
        mult = 100 if is_option else 1

        trade = {
            'qty': filled_qty,
            'price': price,
            'date': filled_at,  # Keep full datetime for sorting
            'date_str': str(filled_at)[:10] if filled_at else '?',
            'value': filled_qty * price * mult,
            'mult': mult
        }

        if side == 'buy':
            trades_by_symbol[sym]['buys'].append(trade)
        else:
            trades_by_symbol[sym]['sells'].append(trade)

    # Inject synthetic sell-at-$0 for expired options
    from datetime import datetime as _dt, date as _date, timezone as _tz

    # 1) From Alpaca OPEXP activities (OTM option expiry events)
    try:
        _opexp_raw = trading.get("/account/activities/OPEXP") or []
        if isinstance(_opexp_raw, dict):
            _opexp_raw = [_opexp_raw]
        _opexp_list = _opexp_raw if isinstance(_opexp_raw, list) else []
    except Exception:
        _opexp_list = []

    for act in _opexp_list:
        if isinstance(act, dict):
            sym = act.get("symbol", "")
            qty_str = act.get("qty", "0")
            exp_date = act.get("date", "")
        else:
            sym = str(getattr(act, "symbol", ""))
            qty_str = str(getattr(act, "qty", "0"))
            exp_date = str(getattr(act, "date", ""))

        qty = abs(float(qty_str)) if qty_str else 0
        if not sym or qty <= 0:
            continue

        try:
            exp_dt = _dt.strptime(exp_date, "%Y-%m-%d").replace(tzinfo=_tz.utc) if exp_date else None
        except Exception:
            exp_dt = None

        is_option = len(sym) > 10 and any(c.isdigit() for c in sym[-8:]) and '/' not in sym
        mult = 100 if is_option else 1

        trades_by_symbol[sym]['sells'].append({
            'qty': qty, 'price': 0.0, 'date': exp_dt,
            'date_str': exp_date[:10] if exp_date else '?',
            'value': 0.0, 'mult': mult,
        })

    # 2) Fallback: detect expired options from OCC symbol date with unmatched buys
    _today = _date.today()
    for sym, data in trades_by_symbol.items():
        if not data['buys'] or data['sells']:
            continue
        is_option = len(sym) > 10 and any(c.isdigit() for c in sym[-8:]) and '/' not in sym
        if not is_option:
            continue
        try:
            i = 0
            while i < len(sym) and not sym[i].isdigit():
                i += 1
            exp = _dt.strptime(f"20{sym[i:][:6]}", "%Y%m%d").date()
        except Exception:
            continue
        if exp >= _today:
            continue
        qty = sum(b['qty'] for b in data['buys'])
        data['sells'].append({
            'qty': qty, 'price': 0.0,
            'date': _dt.combine(exp, _dt.min.time()).replace(tzinfo=_tz.utc),
            'date_str': str(exp), 'value': 0.0, 'mult': 100,
        })

    # Calculate closed trades using FIFO matching
    closed_trades = []

    for sym, data in trades_by_symbol.items():
        # Sort by date (oldest first) for FIFO
        buys = sorted(data['buys'], key=lambda x: x['date'] or '0')
        sells = sorted(data['sells'], key=lambda x: x['date'] or '0')

        display_sym = parse_option_symbol_display(sym)

        # FIFO matching: match each sell with the oldest available buy that came BEFORE it
        total_cost = 0.0
        total_proceeds = 0.0
        closed_qty = 0.0

        buy_queue = []  # Queue of (remaining_qty, price_per_unit, mult, date)
        for b in buys:
            buy_queue.append([b['qty'], b['price'], b['mult'], b['date']])

        for sell in sells:
            sell_qty_remaining = sell['qty']
            sell_date = sell['date']

            while sell_qty_remaining > 0 and buy_queue:
                # Find first buy that came BEFORE this sell
                buy_idx = None
                for i, (bq, bp, bm, bd) in enumerate(buy_queue):
                    if bq > 0 and (bd is None or sell_date is None or bd <= sell_date):
                        buy_idx = i
                        break

                if buy_idx is None:
                    # No matching buy found (sell came before any remaining buys)
                    break

                buy_remaining, buy_price, buy_mult, buy_date = buy_queue[buy_idx]
                match_qty = min(sell_qty_remaining, buy_remaining)

                # Calculate P&L for this match
                cost = match_qty * buy_price * buy_mult
                proceeds = match_qty * sell['price'] * sell['mult']

                total_cost += cost
                total_proceeds += proceeds
                closed_qty += match_qty

                # Update remaining quantities
                buy_queue[buy_idx][0] -= match_qty
                sell_qty_remaining -= match_qty

                # Remove exhausted buys
                if buy_queue[buy_idx][0] <= 0:
                    buy_queue.pop(buy_idx)

        # Only include if we matched something
        if closed_qty > 0:
            realized_pnl = total_proceeds - total_cost
            pnl_pct = (realized_pnl / total_cost * 100) if total_cost > 0 else 0
            # Check if any buys remain unmatched
            remaining_buy_qty = sum(b[0] for b in buy_queue)
            fully_closed = remaining_buy_qty < 0.001  # Floating point tolerance

            # Track entry/exit dates for holding period calculation
            entry_date = buys[0]['date'] if buys else None
            exit_date = sells[-1]['date'] if sells else None

            closed_trades.append({
                'symbol': display_sym,
                'cost': total_cost,
                'proceeds': total_proceeds,
                'pnl': realized_pnl,
                'pnl_pct': pnl_pct,
                'fully_closed': fully_closed,
                'entry_date': entry_date,
                'exit_date': exit_date,
            })

    # Sort by exit date for equity curve construction
    trades_with_dates = [t for t in closed_trades if t.get('exit_date')]
    trades_with_dates.sort(key=lambda x: x['exit_date'])

    # Also sort main list by P&L for display
    closed_trades.sort(key=lambda x: x['pnl'], reverse=True)

    # Calculate totals
    total_realized = sum(t['pnl'] for t in closed_trades)
    total_wins = sum(1 for t in closed_trades if t['pnl'] >= 0)
    total_losses = sum(1 for t in closed_trades if t['pnl'] < 0)
    total_trades = total_wins + total_losses
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    # =========================================================================
    # INSTITUTIONAL-GRADE PERFORMANCE METRICS
    # =========================================================================
    import statistics
    import math

    # Separate wins and losses
    wins_pnl = [t['pnl'] for t in closed_trades if t['pnl'] >= 0]
    losses_pnl = [t['pnl'] for t in closed_trades if t['pnl'] < 0]
    all_pnl = [t['pnl'] for t in closed_trades]
    all_pnl_pct = [t['pnl_pct'] for t in closed_trades]

    # Gross Profit / Gross Loss
    gross_profit = sum(wins_pnl) if wins_pnl else 0
    gross_loss = abs(sum(losses_pnl)) if losses_pnl else 0

    # Profit Factor = Gross Profit / Gross Loss (>1 is good, >2 is excellent)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

    # Average Win / Average Loss (Payoff Ratio)
    avg_win = sum(wins_pnl) / len(wins_pnl) if wins_pnl else 0
    avg_loss = abs(sum(losses_pnl) / len(losses_pnl)) if losses_pnl else 0
    avg_win_pct = sum(t['pnl_pct'] for t in closed_trades if t['pnl'] >= 0) / len(wins_pnl) if wins_pnl else 0
    avg_loss_pct = abs(sum(t['pnl_pct'] for t in closed_trades if t['pnl'] < 0) / len(losses_pnl)) if losses_pnl else 0

    # Payoff Ratio (Avg Win / Avg Loss) - institutional standard metric
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0

    # Expectancy (Edge) = (Win Rate x Avg Win) - (Loss Rate x Avg Loss)
    loss_rate = (100 - win_rate) / 100
    expectancy = ((win_rate / 100) * avg_win) - (loss_rate * avg_loss)
    expectancy_pct = ((win_rate / 100) * avg_win_pct) - (loss_rate * avg_loss_pct)

    # Average P&L per trade
    avg_pnl = total_realized / total_trades if total_trades > 0 else 0
    avg_pnl_pct = sum(all_pnl_pct) / len(all_pnl_pct) if all_pnl_pct else 0

    # Largest Win / Largest Loss
    largest_win = max(wins_pnl) if wins_pnl else 0
    largest_loss = min(losses_pnl) if losses_pnl else 0
    largest_win_pct = max((t['pnl_pct'] for t in closed_trades if t['pnl'] >= 0), default=0)
    largest_loss_pct = min((t['pnl_pct'] for t in closed_trades if t['pnl'] < 0), default=0)

    # =========================================================================
    # DISTRIBUTION STATISTICS (Std Dev, Skew)
    # =========================================================================
    pnl_std = statistics.stdev(all_pnl) if len(all_pnl) > 1 else 0
    pnl_pct_std = statistics.stdev(all_pnl_pct) if len(all_pnl_pct) > 1 else 0

    # Skewness of returns (positive = right tail, negative = left tail)
    # Skew = E[(X - mu)^3] / sigma^3
    skewness = 0
    if len(all_pnl_pct) > 2 and pnl_pct_std > 0:
        mean_pct = avg_pnl_pct
        skewness = sum((x - mean_pct) ** 3 for x in all_pnl_pct) / (len(all_pnl_pct) * (pnl_pct_std ** 3))

    # =========================================================================
    # EQUITY CURVE & DRAWDOWN ANALYSIS
    # =========================================================================
    # Build equity curve from trade sequence (assumes starting capital context)
    equity_curve = []
    running_pnl = 0
    peak = 0
    max_drawdown = 0
    max_drawdown_pct = 0
    drawdown_start = None
    drawdown_end = None
    recovery_date = None
    max_dd_start = None
    max_dd_end = None

    for t in trades_with_dates:
        running_pnl += t['pnl']
        exit_dt = t['exit_date']
        equity_curve.append({'date': exit_dt, 'equity': running_pnl})

        if running_pnl > peak:
            peak = running_pnl
            if drawdown_start and not recovery_date:
                recovery_date = exit_dt
        else:
            dd = peak - running_pnl
            if dd > max_drawdown:
                max_drawdown = dd
                max_dd_start = drawdown_start
                max_dd_end = exit_dt
            if drawdown_start is None:
                drawdown_start = exit_dt

    # Calculate drawdown as % of peak (if we had profits)
    if peak > 0:
        max_drawdown_pct = (max_drawdown / peak) * 100

    # Time to recover (in days)
    recovery_days = None
    if max_dd_start and recovery_date:
        try:
            recovery_days = (recovery_date - max_dd_start).days
        except:
            pass

    # =========================================================================
    # HOLDING PERIOD ANALYSIS
    # =========================================================================
    holding_periods = []
    for t in closed_trades:
        entry = t.get('entry_date')
        exit_dt = t.get('exit_date')
        if entry and exit_dt:
            try:
                days = (exit_dt - entry).days
                if days >= 0:
                    holding_periods.append(days)
            except:
                pass

    avg_holding_days = sum(holding_periods) / len(holding_periods) if holding_periods else None

    # =========================================================================
    # PORTFOLIO-LEVEL SHARPE (Annualized)
    # =========================================================================
    # Calculate from equity curve returns, not individual trades
    # This is the institutionally correct way to measure risk-adjusted return
    portfolio_sharpe = None
    portfolio_sharpe_note = "Insufficient data"

    if len(equity_curve) >= 5:
        # Calculate returns between trades
        returns = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i-1]['equity']
            curr = equity_curve[i]['equity']
            if prev != 0:
                ret = (curr - prev) / abs(prev) if prev != 0 else 0
                returns.append(ret)
            else:
                returns.append(curr / 100 if curr else 0)  # Assume $100 base if starting from 0

        if returns and len(returns) > 1:
            mean_ret = statistics.mean(returns)
            std_ret = statistics.stdev(returns)
            if std_ret > 0:
                # Annualize based on average trade frequency
                if trades_with_dates and len(trades_with_dates) >= 2:
                    first_date = trades_with_dates[0]['exit_date']
                    last_date = trades_with_dates[-1]['exit_date']
                    try:
                        total_days = (last_date - first_date).days
                        if total_days > 0:
                            trades_per_year = (len(trades_with_dates) / total_days) * 252
                            annualization_factor = math.sqrt(trades_per_year)
                            portfolio_sharpe = (mean_ret / std_ret) * annualization_factor
                            portfolio_sharpe_note = f"Annualized ({trades_per_year:.0f} trades/yr equiv)"
                    except:
                        pass

    # =========================================================================
    # DATE RANGE & SAMPLE SIZE (Small-Sample Disclosure)
    # =========================================================================
    first_trade_date = None
    last_trade_date = None
    if trades_with_dates:
        first_trade_date = trades_with_dates[0]['exit_date']
        last_trade_date = trades_with_dates[-1]['exit_date']

    date_range_str = None
    trading_days = None
    if first_trade_date and last_trade_date:
        try:
            date_range_str = f"{first_trade_date.strftime('%b %d')} - {last_trade_date.strftime('%b %d, %Y')}"
            trading_days = (last_trade_date - first_trade_date).days
        except:
            pass

    # Small sample warning
    sample_warning = None
    if total_trades < 20:
        sample_warning = f"Small sample (n={total_trades}). Interpret with caution."
    elif total_trades < 50:
        sample_warning = f"Moderate sample (n={total_trades}). Results may not be statistically robust."

    # =========================================================================
    # LEGACY METRICS (kept for compatibility)
    # =========================================================================
    # Trade-level Sharpe (per-trade consistency, NOT portfolio Sharpe)
    trade_sharpe = avg_pnl / pnl_std if pnl_std > 0 else 0

    # R-Multiple = Total P&L / Average Risk
    r_multiple = total_realized / avg_loss if avg_loss > 0 else 0

    # Kelly Criterion
    if payoff_ratio > 0 and payoff_ratio != float('inf'):
        kelly = ((win_rate / 100) * payoff_ratio - loss_rate) / payoff_ratio
        kelly = max(0, min(kelly, 1))
    else:
        kelly = 0

    # Max Consecutive Wins/Losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0
    for t in trades_with_dates:
        if t['pnl'] >= 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    # =========================================================================
    # OVERALL GRADE (Updated criteria)
    # =========================================================================
    grade_score = 0
    if win_rate >= 55: grade_score += 1
    if win_rate >= 65: grade_score += 1
    if profit_factor >= 1.5: grade_score += 1
    if profit_factor >= 2.0: grade_score += 1
    if expectancy > 0: grade_score += 1
    if payoff_ratio >= 1.0: grade_score += 1
    if payoff_ratio >= 1.5: grade_score += 1
    if max_drawdown_pct < 20 or max_drawdown_pct == 0: grade_score += 1
    if portfolio_sharpe and portfolio_sharpe > 1.0: grade_score += 1
    if portfolio_sharpe and portfolio_sharpe > 2.0: grade_score += 1

    if grade_score >= 8:
        overall_grade = "A"
    elif grade_score >= 6:
        overall_grade = "B"
    elif grade_score >= 4:
        overall_grade = "C"
    elif grade_score >= 2:
        overall_grade = "D"
    else:
        overall_grade = "F"

    return {
        "trades": closed_trades,
        "total_pnl": total_realized,
        "wins": total_wins,
        "losses": total_losses,
        "win_rate": win_rate,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Institutional metrics
        "metrics": {
            # Core performance
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else 999,
            "expectancy": round(expectancy, 2),
            "expectancy_pct": round(expectancy_pct, 2),

            # Payoff analysis
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_win_pct": round(avg_win_pct, 1),
            "avg_loss_pct": round(avg_loss_pct, 1),
            "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio != float('inf') else 999,

            # Distribution
            "pnl_std": round(pnl_std, 2),
            "pnl_pct_std": round(pnl_pct_std, 1),
            "skewness": round(skewness, 2),

            # Extremes
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "largest_win_pct": round(largest_win_pct, 1),
            "largest_loss_pct": round(largest_loss_pct, 1),

            # Drawdown
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 1),
            "recovery_days": recovery_days,

            # Holding period
            "avg_holding_days": round(avg_holding_days, 1) if avg_holding_days else None,

            # Sharpe ratios
            "trade_sharpe": round(trade_sharpe, 2),  # Per-trade consistency
            "portfolio_sharpe": round(portfolio_sharpe, 2) if portfolio_sharpe else None,  # Annualized
            "portfolio_sharpe_note": portfolio_sharpe_note,

            # Risk metrics
            "r_multiple": round(r_multiple, 2),
            "kelly_pct": round(kelly * 100, 1),

            # Streaks
            "max_consec_wins": max_consec_wins,
            "max_consec_losses": max_consec_losses,

            # Totals
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),

            # Sample disclosure
            "date_range": date_range_str,
            "trading_days": trading_days,
            "sample_warning": sample_warning,

            # Grade
            "overall_grade": overall_grade,
        },
    }


def get_open_position_entry_dates() -> dict:
    """
    Fetch filled buy orders and return earliest buy date per symbol
    for currently held positions. Used to place open positions on the timeline.
    Returns: { raw_symbol: datetime, ... }
    """
    try:
        settings = load_settings()
        trading, _ = make_clients(settings)
    except Exception:
        return {}

    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500)
        orders = trading.get_orders(req) or []
    except Exception:
        return {}

    # Get current positions to know which symbols are open
    try:
        positions = trading.get_all_positions()
        open_symbols = {str(getattr(p, "symbol", "")) for p in positions}
    except Exception:
        return {}

    # Find earliest buy date per open symbol
    entry_dates = {}
    for o in orders:
        status = str(getattr(o, 'status', '?')).split('.')[-1].lower()
        if 'filled' not in status:
            continue
        side = str(getattr(o, 'side', '?')).split('.')[-1].lower()
        if side != 'buy':
            continue
        sym = str(getattr(o, 'symbol', ''))
        if sym not in open_symbols:
            continue
        filled_at = getattr(o, 'filled_at', None)
        if filled_at is None:
            continue
        # Keep earliest buy
        if sym not in entry_dates or filled_at < entry_dates[sym]:
            entry_dates[sym] = filled_at

    return entry_dates
