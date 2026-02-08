"""
LOX FUND - Regime History
Backfill and query helpers for historical regime snapshots.
Used to correlate trading performance with macro conditions.
"""

import requests
import pandas as pd
from datetime import datetime, timezone, date as date_type, timedelta

from dashboard.models import db, RegimeSnapshot

# Fund inception date (first trading day — Jan 9 is the first order fill)
FUND_INCEPTION = date_type(2026, 1, 9)


# ============ REGIME CLASSIFICATION ============
# Must match _get_regime_status() in app.py

def classify_regime(vix_val: float | None, hy_val: float | None) -> str:
    """Determine regime from VIX and HY OAS values."""
    if vix_val is None:
        return "UNKNOWN"
    if vix_val > 25 or (hy_val is not None and hy_val > 400):
        return "RISK-OFF"
    elif vix_val > 18 or (hy_val is not None and hy_val > 350):
        return "CAUTIOUS"
    else:
        return "RISK-ON"


# ============ BACKFILL ============

def backfill_regime_history(app):
    """
    Backfill regime snapshots from fund inception to today.
    Fetches historical VIX from FMP and HY OAS from FRED, then classifies
    each trading day and persists to the regime_snapshots table.
    """
    from ai_options_trader.config import load_settings

    with app.app_context():
        settings = load_settings()
        today = datetime.now(timezone.utc).date()
        start_str = FUND_INCEPTION.isoformat()
        end_str = today.isoformat()

        print(f"[RegimeHistory] Backfilling from {start_str} to {end_str}")

        # --- Fetch historical VIX from FMP ---
        vix_by_date = {}
        yield_by_date = {}
        try:
            fmp_key = settings.FMP_API_KEY
            if fmp_key:
                # VIX
                url = "https://financialmodelingprep.com/api/v3/historical-price-full/%5EVIX"
                resp = requests.get(url, params={
                    "apikey": fmp_key,
                    "from": start_str,
                    "to": end_str,
                }, timeout=30)
                data = resp.json()
                for row in data.get("historical", []):
                    d = row.get("date")
                    c = row.get("close")
                    if d and c:
                        vix_by_date[d] = float(c)

                # 10Y yield
                url2 = "https://financialmodelingprep.com/api/v3/historical-price-full/%5ETNX"
                resp2 = requests.get(url2, params={
                    "apikey": fmp_key,
                    "from": start_str,
                    "to": end_str,
                }, timeout=30)
                data2 = resp2.json()
                for row in data2.get("historical", []):
                    d = row.get("date")
                    c = row.get("close")
                    if d and c:
                        price = float(c)
                        yield_by_date[d] = price / 100.0 if price > 20 else price
        except Exception as e:
            print(f"[RegimeHistory] FMP fetch error: {e}")

        # --- Fetch historical HY OAS from FRED ---
        hy_by_date = {}
        try:
            fred_key = getattr(settings, 'FRED_API_KEY', None)
            if fred_key:
                from ai_options_trader.data.fred import FredClient
                fred = FredClient(api_key=fred_key)
                df = fred.fetch_series(
                    series_id="BAMLH0A0HYM2",
                    start_date=start_str,
                    refresh=False,
                )
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        d = str(row["date"])[:10]
                        v = row["value"]
                        if pd.notna(v):
                            hy_by_date[d] = float(v) * 100.0  # convert to bps
        except Exception as e:
            print(f"[RegimeHistory] FRED fetch error: {e}")

        # --- Fetch 2s10s curve from FRED ---
        curve_by_date = {}
        try:
            fred_key = getattr(settings, 'FRED_API_KEY', None)
            if fred_key:
                from ai_options_trader.data.fred import FredClient
                fred = FredClient(api_key=fred_key)
                df = fred.fetch_series(
                    series_id="T10Y2Y",
                    start_date=start_str,
                    refresh=False,
                )
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        d = str(row["date"])[:10]
                        v = row["value"]
                        if pd.notna(v):
                            curve_by_date[d] = float(v) * 100.0  # convert to bps
        except Exception as e:
            print(f"[RegimeHistory] FRED curve fetch error: {e}")

        # --- Build daily snapshots ---
        # Use all dates we have VIX data for (trading days)
        all_dates = sorted(set(list(vix_by_date.keys()) + list(hy_by_date.keys())))
        count = 0
        last_hy = None  # FRED HY updates less frequently; carry forward

        for date_str in all_dates:
            try:
                d = date_type.fromisoformat(date_str)
            except ValueError:
                continue
            if d < FUND_INCEPTION:
                continue

            vix_val = vix_by_date.get(date_str)
            hy_val = hy_by_date.get(date_str, last_hy)
            if hy_val is not None:
                last_hy = hy_val
            yield_val = yield_by_date.get(date_str)
            curve_val = curve_by_date.get(date_str)

            if vix_val is None:
                continue

            regime = classify_regime(vix_val, hy_val)

            RegimeSnapshot.upsert(
                snapshot_date=d,
                regime=regime,
                vix=vix_val,
                hy_oas=hy_val,
                yield_10y=yield_val,
                curve_2s10s=curve_val,
            )
            count += 1

        print(f"[RegimeHistory] Backfill complete: {count} snapshots upserted")
        return count


def snapshot_today(app, vix_val=None, hy_val=None, yield_val=None,
                   cpi_val=None, curve_val=None):
    """Persist a regime snapshot for today (called from Palmer refresh)."""
    with app.app_context():
        today = datetime.now(timezone.utc).date()
        regime = classify_regime(vix_val, hy_val)
        RegimeSnapshot.upsert(
            snapshot_date=today,
            regime=regime,
            vix=vix_val,
            hy_oas=hy_val,
            yield_10y=yield_val,
            cpi_yoy=cpi_val,
            curve_2s10s=curve_val,
        )
        print(f"[RegimeHistory] Snapshot saved: {today} -> {regime} (VIX={vix_val})")


# ============ QUERY HELPERS ============

def get_regime_performance(app, closed_trades: list[dict]) -> dict:
    """
    Tag each closed trade with its entry-date regime, then compute
    per-regime performance metrics.

    Returns:
        {
            "by_regime": {
                "RISK-ON": { trades, wins, losses, win_rate, avg_return,
                             profit_factor, avg_hold_days, avg_cost, ... },
                ...
            },
            "trade_regimes": [ { ...trade, "entry_regime": "RISK-ON" }, ... ],
        }
    """
    with app.app_context():
        snapshots = RegimeSnapshot.get_all_ordered()

    if not snapshots:
        return {"by_regime": {}, "trade_regimes": []}

    # Build date -> regime lookup (forward-fill)
    regime_map = {}
    for s in snapshots:
        regime_map[s.date] = s.regime

    def lookup_regime(dt) -> str:
        """Get regime for a datetime, using closest prior date.
        Falls back to the first available snapshot if trade predates all snapshots."""
        if dt is None:
            return "UNKNOWN"
        if isinstance(dt, datetime):
            d = dt.date()
        elif isinstance(dt, date_type):
            d = dt
        else:
            try:
                d = datetime.fromisoformat(str(dt).replace("Z", "+00:00")).date()
            except Exception:
                return "UNKNOWN"

        sorted_dates = sorted(regime_map.keys())
        if not sorted_dates:
            return "UNKNOWN"

        # Find closest prior or equal date
        best = None
        for snap_date in sorted_dates:
            if snap_date <= d:
                best = snap_date
            else:
                break

        # If no prior snapshot, use the first available (trade just before data starts)
        if best is None:
            best = sorted_dates[0]

        return regime_map[best]

    # Tag trades
    tagged = []
    for t in closed_trades:
        entry_regime = lookup_regime(t.get("entry_date"))
        tagged.append({**t, "entry_regime": entry_regime})

    # Group by regime
    regimes = {}
    for t in tagged:
        r = t["entry_regime"]
        if r == "UNKNOWN":
            continue
        if r not in regimes:
            regimes[r] = []
        regimes[r].append(t)

    # Compute metrics per regime
    by_regime = {}
    for regime_label, trades in regimes.items():
        wins = [t for t in trades if t.get("pnl", 0) >= 0]
        losses = [t for t in trades if t.get("pnl", 0) < 0]
        total = len(trades)
        win_count = len(wins)

        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0

        # Average return %
        avg_return = (sum(t.get("pnl_pct", 0) for t in trades) / total) if total else 0

        # Average cost (position size)
        avg_cost = (sum(t.get("cost", 0) for t in trades) / total) if total else 0

        # Average hold time
        hold_days = []
        for t in trades:
            entry = t.get("entry_date")
            exit_ = t.get("exit_date")
            if entry and exit_:
                try:
                    if isinstance(entry, str):
                        entry = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                    if isinstance(exit_, str):
                        exit_ = datetime.fromisoformat(exit_.replace("Z", "+00:00"))
                    delta = (exit_ - entry).total_seconds() / 86400
                    hold_days.append(delta)
                except Exception:
                    pass
        avg_hold = (sum(hold_days) / len(hold_days)) if hold_days else None

        # Option type breakdown
        puts = sum(1 for t in trades if "P" in t.get("symbol", "").upper()
                   and "$" in t.get("symbol", ""))
        calls = sum(1 for t in trades if "C" in t.get("symbol", "").upper()
                    and "$" in t.get("symbol", ""))

        pf_raw = (gross_profit / gross_loss) if gross_loss > 0 else (
            999.0 if gross_profit > 0 else 0
        )

        by_regime[regime_label] = {
            "trades": total,
            "wins": win_count,
            "losses": len(losses),
            "win_rate": (win_count / total * 100) if total else 0,
            "avg_return": avg_return,
            "profit_factor": min(pf_raw, 999.0),  # Cap for JSON serialization
            "avg_hold_days": round(avg_hold, 1) if avg_hold else None,
            "avg_cost": avg_cost,
            "total_pnl": sum(t.get("pnl", 0) for t in trades),
            "puts": puts,
            "calls": calls,
        }

    return {"by_regime": by_regime, "trade_regimes": tagged}


def get_regime_transitions(app, closed_trades: list[dict]) -> list[dict]:
    """
    Detect regime transitions and annotate with nearby trade performance.

    Returns list of:
        { date, from_regime, to_regime, trades_around: [...],
          pnl_5d_after, positions_reduced }
    """
    with app.app_context():
        snapshots = RegimeSnapshot.get_all_ordered()

    if len(snapshots) < 2:
        return []

    transitions = []
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]
        if prev.regime != curr.regime:
            transitions.append({
                "date": curr.date.isoformat(),
                "from_regime": prev.regime,
                "to_regime": curr.regime,
                "vix": curr.vix,
                "hy_oas": curr.hy_oas,
            })

    # Annotate with trades around each transition
    for trans in transitions:
        trans_date = date_type.fromisoformat(trans["date"])
        window_start = trans_date - timedelta(days=5)
        window_end = trans_date + timedelta(days=5)

        nearby_trades = []
        pnl_after = 0
        for t in closed_trades:
            entry = t.get("entry_date")
            exit_ = t.get("exit_date")
            if entry is None:
                continue
            try:
                if isinstance(entry, str):
                    entry_d = datetime.fromisoformat(entry.replace("Z", "+00:00")).date()
                elif isinstance(entry, datetime):
                    entry_d = entry.date()
                else:
                    continue
            except Exception:
                continue

            if window_start <= entry_d <= window_end:
                nearby_trades.append({
                    "symbol": t.get("symbol", "?"),
                    "pnl": t.get("pnl", 0),
                    "entry_date": str(entry)[:10],
                })
                # Trades entered after transition
                if entry_d >= trans_date:
                    pnl_after += t.get("pnl", 0)

        trans["trades_around"] = nearby_trades
        trans["pnl_5d_after"] = pnl_after
        trans["trades_count"] = len(nearby_trades)

    return transitions


def get_regime_timeline(app, closed_trades: list[dict],
                        open_positions: list[dict] | None = None) -> list[dict]:
    """
    Build a merged chronological timeline of regime changes, closed trades,
    and open positions.
    Returns list of { type: "regime_change"|"trade_exit"|"open_position", date, ... }
    """
    with app.app_context():
        snapshots = RegimeSnapshot.get_all_ordered()

    events = []

    # Add regime change events
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]
        if prev.regime != curr.regime:
            events.append({
                "type": "regime_change",
                "date": curr.date.isoformat(),
                "from_regime": prev.regime,
                "to_regime": curr.regime,
                "vix": curr.vix,
                "hy_oas": curr.hy_oas,
            })

    # Add first regime as "start" event
    if snapshots:
        events.append({
            "type": "regime_start",
            "date": snapshots[0].date.isoformat(),
            "regime": snapshots[0].regime,
            "vix": snapshots[0].vix,
            "hy_oas": snapshots[0].hy_oas,
        })

    # Add closed trade exit events
    for t in closed_trades:
        entry = t.get("entry_date")
        exit_ = t.get("exit_date")

        entry_str = None
        if entry:
            try:
                if isinstance(entry, datetime):
                    entry_str = entry.date().isoformat()
                else:
                    entry_str = str(entry)[:10]
            except Exception:
                pass

        exit_str = None
        if exit_:
            try:
                if isinstance(exit_, datetime):
                    exit_str = exit_.date().isoformat()
                else:
                    exit_str = str(exit_)[:10]
            except Exception:
                pass

        if exit_str:
            events.append({
                "type": "trade_exit",
                "date": exit_str,
                "symbol": t.get("symbol", "?"),
                "pnl": t.get("pnl", 0),
                "pnl_pct": t.get("pnl_pct", 0),
                "entry_date": entry_str,
            })

    # Add open position events (placed at their actual entry date)
    if open_positions:
        today_str = datetime.now(timezone.utc).date().isoformat()
        for pos in open_positions:
            # Build display symbol
            symbol = pos.get("symbol", "?")
            opt_info = pos.get("opt_info")
            if opt_info:
                opt_type = (opt_info.get("opt_type", "") or "").upper()
                t = "C" if opt_type.startswith("C") else "P"
                symbol = f"{opt_info.get('underlying', '?')} ${opt_info.get('strike', '?')}{t} {opt_info.get('expiry', '')}"

            pnl = pos.get("pnl", 0)
            mv = pos.get("market_value", 0)
            cost = abs(mv) - pnl if mv else 0
            pnl_pct = (pnl / abs(cost) * 100) if cost and abs(cost) > 0.01 else 0

            # Use actual entry date if available, else today
            entry_dt = pos.get("_entry_date")
            if entry_dt:
                if isinstance(entry_dt, datetime):
                    entry_date_str = entry_dt.date().isoformat()
                else:
                    entry_date_str = str(entry_dt)[:10]
            else:
                entry_date_str = today_str

            events.append({
                "type": "open_position",
                "date": entry_date_str,
                "symbol": symbol,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "qty": pos.get("qty", 0),
                "market_value": mv,
            })

    # Sort chronologically
    events.sort(key=lambda e: e.get("date", "0"))
    return events


def get_edge_summary(app, closed_trades: list[dict], nav_twr: float = None,
                     spy_return: float = None) -> dict:
    """
    Compute headline 'edge' stats for the resume card.
    """
    perf = get_regime_performance(app, closed_trades)
    by_regime = perf.get("by_regime", {})

    # Best environment by profit factor
    best_regime = None
    best_pf = 0
    for regime_label, stats in by_regime.items():
        pf = stats.get("profit_factor", 0)
        if pf == float("inf"):
            pf = 999
        if pf > best_pf:
            best_pf = pf
            best_regime = regime_label

    best_wr = by_regime[best_regime]["win_rate"] if best_regime else 0

    # Alpha (nav_twr is decimal like 0.357, spy_return is pct like -0.5)
    alpha = None
    if nav_twr is not None and spy_return is not None:
        alpha = (nav_twr * 100) - spy_return

    # Total stats
    total_trades = sum(s["trades"] for s in by_regime.values())
    total_wins = sum(s["wins"] for s in by_regime.values())
    overall_wr = (total_wins / total_trades * 100) if total_trades else 0

    # Regime anticipation: did we reduce exposure before risk-off?
    transitions = get_regime_transitions(app, closed_trades)
    risk_off_transitions = [t for t in transitions if t["to_regime"] == "RISK-OFF"]
    anticipated = 0
    for t in risk_off_transitions:
        # If fewer trades entered in the 5 days before, count as anticipated
        if t["trades_count"] == 0 or t["pnl_5d_after"] >= 0:
            anticipated += 1
    anticipation_text = None
    if risk_off_transitions:
        anticipation_text = f"{anticipated}/{len(risk_off_transitions)} risk-off shifts"

    return {
        "alpha": round(alpha, 2) if alpha is not None else None,
        "nav_twr": round(nav_twr * 100, 1) if nav_twr is not None else None,
        "spy_return": round(spy_return, 1) if spy_return is not None else None,
        "best_regime": best_regime,
        "best_regime_wr": round(best_wr, 0) if best_wr else None,
        "best_regime_pf": round(best_pf, 1) if best_pf and best_pf < 999 else None,
        "total_trades": total_trades,
        "overall_win_rate": round(overall_wr, 0),
        "regime_anticipation": anticipation_text,
        "regimes_active": len(by_regime),
    }
