"""Ticker display — Rich console panels and tables for price, fundamentals, technicals, peers, ETF flows."""
from __future__ import annotations

import logging

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lox.utils.formatting import safe_float
from lox.cli_commands.research.ticker.data import fetch_peers

logger = logging.getLogger(__name__)


def show_price_panel(
    console: Console,
    symbol: str,
    price_data: dict,
    technicals: dict,
    implied_vol: float | None = None,
):
    """Display price information panel. Vol and support/resistance show method; trend is factual."""
    quote = price_data.get("quote", {})
    price = quote.get("price", technicals.get("current", 0))
    change = quote.get("change", 0)
    change_pct = quote.get("changesPercentage", 0)
    change_color = "green" if change >= 0 else "red"

    vol_30d = technicals.get("volatility_30d") or technicals.get("volatility")
    vol_str = f"Historical Vol (30d, ann.): {vol_30d:.1f}%" if vol_30d is not None else "Historical Vol: N/A"
    if implied_vol is not None:
        iv_pct = implied_vol * 100
        vol_str += f"  |  Implied Vol (ATM): {iv_pct:.1f}%"
        if vol_30d is not None:
            spread = iv_pct - vol_30d
            vol_str += f"  (IV–HV: {spread:+.1f}%)"

    sup = technicals.get("support")
    res = technicals.get("resistance")
    sup_method = technicals.get("support_method") or "N/A"
    res_method = technicals.get("resistance_method") or "N/A"
    support_str = f"${sup:,.2f} ({sup_method})" if sup is not None else "N/A"
    resistance_str = f"${res:,.2f} ({res_method})" if res is not None else "N/A"
    trend_label = technicals.get("trend_label") or technicals.get("trend", "N/A")

    content = f"""[bold]{symbol}[/bold]  ${price:,.2f}  [{change_color}]{change:+.2f} ({change_pct:+.2f}%)[/{change_color}]

52W Range: ${technicals.get('low_52w', 0):,.2f} - ${technicals.get('high_52w', 0):,.2f}
Support: {support_str}  |  Resistance: {resistance_str}
Trend: {trend_label}  |  {vol_str}"""
    console.print(Panel(content, title="[bold]Price[/bold]", border_style="blue"))


def show_fundamentals(
    console: Console,
    fundamentals: dict,
    technicals: dict | None = None,
    price_data: dict | None = None,
):
    """Display fundamentals table (detects ETFs vs stocks)."""
    profile = fundamentals.get("profile", {})
    etf_info = fundamentals.get("etf_info", {})

    is_etf = profile.get("isEtf", False) or bool(etf_info)
    technicals = technicals or {}
    price_data = price_data or {}

    if is_etf:
        _show_etf_fundamentals(console, profile, etf_info, price_data)
    else:
        _show_stock_fundamentals(console, fundamentals, technicals)


def show_key_risks_summary(
    console: Console,
    symbol: str,
    fundamentals: dict,
    technicals: dict,
):
    """Auto-generated key risks box (red/green flags) from data — not LLM."""
    profile = fundamentals.get("profile", {})
    ratios = fundamentals.get("ratios", {})
    etf_info = fundamentals.get("etf_info", {})
    is_etf = profile.get("isEtf", False) or bool(etf_info)
    lines = []

    if not is_etf:
        pe_f = safe_float(ratios.get("peRatioTTM"))
        if pe_f is not None and pe_f < 0:
            lines.append(("⚠️", "Negative earnings (TTM EPS < 0)", "red"))
        pb_f = safe_float(ratios.get("priceToBookRatioTTM"))
        if pb_f is not None and pb_f > 10:
            lines.append(("⚠️", f"P/B of {pb_f:.0f}x (asset-light risk)", "red"))
        nm_f = safe_float(ratios.get("netProfitMarginTTM"))
        if nm_f is not None and nm_f < 0:
            lines.append(("⚠️", "Negative net margin", "red"))
        vol = technicals.get("volatility_30d") or technicals.get("volatility")
        if vol is not None and vol > 50:
            lines.append(("⚠️", f"{vol:.1f}% ann. volatility", "red"))
        income = (fundamentals.get("income_statement") or [{}])[0]
        income_prev = (fundamentals.get("income_statement") or [None, None])[1] if len(fundamentals.get("income_statement") or []) >= 2 else None
        rev = safe_float(income.get("revenue"))
        rev_prev = safe_float(income_prev.get("revenue") if income_prev else None)
        if rev and rev_prev:
            growth = (rev - rev_prev) / rev_prev * 100
            if growth > 0:
                lines.append(("✅", f"Revenue growing {growth:.1f}% YoY", "green"))
        trend_label = technicals.get("trend_label") or ""
        if "Above all major" in (trend_label or ""):
            lines.append(("✅", "Above all major moving averages", "green"))
    else:
        nav_f = safe_float(etf_info.get("nav"))
        price_f = safe_float(profile.get("price"))
        if nav_f and price_f:
            prem = (price_f - nav_f) / nav_f * 100
            if prem > 3:
                lines.append(("⚠️", f"NAV premium +{prem:.1f}%", "red"))
            elif prem < -3:
                lines.append(("⚠️", f"NAV discount {prem:.1f}%", "yellow"))

    if not lines:
        return
    text = Text()
    for icon, msg, color in lines:
        text.append(f"{icon}  {msg}\n", style=color)
    console.print()
    console.print(Panel(text, title="[bold]Key Risks[/bold]", border_style="dim"))


def show_peer_comparison(console: Console, settings, symbol: str, fundamentals: dict):
    """Structured peer comparison table (Ticker, Price, Mkt Cap, EV/Rev, Rev Growth, Gross Margin, Net Margin, P/E)."""
    peers = fetch_peers(settings, symbol)
    if not peers:
        return
    try:
        import requests
        symbols = [symbol] + list(peers)
        joined = ",".join(symbols)
        api = settings.fmp_api_key

        # Batch quote (price)
        resp_q = requests.get(
            f"https://financialmodelingprep.com/api/v3/quote/{joined}",
            params={"apikey": api}, timeout=15,
        )
        quotes = resp_q.json() if resp_q.ok and isinstance(resp_q.json(), list) else []
        quote_by = {q["symbol"]: q for q in quotes if isinstance(q, dict) and q.get("symbol")}

        # Batch profile (mktCap, sector)
        resp_p = requests.get(
            f"https://financialmodelingprep.com/api/v3/profile/{joined}",
            params={"apikey": api}, timeout=15,
        )
        profiles = resp_p.json() if resp_p.ok and isinstance(resp_p.json(), list) else []
        prof_by = {p["symbol"]: p for p in profiles if isinstance(p, dict) and p.get("symbol")}

        # Per-symbol detail calls (key-metrics-ttm, ratios-ttm, financial-growth)
        rows = []
        for sym in symbols:
            q = quote_by.get(sym, {})
            p = prof_by.get(sym, {})
            price = q.get("price") or p.get("price")
            mkt_cap = p.get("mktCap")

            # key-metrics-ttm -> evToSalesTTM (pre-calculated EV/Rev)
            try:
                r_m = requests.get(
                    f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{sym}",
                    params={"apikey": api}, timeout=8,
                )
                metrics = (r_m.json() or [{}])[0] if r_m.ok and isinstance(r_m.json(), list) and r_m.json() else {}
            except (requests.RequestException, ValueError, IndexError, KeyError):
                metrics = {}

            # ratios-ttm -> margins, P/E
            try:
                r_r = requests.get(
                    f"https://financialmodelingprep.com/api/v3/ratios-ttm/{sym}",
                    params={"apikey": api}, timeout=8,
                )
                ratios = (r_r.json() or [{}])[0] if r_r.ok and isinstance(r_r.json(), list) and r_r.json() else {}
            except (requests.RequestException, ValueError, IndexError, KeyError):
                ratios = {}

            # financial-growth -> revenueGrowth (more reliable than income-statement-growth)
            try:
                r_g = requests.get(
                    f"https://financialmodelingprep.com/api/v3/financial-growth/{sym}",
                    params={"apikey": api, "limit": 1}, timeout=8,
                )
                growth = (r_g.json() or [{}])[0] if r_g.ok and isinstance(r_g.json(), list) and r_g.json() else {}
            except (requests.RequestException, ValueError, IndexError, KeyError):
                growth = {}

            # EV/Revenue: use pre-calculated evToSalesTTM
            ev_rev = safe_float(metrics.get("evToSalesTTM"))

            # Revenue growth YoY from financial-growth endpoint
            raw_rg = safe_float(growth.get("revenueGrowth"))
            rev_growth = raw_rg * 100 if raw_rg is not None else None

            # Gross margin from ratios-ttm
            raw_gm = safe_float(ratios.get("grossProfitMarginTTM") or metrics.get("grossProfitMarginTTM"))
            gross_margin = raw_gm * 100 if raw_gm is not None else None

            # Net margin from ratios-ttm
            raw_nm = safe_float(ratios.get("netProfitMarginTTM"))
            net_margin = raw_nm * 100 if raw_nm is not None else None

            # P/E from ratios or quote
            pe = safe_float(ratios.get("peRatioTTM") or q.get("pe"))

            rows.append({
                "ticker": sym,
                "price": price,
                "mkt_cap": mkt_cap,
                "ev_rev": ev_rev,
                "rev_growth": rev_growth,
                "gross_margin": gross_margin,
                "net_margin": net_margin,
                "pe": pe,
            })
    except Exception as e:
        logger.debug("Peer comparison failed: %s", e)
        return
    if not rows:
        return

    table = Table(title="[bold]Peer Comparison[/bold]", box=None, padding=(0, 1), header_style="bold dim")
    table.add_column("Ticker", style="bold")
    table.add_column("Price", justify="right")
    table.add_column("Mkt Cap", justify="right")
    table.add_column("P/E", justify="right")
    table.add_column("EV/Rev", justify="right")
    table.add_column("Rev Growth", justify="right")
    table.add_column("Gross Mgn", justify="right")
    table.add_column("Net Mgn", justify="right")
    for r in rows:
        is_target = r["ticker"] == symbol
        mc = r.get("mkt_cap")
        mc_str = f"${mc/1e9:.1f}B" if mc is not None else "—"
        price = r.get("price")
        price_str = f"${price:,.2f}" if price is not None else "—"
        ev_rev = r.get("ev_rev")
        ev_rev_str = f"{ev_rev:.1f}x" if ev_rev is not None else "—"
        pe = r.get("pe")
        pe_str = f"{pe:.1f}x" if pe is not None else "—"
        rg = r.get("rev_growth")
        rg_color = "green" if rg and rg > 0 else "red" if rg and rg < 0 else ""
        rg_str = f"[{rg_color}]{rg:+.1f}%[/{rg_color}]" if rg is not None else "—"
        gm = r.get("gross_margin")
        gm_str = f"{gm:.1f}%" if gm is not None else "—"
        nm = r.get("net_margin")
        nm_color = "green" if nm and nm > 0 else "red" if nm and nm < 0 else ""
        nm_str = f"[{nm_color}]{nm:.1f}%[/{nm_color}]" if nm is not None else "—"
        ticker_style = f"[bold cyan]{r['ticker']}[/bold cyan]" if is_target else r["ticker"]
        table.add_row(ticker_style, price_str, mc_str, pe_str, ev_rev_str, rg_str, gm_str, nm_str)
    console.print()
    console.print(table)


def _show_etf_fundamentals(console: Console, profile: dict, etf_info: dict, price_data: dict):
    """Display ETF-specific fundamentals. NAV premium highlighted; physical trust holdings fixed."""
    table = Table(title="[bold]ETF Profile[/bold]", box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    def fmt(val, pct=False, billions=False, dollar=False):
        if val is None:
            return "N/A"
        try:
            v = float(val)
            if billions:
                return f"${v/1e9:.1f}B"
            if pct:
                return f"{v:.2f}%"
            if dollar:
                return f"${v:.2f}"
            return f"{v:,.0f}"
        except Exception:
            return str(val)[:20]

    aum = etf_info.get("aum") or profile.get("mktCap")
    expense = etf_info.get("expenseRatio")
    nav = etf_info.get("nav")
    holdings_raw = etf_info.get("holdingsCount")
    div_yield = profile.get("lastDiv")
    price = profile.get("price") or (price_data.get("historical", [{}])[-1].get("close") if price_data.get("historical") else None)
    if not price and price_data.get("quote"):
        price = price_data["quote"].get("price")
    beta = profile.get("beta")
    inception = etf_info.get("inceptionDate") or profile.get("ipoDate")
    asset_class = (etf_info.get("assetClass") or "N/A").lower()
    company = etf_info.get("etfCompany", "N/A")
    desc_lower = (etf_info.get("description") or profile.get("description") or "").lower()

    # Holdings: physical commodity trusts often show 0 — show descriptive text
    if holdings_raw == 0 or holdings_raw is None:
        if "physical" in desc_lower or "commodity" in asset_class or "trust" in desc_lower or "precious" in desc_lower:
            holdings_str = "Physical (trust structure)"
        else:
            holdings_str = "N/A"
    else:
        holdings_str = fmt(holdings_raw)

    # NAV premium/discount — standalone highlighted line
    nav_val = etf_info.get("nav")
    current_price = float(price) if price is not None else None
    prem_disc_pct = None
    if nav_val is not None and current_price is not None:
        try:
            nav_f = float(nav_val)
            if nav_f > 0:
                prem_disc_pct = ((current_price - nav_f) / nav_f) * 100
        except (TypeError, ValueError):
            pass

    table.add_row("AUM", fmt(aum, billions=True), "Expense Ratio", fmt(expense, pct=True))
    table.add_row("NAV", fmt(nav, dollar=True), "Holdings", holdings_str)
    table.add_row("Yield", fmt((float(div_yield) / float(price) * 100) if div_yield and price else None, pct=True) or fmt(div_yield, dollar=True), "Beta (vs SPY)", fmt(beta))
    table.add_row("Asset Class", str(etf_info.get("assetClass", "N/A"))[:15], "Issuer", str(company)[:15])
    table.add_row("Inception", str(inception)[:10] if inception else "N/A", "Avg Volume", fmt(etf_info.get("avgVolume")))

    console.print()
    console.print(table)

    # NAV Premium/Discount — highlighted (green <1%, yellow 1–3%, red >3%)
    if prem_disc_pct is not None:
        if prem_disc_pct < 1 and prem_disc_pct > -1:
            color = "green"
        elif abs(prem_disc_pct) <= 3:
            color = "yellow"
        else:
            color = "red"
        label = "NAV Premium" if prem_disc_pct > 0 else "NAV Discount"
        msg = " — trading significantly above NAV" if prem_disc_pct > 3 else ""
        if prem_disc_pct < -3:
            msg = " — trading significantly below NAV"
        console.print(f"  [{color}]⚠ {label}: {prem_disc_pct:+.2f}%{msg}[/{color}]")
        console.print()

    desc = etf_info.get("description") or profile.get("description")
    if desc:
        console.print(f"[dim]{desc[:200]}[/dim]")


def show_etf_flows(console: Console, price_data: dict, fundamentals: dict):
    """Compute and display ETF flow signals from volume data."""
    historical = price_data.get("historical", [])
    if len(historical) < 21:
        return

    # Sort oldest first
    hist = list(reversed(historical[:60]))

    closes = [h["close"] for h in hist]
    volumes = [h["volume"] for h in hist]
    highs = [h["high"] for h in hist]
    lows = [h["low"] for h in hist]

    # Money Flow Index (14-day)
    mfi = None
    if len(hist) >= 15:
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        pos_mf = sum(
            typical_prices[i] * volumes[i]
            for i in range(-14, 0)
            if typical_prices[i] > typical_prices[i - 1]
        )
        neg_mf = sum(
            typical_prices[i] * volumes[i]
            for i in range(-14, 0)
            if typical_prices[i] <= typical_prices[i - 1]
        )
        if neg_mf > 0:
            mfi = 100 - (100 / (1 + pos_mf / neg_mf))
        elif pos_mf > 0:
            mfi = 100.0

    # Dollar volume averages
    dv = [c * v for c, v in zip(closes, volumes)]
    dv_5d = np.mean(dv[-5:]) if len(dv) >= 5 else None
    dv_20d = np.mean(dv[-20:]) if len(dv) >= 20 else None

    # Volume vs 20d average
    vol_20d = np.mean(volumes[-20:])
    vol_today = volumes[-1]
    vol_ratio = vol_today / vol_20d if vol_20d > 0 else 1.0

    # OBV trend
    obv = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])

    obv_5d = "Inflows" if len(obv) >= 6 and obv[-1] > obv[-6] else "Outflows"
    obv_20d = "Inflows" if len(obv) >= 21 and obv[-1] > obv[-21] else "Outflows"

    # Up/Down volume ratio (20d)
    if len(closes) >= 21:
        up_vol = sum(v for c1, c0, v in zip(closes[-20:], closes[-21:-1], volumes[-20:]) if c1 > c0)
        dn_vol = sum(v for c1, c0, v in zip(closes[-20:], closes[-21:-1], volumes[-20:]) if c1 <= c0)
        flow_ratio = up_vol / dn_vol if dn_vol > 0 else 999.0
    else:
        flow_ratio = 1.0

    # Net flow signal
    if flow_ratio > 1.3:
        net_signal = "[green]STRONG INFLOWS[/green]"
    elif flow_ratio > 1.1:
        net_signal = "[green]INFLOWS[/green]"
    elif flow_ratio < 0.7:
        net_signal = "[red]STRONG OUTFLOWS[/red]"
    elif flow_ratio < 0.9:
        net_signal = "[red]OUTFLOWS[/red]"
    else:
        net_signal = "[yellow]BALANCED[/yellow]"

    # Premium/Discount to NAV
    etf_info = fundamentals.get("etf_info", {})
    nav = etf_info.get("nav")
    current_price = closes[-1] if closes else None
    prem_disc = None
    if nav and current_price:
        try:
            prem_disc = ((current_price - float(nav)) / float(nav)) * 100
        except Exception:
            pass

    # MFI signal
    mfi_signal = ""
    if mfi is not None:
        if mfi > 80:
            mfi_signal = "[red]Overbought[/red]"
        elif mfi < 20:
            mfi_signal = "[green]Oversold[/green]"
        elif mfi < 40:
            mfi_signal = "[yellow]Weak[/yellow]"
        elif mfi > 60:
            mfi_signal = "[green]Strong[/green]"
        else:
            mfi_signal = "Neutral"

    # Build table
    table = Table(title="[bold]Fund Flows (Volume-Based)[/bold]", box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row(
        "20d Net Flow",
        net_signal,
        "MFI (14d)",
        f"{mfi:.0f} {mfi_signal}" if mfi is not None else "N/A",
    )

    obv_5d_color = "green" if obv_5d == "Inflows" else "red"
    obv_20d_color = "green" if obv_20d == "Inflows" else "red"
    table.add_row(
        "OBV 5d",
        f"[{obv_5d_color}]{obv_5d}[/{obv_5d_color}]",
        "OBV 20d",
        f"[{obv_20d_color}]{obv_20d}[/{obv_20d_color}]",
    )

    table.add_row(
        "$ Vol 5d Avg",
        f"${dv_5d / 1e6:.0f}M" if dv_5d else "N/A",
        "$ Vol 20d Avg",
        f"${dv_20d / 1e6:.0f}M" if dv_20d else "N/A",
    )

    vol_color = "green" if vol_ratio > 1.2 else "red" if vol_ratio < 0.8 else "yellow"
    table.add_row(
        "Vol vs 20d Avg",
        f"[{vol_color}]{vol_ratio:.2f}x[/{vol_color}]",
        "Up/Down Ratio",
        f"{'[green]' if flow_ratio > 1.1 else '[red]' if flow_ratio < 0.9 else ''}{flow_ratio:.2f}x{'[/green]' if flow_ratio > 1.1 else '[/red]' if flow_ratio < 0.9 else ''}",
    )

    if prem_disc is not None:
        pd_color = "green" if prem_disc > 0.1 else "red" if prem_disc < -0.1 else "yellow"
        table.add_row(
            "NAV Premium",
            f"[{pd_color}]{prem_disc:+.2f}%[/{pd_color}]",
            "",
            "",
        )

    console.print()
    console.print(table)


def show_refinancing_wall(console: Console, settings, symbol: str):
    """Fetch bond ETF holdings and display maturity (refinancing) wall."""
    import re
    import requests
    from collections import defaultdict
    from datetime import datetime

    try:
        url = f"https://financialmodelingprep.com/api/v3/etf-holder/{symbol}"
        resp = requests.get(
            url, params={"apikey": settings.fmp_api_key}, timeout=20
        )
        if not resp.ok:
            return
        holdings = resp.json()
        if not holdings or not isinstance(holdings, list):
            return
    except Exception:
        return

    # Parse maturity year from bond names (format: "COMPANY 144A MM/DD/YYYY")
    date_pattern = re.compile(r"(\d{2}/\d{2}/(\d{4}))\s*$")
    by_year: dict[int, dict] = defaultdict(lambda: {"count": 0, "mv": 0, "weight": 0.0})
    total_mv = 0
    parsed = 0
    current_year = datetime.now().year

    for h in holdings:
        name = h.get("name", "")
        mv = h.get("marketValue", 0) or 0
        wt = h.get("weightPercentage", 0) or 0
        match = date_pattern.search(name)
        if match:
            year = int(match.group(2))
            by_year[year]["count"] += 1
            by_year[year]["mv"] += mv
            by_year[year]["weight"] += wt
            total_mv += mv
            parsed += 1

    if parsed < 10 or total_mv <= 0:
        return  # Not enough bond data

    # Build display — only show meaningful years
    years_to_show = sorted(y for y in by_year if current_year <= y <= current_year + 12)
    if not years_to_show:
        return

    max_mv = max(by_year[y]["mv"] for y in years_to_show) if years_to_show else 1

    # Near-term / mid-term buckets
    near_mv = sum(by_year[y]["mv"] for y in years_to_show if y <= current_year + 2)
    mid_mv = sum(by_year[y]["mv"] for y in years_to_show if current_year + 3 <= y <= current_year + 5)
    near_pct = near_mv / total_mv * 100 if total_mv else 0
    mid_pct = mid_mv / total_mv * 100 if total_mv else 0

    # Build table
    table = Table(
        title=f"[bold]Refinancing Wall ({parsed:,} bonds, ${total_mv / 1e9:.1f}B)[/bold]",
        box=None, padding=(0, 1),
    )
    table.add_column("Year", style="bold", min_width=6, no_wrap=True)
    table.add_column("Bonds", justify="right", min_width=6)
    table.add_column("Mkt Value", justify="right", min_width=10)
    table.add_column("% Fund", justify="right", min_width=7)
    table.add_column("Distribution", min_width=30)

    for year in years_to_show:
        d = by_year[year]
        pct = d["mv"] / total_mv * 100
        bar_len = int(d["mv"] / max_mv * 25)

        # Color: red for near-term (pressure), yellow for mid-term, green for far
        if year <= current_year + 1:
            color = "red"
        elif year <= current_year + 3:
            color = "yellow"
        else:
            color = "green"

        bar = f"[{color}]{'█' * bar_len}[/{color}]"
        year_str = f"[{color}]{year}[/{color}]"
        pct_str = f"[{color}]{pct:.1f}%[/{color}]"

        table.add_row(
            year_str,
            str(d["count"]),
            f"${d['mv'] / 1e9:.2f}B",
            pct_str,
            bar,
        )

    console.print()
    console.print(table)

    # Summary line
    near_color = "red" if near_pct > 15 else "yellow" if near_pct > 8 else "green"
    console.print(
        f"\n  [{near_color}]Near-term (≤{current_year + 2}): ${near_mv / 1e9:.1f}B ({near_pct:.0f}%)[/{near_color}]"
        f"  |  Mid-term ({current_year + 3}-{current_year + 5}): ${mid_mv / 1e9:.1f}B ({mid_pct:.0f}%)"
    )


def _show_stock_fundamentals(console: Console, fundamentals: dict, technicals: dict):
    """Display stock fundamentals (CFA-aligned). Negative vs positive earnings; EV/Rev, growth, leverage, liquidity."""
    profile = fundamentals.get("profile", {})
    metrics = fundamentals.get("metrics", {})
    ratios = fundamentals.get("ratios", {})
    income = (fundamentals.get("income_statement") or [{}])[0]
    income_prev = (fundamentals.get("income_statement") or [None, None])[1] if len(fundamentals.get("income_statement") or []) >= 2 else None
    cash_flow = (fundamentals.get("cash_flow") or [{}])[0]
    balance = fundamentals.get("balance_sheet") or {}
    growth_list = fundamentals.get("income_growth") or []

    def fmt(val, pct=False, billions=False):
        if val is None:
            return "N/A"
        try:
            v = float(val)
            if billions:
                return f"${v/1e9:.1f}B"
            if pct:
                return f"{v:.1f}%"
            return f"{v:.2f}"
        except Exception:
            return str(val)[:15]

    mkt_cap = profile.get("mktCap")
    pe = ratios.get("peRatioTTM") or profile.get("pe")
    pe_f = safe_float(pe)
    negative_earnings = pe_f is not None and pe_f < 0

    # Enterprise value (EV = mktCap + totalDebt - cash)
    total_debt = balance.get("totalDebt")
    cash = balance.get("cashAndCashEquivalents")
    ev = metrics.get("enterpriseValue")
    if ev is None and mkt_cap is not None:
        try:
            ev = float(mkt_cap) + float(total_debt or 0) - float(cash or 0)
        except (TypeError, ValueError):
            ev = None

    revenue_f = safe_float(income.get("revenue"))
    gross_profit_f = safe_float(income.get("grossProfit"))
    gross_margin = (gross_profit_f / revenue_f * 100) if revenue_f and gross_profit_f else None
    revenue_prev_f = safe_float(income_prev.get("revenue") if income_prev else None)
    revenue_growth_yoy = None
    if revenue_f and revenue_prev_f:
        revenue_growth_yoy = (revenue_f - revenue_prev_f) / revenue_prev_f * 100
    if revenue_growth_yoy is None and growth_list:
        raw_rg = safe_float(growth_list[0].get("revenueGrowth"))
        if raw_rg is not None:
            revenue_growth_yoy = raw_rg * 100

    ebitda = income.get("ebitda")
    fcf = cash_flow.get("freeCashFlow")
    debt_equity = ratios.get("debtEquityRatio") or metrics.get("debtEquityRatio")
    interest_coverage = ratios.get("interestCoverage") or metrics.get("interestCoverage")
    current_ratio = ratios.get("currentRatio") or metrics.get("currentRatio") or balance.get("totalCurrentAssets") and balance.get("totalCurrentLiabilities") and (float(balance["totalCurrentAssets"]) / float(balance["totalCurrentLiabilities"]) if balance.get("totalCurrentLiabilities") else None)
    net_margin = ratios.get("netProfitMarginTTM")
    roe = ratios.get("returnOnEquityTTM")
    roa = ratios.get("returnOnAssetsTTM")
    ps = ratios.get("priceToSalesRatioTTM")
    pb = ratios.get("priceToBookRatioTTM")

    # 52-week percentile
    current = technicals.get("current")
    high_52w = technicals.get("high_52w")
    low_52w = technicals.get("low_52w")
    pct_52w = None
    if current is not None and high_52w is not None and low_52w is not None and high_52w > low_52w:
        try:
            pct_52w = (float(current) - float(low_52w)) / (float(high_52w) - float(low_52w)) * 100
        except (TypeError, ValueError):
            pass

    ev_f = safe_float(ev)
    ev_rev = (ev_f / revenue_f) if ev_f and revenue_f else None
    ebitda_f = safe_float(ebitda)
    ev_ebitda = (ev_f / ebitda_f) if ev_f and ebitda_f else None
    fcf_f = safe_float(fcf)
    mkt_cap_f = safe_float(mkt_cap)
    fcf_yield = (fcf_f / mkt_cap_f * 100) if fcf_f and mkt_cap_f else None

    table = Table(title="[bold]Fundamentals[/bold]", box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    # P/E: N/M for negative earnings
    pe_display = "N/M (negative earnings)" if negative_earnings else fmt(pe)
    table.add_row("Market Cap", fmt(mkt_cap, billions=True), "P/E", pe_display)

    if negative_earnings:
        table.add_row("EV/Revenue", fmt(ev_rev) if ev_rev else "N/A", "EV/EBITDA", fmt(ev_ebitda) if ev_ebitda else "N/A")
        table.add_row("Revenue Growth (YoY)", fmt(revenue_growth_yoy, pct=True) if revenue_growth_yoy is not None else "N/A", "Gross Margin", fmt(gross_margin, pct=True) if gross_margin is not None else "N/A")
        table.add_row("FCF", fmt(fcf, billions=True) if fcf else "N/A", "FCF Yield", fmt(fcf_yield, pct=True) if fcf_yield else "N/A")
        table.add_row("Debt/Equity", fmt(debt_equity), "Interest Coverage", fmt(interest_coverage))
        table.add_row("Cash & Equiv.", fmt(cash, billions=True) if cash else "N/A", "Current Ratio", fmt(current_ratio))
    else:
        table.add_row("Forward P/E", fmt(profile.get("forwardPE") or ratios.get("forwardPeRatio")), "P/S", fmt(ps))
        table.add_row("P/B", fmt(pb), "PEG Ratio", fmt(ratios.get("pegRatio") or metrics.get("pegRatio")))
        table.add_row("EV/Revenue", fmt(ev_rev) if ev_rev else "N/A", "EV/EBITDA", fmt(ev_ebitda) if ev_ebitda else "N/A")
        table.add_row("Revenue Growth (YoY)", fmt(revenue_growth_yoy, pct=True) if revenue_growth_yoy is not None else "N/A", "Gross Margin", fmt(gross_margin, pct=True) if gross_margin is not None else "N/A")
        table.add_row("FCF", fmt(fcf, billions=True) if fcf else "N/A", "FCF Yield", fmt(fcf_yield, pct=True) if fcf_yield else "N/A")
        table.add_row("Debt/Equity", fmt(debt_equity), "Interest Coverage", fmt(interest_coverage))
        table.add_row("Cash & Equiv.", fmt(cash, billions=True) if cash else "N/A", "Current Ratio", fmt(current_ratio))

    table.add_row("Net Margin", fmt(net_margin, pct=True), "ROE", fmt(roe, pct=True))
    table.add_row("ROA", fmt(roa, pct=True), "Sector", (profile.get("sector") or "N/A")[:15])
    table.add_row("Beta (vs SPY)", fmt(profile.get("beta")), "52W Percentile", f"{pct_52w:.0f}%" if pct_52w is not None else "N/A")

    console.print()
    console.print(table)


def show_technicals(console: Console, technicals: dict):
    """Display technicals table with volume context, 50/200 crossover, MACD."""
    table = Table(title="[bold]Technical Levels[/bold]", box=None, padding=(0, 2))
    table.add_column("Indicator", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Signal")

    current = technicals.get("current", 0)
    ma_20 = technicals.get("ma_20")
    ma_50 = technicals.get("ma_50")
    ma_200 = technicals.get("ma_200")

    def ma_signal(ma):
        if not ma or not current:
            return "[dim]N/A[/dim]"
        if current > ma:
            return "[green]Above[/green]"
        return "[red]Below[/red]"

    table.add_row("20-Day MA", f"${ma_20:,.2f}" if ma_20 else "N/A", ma_signal(ma_20))
    table.add_row("50-Day MA", f"${ma_50:,.2f}" if ma_50 else "N/A", ma_signal(ma_50))
    table.add_row("200-Day MA", f"${ma_200:,.2f}" if ma_200 else "N/A", ma_signal(ma_200))

    # 50/200 SMA crossover
    crossover = technicals.get("sma_crossover")
    if crossover:
        is_golden = "Golden" in crossover
        table.add_row("50/200 SMA", crossover, "[green]Bullish[/green]" if is_golden else "[red]Bearish[/red]")

    # RSI
    rsi = technicals.get("rsi")
    rsi_signal = "[dim]N/A[/dim]"
    if rsi:
        if rsi > 70:
            rsi_signal = "[red]Overbought[/red]"
        elif rsi < 30:
            rsi_signal = "[green]Oversold[/green]"
        else:
            rsi_signal = "[yellow]Neutral[/yellow]"
    table.add_row("RSI (14)", f"{rsi:.1f}" if rsi else "N/A", rsi_signal)

    # MACD
    macd = technicals.get("macd_signal")
    if macd:
        table.add_row("MACD (12,26,9)", macd.split("(")[0].strip(), "[green]Bullish[/green]" if "Bullish" in macd else "[red]Bearish[/red]")

    # Volume context
    avg_vol = technicals.get("avg_volume")
    vol_vs_avg = technicals.get("vol_vs_avg")
    if avg_vol is not None:
        avg_str = f"{avg_vol/1e6:.1f}M" if avg_vol >= 1e6 else f"{avg_vol/1e3:.0f}K"
        vs_str = f"{vol_vs_avg:.2f}x" if vol_vs_avg is not None else "N/A"
        table.add_row("Avg Volume", avg_str, f"Today vs Avg: {vs_str}")

    console.print()
    console.print(table)
