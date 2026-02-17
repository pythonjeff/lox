"""
LOX Research: CVNA ABS Stress Deep-Dive

Short thesis workbench — Carvana's loan book health, credit quality,
auto-loan securitization stress, ABS counterparty exposure,
SEC filings, and catalyst timeline.

Usage:
    lox research cvna
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lox.config import load_settings


def register(app: typer.Typer) -> None:
    """Register the cvna research command."""

    @app.command("cvna")
    def cvna_cmd():
        """
        CVNA ABS Stress Deep-Dive — short thesis workbench.

        Pulls CVNA financials (FMP), auto-loan delinquency data (FRED),
        stock price (Alpaca), SEC filings (EDGAR), ABS counterparty
        exposure, and a catalyst timeline.

        Examples:
            lox research cvna
        """
        console = Console()
        settings = load_settings()

        console.print()
        console.rule("[bold red]CVNA Short Thesis Workbench[/bold red]")
        console.print()

        _show_thesis(console)
        _show_stock(console, settings)
        _show_loan_book(console, settings)
        _show_credit_quality(console, settings)
        _show_abs_exposure(console, settings)
        _show_abs_trust_filings(console)
        _show_filings(console, settings)
        _show_catalysts(console)

        console.print()
        console.rule("[dim]End of report[/dim]")
        console.print()


# ═══════════════════════════════════════════════════════════════════════
# Section 1: Thesis
# ═══════════════════════════════════════════════════════════════════════

def _show_thesis(console: Console) -> None:
    thesis = Text()
    thesis.append("Thesis: ", style="bold")
    thesis.append(
        "Carvana auto-approves subprime borrowers, packages the loans "
        "into ABS tranches (CRVNA trusts), and offloads credit risk to "
        "bond investors. If consumer credit deteriorates — rising "
        "delinquencies, tighter lending standards, higher charge-offs — "
        "the ABS market reprices and CVNA's origination-to-securitization "
        "flywheel stalls.\n\n"
    )
    thesis.append("Kill chain: ", style="bold red")
    thesis.append(
        "Delinquencies rise → ABS spreads widen → warehouse lenders "
        "tighten terms → CVNA retains more risk on balance sheet → "
        "leverage spikes → equity dilution or restructuring.\n"
    )
    console.print(Panel(thesis, title="[bold]Short Thesis[/bold]", border_style="red"))


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Stock Price
# ═══════════════════════════════════════════════════════════════════════

def _show_stock(console: Console, settings) -> None:
    console.print()
    console.print("[bold cyan]CVNA Stock[/bold cyan]")

    try:
        import pandas as pd
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
        from alpaca.data.timeframe import TimeFrame

        data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key
            if hasattr(settings, "alpaca_secret_key")
            else settings.alpaca_api_secret,
        )

        quote_req = StockLatestQuoteRequest(symbol_or_symbols="CVNA")
        quote = data_client.get_stock_latest_quote(quote_req)
        cvna_quote = quote.get("CVNA")
        price = (
            round((float(cvna_quote.ask_price) + float(cvna_quote.bid_price)) / 2, 2)
            if cvna_quote
            else None
        )

        end = date.today()
        start = date(end.year - 1, end.month, end.day)
        bars_req = StockBarsRequest(
            symbol_or_symbols="CVNA",
            timeframe=TimeFrame.Day,
            start=pd.Timestamp(start, tz="UTC"),
            end=pd.Timestamp(end, tz="UTC"),
        )
        bars_df = data_client.get_stock_bars(bars_req).df

        high_52w = None
        low_52w = None
        if bars_df is not None and len(bars_df) > 0:
            df = bars_df.reset_index()
            cvna_rows = df[df["symbol"] == "CVNA"] if "symbol" in df.columns else df
            if not cvna_rows.empty:
                high_52w = float(cvna_rows["high"].max())
                low_52w = float(cvna_rows["low"].min())

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Price", f"${price:,.2f}" if price else "N/A")
        table.add_row("52-Week High", f"${high_52w:,.2f}" if high_52w else "N/A")
        table.add_row("52-Week Low", f"${low_52w:,.2f}" if low_52w else "N/A")
        if price and high_52w and low_52w and high_52w > low_52w:
            pct_from_high = (price - high_52w) / high_52w * 100
            range_pct = (price - low_52w) / (high_52w - low_52w) * 100
            table.add_row("% From 52W High", f"{pct_from_high:+.1f}%")
            table.add_row("52W Range Percentile", f"{range_pct:.0f}%")
        console.print(table)
    except Exception as e:
        console.print(f"  [yellow]Stock data unavailable: {e}[/yellow]")


# ═══════════════════════════════════════════════════════════════════════
# Section 3: CVNA Loan Book & Balance Sheet
# ═══════════════════════════════════════════════════════════════════════

def _fmp_get(settings, endpoint: str, params: dict | None = None) -> list | dict | None:
    """Helper to call FMP API. Returns parsed JSON or None."""
    import requests as req

    api_key = getattr(settings, "fmp_api_key", None)
    if not api_key:
        return None
    p = {"apikey": api_key}
    if params:
        p.update(params)
    try:
        resp = req.get(
            f"https://financialmodelingprep.com/api/v3/{endpoint}",
            params=p,
            timeout=15,
        )
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return None


def _fmt_b(val) -> str:
    """Format a number as $XB or $XM."""
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if abs(v) >= 1e9:
            return f"${v / 1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"${v / 1e6:.0f}M"
        return f"${v:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _pct_chg(current, prior) -> str:
    """YoY % change string."""
    try:
        c, p = float(current), float(prior)
        if p == 0:
            return "N/A"
        chg = (c - p) / abs(p) * 100
        color = "red" if chg > 0 else "green" if chg < 0 else "yellow"
        return f"[{color}]{chg:+.1f}%[/{color}]"
    except (TypeError, ValueError):
        return "N/A"


def _show_loan_book(console: Console, settings) -> None:
    console.print()
    console.print("[bold cyan]CVNA Loan Book & Balance Sheet[/bold cyan]")

    api_key = getattr(settings, "fmp_api_key", None)
    if not api_key:
        console.print("  [yellow]FMP_API_KEY not configured — skipping.[/yellow]")
        return

    bs_data = _fmp_get(settings, "balance-sheet-statement/CVNA", {"limit": "5", "period": "quarter"})
    is_data = _fmp_get(settings, "income-statement/CVNA", {"limit": "5", "period": "quarter"})
    cf_data = _fmp_get(settings, "cash-flow-statement/CVNA", {"limit": "5", "period": "quarter"})

    if not bs_data or not isinstance(bs_data, list) or len(bs_data) < 2:
        console.print("  [yellow]Financial data unavailable.[/yellow]")
        return

    latest = bs_data[0]
    prior_q = bs_data[1]
    prior_y = bs_data[4] if len(bs_data) >= 5 else bs_data[-1]

    table = Table(
        title="[bold]Balance Sheet Snapshot[/bold]",
        show_header=True, box=None, padding=(0, 2),
    )
    table.add_column("Item", style="bold")
    table.add_column("Latest Q", justify="right")
    table.add_column("Prior Q", justify="right", style="dim")
    table.add_column("YoY Q", justify="right")
    table.add_column("Signal")

    rows = [
        ("Finance Receivables", "netReceivables", "Loans on book — up = more risk retained"),
        ("Total Assets", "totalAssets", ""),
        ("Cash & Equivalents", "cashAndCashEquivalents", "Liquidity buffer"),
        ("Total Debt", "totalDebt", "Leverage — up = more pressure"),
        ("Stockholders Equity", "totalStockholdersEquity", "Negative = technically insolvent"),
    ]

    for label, key, signal_hint in rows:
        val_now = latest.get(key)
        val_pq = prior_q.get(key)
        val_py = prior_y.get(key)
        yoy = _pct_chg(val_now, val_py)

        signal = ""
        if signal_hint:
            signal = f"[dim]{signal_hint}[/dim]"
        try:
            if key == "totalStockholdersEquity" and val_now is not None and float(val_now) < 0:
                signal = "[red]NEGATIVE EQUITY[/red]"
            if key == "totalDebt" and val_now and val_py:
                if float(val_now) > float(val_py) * 1.10:
                    signal = "[red]Debt growing > 10% YoY[/red]"
            if key == "netReceivables" and val_now and val_py:
                if float(val_now) > float(val_py) * 1.15:
                    signal = "[red]Receivables up > 15% YoY — retaining more loans?[/red]"
                elif float(val_now) < float(val_py) * 0.85:
                    signal = "[green]Receivables shrinking — securitizing faster[/green]"
        except (TypeError, ValueError):
            pass

        table.add_row(label, _fmt_b(val_now), _fmt_b(val_pq), yoy, signal)

    console.print(table)

    # Leverage ratios
    try:
        debt = float(latest.get("totalDebt", 0) or 0)
        equity = float(latest.get("totalStockholdersEquity", 1) or 1)
        assets = float(latest.get("totalAssets", 1) or 1)
        cash = float(latest.get("cashAndCashEquivalents", 0) or 0)
        d_e = debt / equity if equity != 0 else float("inf")
        d_a = debt / assets if assets > 0 else 0
        net_debt = debt - cash

        de_color = "red" if d_e > 5 or d_e < 0 else "yellow" if d_e > 3 else "green"
        console.print(
            f"\n  [bold]Leverage:[/bold]  "
            f"Debt/Equity: [{de_color}]{d_e:.1f}x[/{de_color}]  |  "
            f"Debt/Assets: {d_a:.1%}  |  "
            f"Net Debt: {_fmt_b(net_debt)}"
        )
    except Exception:
        pass

    # Revenue / income context
    if is_data and isinstance(is_data, list) and len(is_data) >= 2:
        is_latest = is_data[0]
        is_prior_y = is_data[4] if len(is_data) >= 5 else is_data[-1]
        console.print()

        rev_table = Table(
            title="[bold]Income (Quarterly)[/bold]",
            show_header=True, box=None, padding=(0, 2),
        )
        rev_table.add_column("Item", style="bold")
        rev_table.add_column("Latest Q", justify="right")
        rev_table.add_column("YoY Q", justify="right")

        for label, key in [
            ("Revenue", "revenue"),
            ("Gross Profit", "grossProfit"),
            ("Net Income", "netIncome"),
            ("Interest Expense", "interestExpense"),
        ]:
            val = is_latest.get(key)
            val_py = is_prior_y.get(key)
            rev_table.add_row(label, _fmt_b(val), _pct_chg(val, val_py))

        console.print(rev_table)

        try:
            rev = float(is_latest.get("revenue", 0) or 0)
            gp = float(is_latest.get("grossProfit", 0) or 0)
            ni = float(is_latest.get("netIncome", 0) or 0)
            if rev > 0:
                gm = gp / rev * 100
                nm = ni / rev * 100
                nm_color = "green" if nm > 0 else "red"
                console.print(
                    f"  [bold]Margins:[/bold]  "
                    f"Gross: {gm:.1f}%  |  "
                    f"Net: [{nm_color}]{nm:.1f}%[/{nm_color}]"
                )
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Credit Quality & Auto Loan Stress
# ═══════════════════════════════════════════════════════════════════════

def _show_credit_quality(console: Console, settings) -> None:
    console.print()
    console.print("[bold cyan]Auto Loan Credit Quality[/bold cyan]")

    fred_key = getattr(settings, "fred_api_key", None)
    if not fred_key:
        console.print("  [yellow]FRED_API_KEY not configured — skipping.[/yellow]")
        return

    from lox.data.fred import FredClient

    client = FredClient(api_key=fred_key)

    series_defs = [
        ("DRALACBN", "Consumer Loan Delinquency Rate", "%", "All banks — CVNA originates in this pool"),
        ("DRCCLACBS", "Credit Card Delinquency Rate", "%", "Leading indicator — cards stress before auto"),
        ("SUBLPDCLATRNQ", "Banks Tightening Subprime Auto Stds", " net%", "Positive = banks pulling back on subprime"),
        ("DRTSCLCC", "SLOOS: Tightening C&I Loan Standards", " net%", "Broad credit tightening — spills into ABS market"),
    ]

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Indicator", style="bold", max_width=38)
    table.add_column("Latest", justify="right")
    table.add_column("1Y Ago", justify="right", style="dim")
    table.add_column("YoY Chg", justify="right")
    table.add_column("Thesis Impact")

    for sid, label, unit, thesis_note in series_defs:
        try:
            df = client.fetch_series(sid, start_date="2018-01-01")
            if df.empty:
                table.add_row(label, "N/A", "", "", "")
                continue

            latest_val = float(df["value"].iloc[-1])
            latest_date = df["date"].iloc[-1]

            yoy_target = latest_date - timedelta(days=365)
            prior = df[df["date"] <= yoy_target]
            prior_val = float(prior["value"].iloc[-1]) if not prior.empty else None

            if prior_val is not None:
                yoy_chg = latest_val - prior_val
                if yoy_chg > 0.1:
                    direction = "[red]Rising[/red]"
                    impact = f"[red]Bearish for CVNA[/red]"
                elif yoy_chg < -0.1:
                    direction = "[green]Falling[/green]"
                    impact = f"[green]Easing[/green]"
                else:
                    direction = "[yellow]Flat[/yellow]"
                    impact = "[yellow]Neutral[/yellow]"
                yoy_str = f"{yoy_chg:+.2f}"
                prior_str = f"{prior_val:.2f}{unit}"
            else:
                yoy_str = "—"
                prior_str = "—"
                direction = ""
                impact = ""

            table.add_row(
                label,
                f"{latest_val:.2f}{unit}",
                prior_str,
                f"{yoy_str} {direction}",
                impact,
            )
        except Exception as e:
            table.add_row(label, "[red]err[/red]", "", "", str(e)[:30])

    console.print(table)
    console.print(f"\n  [dim]Note: CVNA doesn't report loan-level delinquency publicly outside "
                  f"10-Q/ABS filings. These are market-wide proxies.[/dim]")

    # Historical context
    try:
        df_del = client.fetch_series("DRALACBN", start_date="2018-01-01")
        if not df_del.empty:
            current = float(df_del["value"].iloc[-1])
            pre_covid = df_del[df_del["date"] < "2020-03-01"]
            pre_covid_avg = float(pre_covid["value"].mean()) if not pre_covid.empty else None
            peak = float(df_del["value"].max())
            trough = float(df_del["value"].min())
            if pre_covid_avg:
                vs_pre = current - pre_covid_avg
                vs_color = "red" if vs_pre > 0 else "green"
                console.print(
                    f"  [bold]Consumer delinquency context:[/bold]  "
                    f"Current: {current:.2f}%  |  "
                    f"Pre-COVID avg: {pre_covid_avg:.2f}%  "
                    f"([{vs_color}]{vs_pre:+.2f}pp[/{vs_color}])  |  "
                    f"Range: {trough:.2f}%–{peak:.2f}%"
                )
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════
# Section 5: ABS Counterparty Exposure
# ═══════════════════════════════════════════════════════════════════════

def _show_abs_exposure(console: Console, settings) -> None:
    console.print()
    console.print("[bold cyan]ABS Counterparty Exposure[/bold cyan]")

    # Known CVNA securitization structure (from public filings)
    structure = Text()
    structure.append("Securitization Vehicle: ", style="bold")
    structure.append("Carvana Auto Receivables Trust (CRVNA)\n")
    structure.append("Deal Structure: ", style="bold")
    structure.append(
        "CVNA originates auto loans → pools into CRVNA trusts → issues "
        "ABS tranches (A/B/C/D rated) → retains residual/subordinate interests\n"
    )
    structure.append("Key Risk: ", style="bold red")
    structure.append(
        "CVNA retains \"beneficial interests\" (subordinate tranches + residuals). "
        "If loan losses exceed subordination levels, CVNA eats the first losses. "
        "If spreads widen, new issuance becomes more expensive or freezes.\n"
    )
    console.print(Panel(structure, title="[bold]ABS Structure[/bold]", border_style="yellow"))

    # Known warehouse lenders & underwriters
    console.print()
    known_table = Table(
        title="[bold]Known Warehouse Lenders & ABS Underwriters[/bold]",
        show_header=True, box=None, padding=(0, 2),
    )
    known_table.add_column("Counterparty", style="bold")
    known_table.add_column("Role", width=20)
    known_table.add_column("Exposure Type")
    known_table.add_column("Risk If CVNA ABS Deteriorates")

    counterparties = [
        ("Ally Financial", "Warehouse Lender", "Credit facility for loan inventory",
         "[red]Direct loss on warehouse line; forced to tighten[/red]"),
        ("JP Morgan Chase", "ABS Underwriter", "Lead bookrunner on CRVNA deals",
         "[yellow]Reputation risk; may retain unsold tranches[/yellow]"),
        ("Citigroup", "ABS Underwriter", "Co-manager on CRVNA deals",
         "[yellow]Distribution risk on unsold bonds[/yellow]"),
        ("Barclays", "ABS Underwriter", "Co-manager on CRVNA deals",
         "[yellow]Distribution risk[/yellow]"),
        ("Deutsche Bank", "ABS Underwriter", "Historical co-manager",
         "[yellow]Distribution risk[/yellow]"),
        ("Various CLO/Fund Mgrs", "ABS Investors", "Hold CRVNA A/B/C/D tranches",
         "[red]Mark-to-market losses if spreads widen[/red]"),
    ]

    for name, role, exposure, risk in counterparties:
        known_table.add_row(name, role, exposure, risk)

    console.print(known_table)

    # Pull institutional holders from FMP for additional context
    api_key = getattr(settings, "fmp_api_key", None)
    if api_key:
        try:
            import requests as req

            resp = req.get(
                "https://financialmodelingprep.com/api/v3/institutional-holder/CVNA",
                params={"apikey": api_key},
                timeout=15,
            )
            if resp.ok:
                holders = resp.json()
                if holders and isinstance(holders, list):
                    console.print()
                    h_table = Table(
                        title="[bold]Top Institutional Holders (Equity)[/bold]",
                        show_header=True, box=None, padding=(0, 2),
                    )
                    h_table.add_column("Institution", style="bold", max_width=40)
                    h_table.add_column("Shares", justify="right")
                    h_table.add_column("Value", justify="right")
                    h_table.add_column("% Out", justify="right")
                    h_table.add_column("ABS Overlap?", style="dim")

                    abs_players = {
                        "vanguard": "Index — passive",
                        "blackrock": "Index — passive",
                        "state street": "Index — passive",
                        "citadel": "Likely ABS arb desk",
                        "millennium": "Multi-strat — may trade ABS",
                        "jpmorgan": "ABS underwriter + equity",
                        "jp morgan": "ABS underwriter + equity",
                        "citigroup": "ABS underwriter + equity",
                        "ally": "Warehouse lender + equity",
                        "goldman": "Possible ABS market maker",
                        "morgan stanley": "Possible ABS market maker",
                        "two sigma": "Quant — may trade ABS",
                        "d.e. shaw": "Multi-strat",
                        "susquehanna": "Options/equity MM",
                    }

                    for h in holders[:15]:
                        name = h.get("holder", "Unknown")
                        shares = h.get("shares", 0)
                        value = h.get("value", 0)
                        pct = h.get("weightPercentage", 0)

                        overlap = ""
                        name_lower = name.lower()
                        for key, note in abs_players.items():
                            if key in name_lower:
                                overlap = note
                                break

                        h_table.add_row(
                            name[:40],
                            f"{shares:,.0f}" if shares else "N/A",
                            _fmt_b(value),
                            f"{pct:.2f}%" if pct else "N/A",
                            overlap,
                        )

                    console.print(h_table)
                    console.print(
                        "\n  [dim]Note: Equity holders != ABS holders. But major banks "
                        "with underwriting roles often hold both. True ABS exposure "
                        "is in CRVNA trust 10-D filings.[/dim]"
                    )
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Section 6: ABS Trust Filings (EDGAR)
# ═══════════════════════════════════════════════════════════════════════

def _show_abs_trust_filings(console: Console) -> None:
    console.print()
    console.print("[bold cyan]CVNA ABS Trust Filings (EDGAR)[/bold cyan]")

    try:
        import requests as req

        resp = req.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q": '"carvana auto receivables"',
                "forms": "10-D,ABS-EE",
                "dateRange": "custom",
                "startdt": "2024-01-01",
            },
            headers={"User-Agent": "LoxCapital/1.0 (research@loxcapital.com)"},
            timeout=15,
        )

        if not resp.ok:
            console.print("  [yellow]EDGAR search unavailable — try SEC EDGAR directly.[/yellow]")
            _show_abs_trust_fallback(console)
            return

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])

        if not hits:
            console.print("  [dim]No recent ABS trust filings found via EDGAR search.[/dim]")
            _show_abs_trust_fallback(console)
            return

        import re

        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Trust", style="bold", max_width=45)
        table.add_column("Form", width=8)
        table.add_column("Filed", width=12)
        table.add_column("Period Ending", width=14, style="dim")

        seen = set()
        rows_added = 0
        for hit in hits:
            if rows_added >= 15:
                break
            src = hit.get("_source", {})
            names = src.get("display_names", [])
            raw_name = names[0] if names else "Unknown"
            entity_name = re.sub(r"\s*\(CIK\s*\d+\)", "", raw_name).strip()
            form_type = src.get("form", "") or src.get("file_type", "")
            file_date = src.get("file_date", "")
            period = src.get("period_ending", "")

            dedup_key = f"{entity_name}|{form_type}|{file_date}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            table.add_row(entity_name[:45], form_type, file_date, period)
            rows_added += 1

        console.print(table)

        total = data.get("hits", {}).get("total", {}).get("value", 0)
        if total > rows_added:
            console.print(f"  [dim]{total:,} total ABS trust filings on EDGAR since 2024.[/dim]")

    except Exception as e:
        console.print(f"  [yellow]EDGAR search failed: {e}[/yellow]")
        _show_abs_trust_fallback(console)


def _show_abs_trust_fallback(console: Console) -> None:
    """Show known CRVNA trust deals when EDGAR search is unavailable."""
    console.print()
    console.print("  [bold]Known Recent CRVNA Trusts:[/bold]")
    trusts = [
        "CRVNA 2024-N1  |  CRVNA 2024-N2  |  CRVNA 2024-N3",
        "CRVNA 2023-N1  |  CRVNA 2023-N2  |  CRVNA 2023-N3  |  CRVNA 2023-N4",
        "CRVNA 2023-P1  |  CRVNA 2023-P2  |  CRVNA 2023-P3",
    ]
    for line in trusts:
        console.print(f"    {line}")
    console.print(
        "\n  [dim]Search EDGAR for 'Carvana Auto Receivables Trust' to find "
        "10-D (distribution reports) and ABS-EE (asset-level data).[/dim]"
    )
    console.print(
        "  [dim]URL: https://www.sec.gov/cgi-bin/browse-edgar?"
        "company=carvana+auto+receivables&type=10-D&action=getcompany[/dim]"
    )


# ═══════════════════════════════════════════════════════════════════════
# Section 7: CVNA Corporate SEC Filings
# ═══════════════════════════════════════════════════════════════════════

def _show_filings(console: Console, settings) -> None:
    console.print()
    console.print("[bold cyan]CVNA Corporate SEC Filings[/bold cyan]")

    try:
        from lox.altdata.sec import fetch_sec_filings

        filings = fetch_sec_filings(
            settings=settings,
            ticker="CVNA",
            form_types=["8-K", "10-K", "10-Q"],
            limit=10,
        )

        if not filings:
            console.print("  [dim]No filings found.[/dim]")
            return

        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Form", style="bold", width=6)
        table.add_column("Filed", width=12)
        table.add_column("Description", max_width=55)
        table.add_column("Items", style="dim", max_width=20)

        for f in filings:
            form_style = "[red]" if f.form_type.startswith("8-K") else "[cyan]"
            form_end = "[/red]" if f.form_type.startswith("8-K") else "[/cyan]"
            items_str = ", ".join(f.items) if f.items else ""
            desc = (f.description or "")[:55]
            table.add_row(
                f"{form_style}{f.form_type}{form_end}",
                f.filed_date,
                desc,
                items_str,
            )

        console.print(table)

        if filings and filings[0].filing_url:
            console.print(
                f"\n  [dim]Latest: {filings[0].filing_url}[/dim]"
            )

    except Exception as e:
        console.print(f"  [yellow]SEC data unavailable: {e}[/yellow]")


# ═══════════════════════════════════════════════════════════════════════
# Section 8: Catalyst Timeline
# ═══════════════════════════════════════════════════════════════════════

def _show_catalysts(console: Console) -> None:
    console.print()
    console.print("[bold cyan]Catalyst Timeline[/bold cyan]")

    today = date.today()

    catalysts = [
        (date(2026, 2, 19), "CVNA Q4 2025 Earnings",
         "Loan origination volume, ABS residual, delinquency commentary, "
         "beneficial interest balance"),
        (date(2026, 3, 15), "FRED Q4 Delinquency Data",
         "Consumer loan delinquency update (DRALACBN) — watch for re-acceleration"),
        (date(2026, 4, 15), "Tax Day",
         "Refund season ends — subprime borrowers lose cash cushion, "
         "delinquencies historically tick up"),
        (date(2026, 5, 14), "CVNA Q1 2026 Earnings",
         "Post-refund season loan performance; warehouse line utilization"),
        (date(2026, 6, 15), "FRED Q1 Delinquency Data",
         "Quarterly delinquency + SLOOS update — credit tightening signal"),
        (date(2026, 8, 6), "CVNA Q2 2026 Earnings",
         "Summer selling season results; ABS issuance pace"),
    ]

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Date", style="bold", width=12)
    table.add_column("Days", justify="right", width=6)
    table.add_column("Event", width=28)
    table.add_column("Watch For", style="dim")

    for cat_date, event, watch in catalysts:
        days_until = (cat_date - today).days
        if days_until < -30:
            continue

        if days_until < 0:
            days_str = f"[dim]{days_until}d[/dim]"
            date_style = "dim"
        elif days_until <= 14:
            days_str = f"[red]{days_until}d[/red]"
            date_style = "red"
        elif days_until <= 45:
            days_str = f"[yellow]{days_until}d[/yellow]"
            date_style = "yellow"
        else:
            days_str = f"{days_until}d"
            date_style = ""

        date_str = (
            f"[{date_style}]{cat_date.strftime('%Y-%m-%d')}[/{date_style}]"
            if date_style
            else cat_date.strftime("%Y-%m-%d")
        )
        table.add_row(date_str, days_str, event, watch)

    console.print(table)
