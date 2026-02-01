"""
CLI formatting utilities - centralized color/table/display helpers.

This module provides consistent formatting across all CLI commands.
"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def color_for_return(value: float) -> str:
    """Get color for a return value."""
    if value > 5:
        return "bold green"
    elif value > 1:
        return "green"
    elif value < -5:
        return "bold red"
    elif value < -1:
        return "red"
    return "white"


def color_for_risk(score: int) -> str:
    """Get color for a risk score (0-100)."""
    if score >= 80:
        return "bold red"
    elif score >= 60:
        return "red"
    elif score >= 40:
        return "yellow"
    elif score >= 20:
        return "green"
    return "bold green"


def color_for_pnl(value: float) -> str:
    """Get color for P&L."""
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    return "white"


def format_currency(value: float, decimals: int = 2, billions: bool = False) -> str:
    """Format a currency value."""
    if billions:
        return f"${value:.{decimals}f}B"
    return f"${value:,.{decimals}f}"


def format_pct(value: float, decimals: int = 1, show_sign: bool = False) -> str:
    """Format a percentage value."""
    if show_sign:
        return f"{value:+.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def create_metric_table(title: str, rows: list[tuple[str, str]], expand: bool = False) -> Table:
    """
    Create a simple two-column metric table.
    
    Args:
        title: Table title
        rows: List of (metric_name, value) tuples
        expand: Whether to expand the table to full width
    """
    table = Table(title=title, expand=expand, show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    
    for metric, value in rows:
        table.add_row(metric, value)
    
    return table


def print_stock_fundamentals(console: Console, f: Any) -> None:
    """
    Print stock fundamentals in a formatted table.
    
    Args:
        console: Rich console instance
        f: CompanyFundamentals dataclass with stock data
    """
    fund_table = Table(title=f"Fundamentals: {f.name}", expand=False, show_header=True)
    fund_table.add_column("Valuation", style="bold")
    fund_table.add_column("Value", justify="right")
    fund_table.add_column("Quality", style="bold")
    fund_table.add_column("Value", justify="right")
    
    fund_table.add_row(
        "Market Cap", f"${f.market_cap_b:.1f}B",
        "Gross Margin", f"{f.gross_margin*100:.1f}%",
    )
    fund_table.add_row(
        "P/E Ratio", f"{f.pe_ratio:.1f}" if f.pe_ratio > 0 else "N/A",
        "Operating Margin", f"{f.operating_margin*100:.1f}%",
    )
    fund_table.add_row(
        "Forward P/E", f"{f.forward_pe:.1f}" if f.forward_pe > 0 else "N/A",
        "Net Margin", f"{f.net_margin*100:.1f}%",
    )
    fund_table.add_row(
        "P/S Ratio", f"{f.ps_ratio:.1f}",
        "ROE", f"{f.roe*100:.1f}%",
    )
    fund_table.add_row(
        "P/B Ratio", f"{f.pb_ratio:.1f}",
        "ROA", f"{f.roa*100:.1f}%",
    )
    fund_table.add_row(
        "EV/EBITDA", f"{f.ev_ebitda:.1f}" if f.ev_ebitda > 0 else "N/A",
        "ROIC", f"{f.roic*100:.1f}%",
    )
    fund_table.add_row(
        "Revenue Growth", f"{f.revenue_growth_yoy*100:+.1f}%",
        "Debt/Equity", f"{f.debt_to_equity:.1f}",
    )
    fund_table.add_row(
        "EPS Growth", f"{f.earnings_growth_yoy*100:+.1f}%",
        "Current Ratio", f"{f.current_ratio:.1f}",
    )
    fund_table.add_row(
        "Dividend Yield", f"{f.dividend_yield*100:.2f}%",
        "Sector", f.sector[:20] if f.sector else "N/A",
    )
    
    console.print(fund_table)


def print_etf_fundamentals(console: Console, f: Any) -> None:
    """
    Print ETF fundamentals in a formatted panel and table.
    
    Args:
        console: Rich console instance
        f: ETFFundamentals dataclass with ETF data
    """
    etf_lines = [
        f"[bold]{f.name}[/bold]",
        "",
        f"[bold]Category:[/bold] {f.category}",
        f"[bold]AUM:[/bold] ${f.aum_b:.1f}B",
        f"[bold]Expense Ratio:[/bold] {f.expense_ratio:.2%}",
        f"[bold]Inception:[/bold] {f.inception_date}",
        f"[bold]Holdings:[/bold] {f.holdings_count}",
        f"[bold]Top 10 Weight:[/bold] {f.top_10_weight:.1f}%",
    ]
    
    console.print(Panel("\n".join(etf_lines), title="ETF Profile", expand=False))
    
    if f.top_holdings:
        hold_table = Table(title="Top Holdings", expand=False)
        hold_table.add_column("#", justify="right", style="dim")
        hold_table.add_column("Ticker", style="bold cyan")
        hold_table.add_column("Name")
        hold_table.add_column("Weight", justify="right")
        
        for i, h in enumerate(f.top_holdings[:10], 1):
            hold_table.add_row(
                str(i),
                h.get("ticker", ""),
                h.get("name", "")[:25],
                f"{h.get('weight_pct', 0):.1f}%",
            )
        
        console.print()
        console.print(hold_table)


def print_momentum_metrics(console: Console, m: Any) -> None:
    """
    Print momentum metrics in a formatted table.
    
    Args:
        console: Rich console instance
        m: MomentumMetrics dataclass
    """
    table = Table(title="Momentum Metrics", expand=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Interpretation", style="dim")
    
    # Returns
    ret_1d = m.return_1d * 100
    ret_5d = m.return_5d * 100
    ret_20d = m.return_20d * 100
    ret_60d = m.return_60d * 100
    
    table.add_row("1D Return", f"[{color_for_return(ret_1d)}]{ret_1d:+.2f}%[/{color_for_return(ret_1d)}]", "")
    table.add_row("5D Return", f"[{color_for_return(ret_5d)}]{ret_5d:+.2f}%[/{color_for_return(ret_5d)}]", "")
    table.add_row("20D Return", f"[{color_for_return(ret_20d)}]{ret_20d:+.2f}%[/{color_for_return(ret_20d)}]", "")
    table.add_row("60D Return", f"[{color_for_return(ret_60d)}]{ret_60d:+.2f}%[/{color_for_return(ret_60d)}]", "")
    table.add_row("", "", "")
    
    # RSI
    rsi_color = "red" if m.rsi_14 > 70 else "green" if m.rsi_14 < 30 else "white"
    rsi_desc = "Overbought" if m.rsi_14 > 70 else "Oversold" if m.rsi_14 < 30 else "Neutral"
    table.add_row("RSI (14)", f"[{rsi_color}]{m.rsi_14:.1f}[/{rsi_color}]", rsi_desc)
    
    # Trend
    trend_color = "green" if m.price_vs_sma_50 > 0 else "red"
    trend_desc = "Above 50-day MA" if m.price_vs_sma_50 > 0 else "Below 50-day MA"
    table.add_row("vs SMA50", f"[{trend_color}]{m.price_vs_sma_50*100:+.1f}%[/{trend_color}]", trend_desc)
    
    console.print(table)


def print_hf_metrics(console: Console, h: Any) -> None:
    """
    Print hedge fund grade metrics in a formatted table.
    
    Args:
        console: Rich console instance
        h: HedgeFundMetrics dataclass
    """
    table = Table(title="Hedge Fund Metrics", expand=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Grade", style="dim")
    
    # Volatility
    vol_grade = "High" if h.volatility_30d > 30 else "Medium" if h.volatility_30d > 15 else "Low"
    table.add_row("Volatility (30D)", f"{h.volatility_30d:.1f}%", vol_grade)
    table.add_row("Volatility (1Y)", f"{h.volatility_1y:.1f}%", "")
    
    # Beta
    beta_grade = "High" if h.beta_90d > 1.5 else "Medium" if h.beta_90d > 0.8 else "Defensive"
    table.add_row("Beta (90D)", f"{h.beta_90d:.2f}", beta_grade)
    
    # Risk-adjusted returns
    sharpe_color = "green" if h.sharpe_ratio > 1 else "yellow" if h.sharpe_ratio > 0 else "red"
    table.add_row("Sharpe Ratio", f"[{sharpe_color}]{h.sharpe_ratio:.2f}[/{sharpe_color}]", "")
    table.add_row("Sortino Ratio", f"{h.sortino_ratio:.2f}", "")
    
    # Drawdown
    dd_color = "red" if h.max_drawdown < -20 else "yellow" if h.max_drawdown < -10 else "green"
    table.add_row("Max Drawdown", f"[{dd_color}]{h.max_drawdown:.1f}%[/{dd_color}]", "")
    
    # Correlation
    table.add_row("Corr (SPY)", f"{h.correlation_spy:.2f}", "")
    table.add_row("Corr (QQQ)", f"{h.correlation_qqq:.2f}", "")
    
    # Scores
    risk_color = color_for_risk(h.risk_score)
    table.add_row("Risk Score", f"[{risk_color}]{h.risk_score}/100[/{risk_color}]", "")
    
    console.print(table)
