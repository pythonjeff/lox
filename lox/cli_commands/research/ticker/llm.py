"""Ticker LLM analysis — build snapshot and hand off to regime chat."""
from __future__ import annotations

import logging

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from lox.cli_commands.research.ticker.compute import compute_flow_context, compute_refinancing_wall

logger = logging.getLogger(__name__)


def show_llm_analysis(console: Console, settings, symbol: str, price_data: dict, fundamentals: dict, technicals: dict):
    """Show LLM analysis."""
    snapshot = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Preparing analysis...[/bold cyan]"),
        transient=True,
    ) as progress:
        progress.add_task("llm", total=None)

        try:
            snapshot = {
                "symbol": symbol,
                "price": technicals.get("current"),
                "change_pct": price_data.get("quote", {}).get("changesPercentage"),
                "52w_high": technicals.get("high_52w"),
                "52w_low": technicals.get("low_52w"),
                "rsi": technicals.get("rsi"),
                "trend": technicals.get("trend"),
                "volatility": technicals.get("volatility"),
                "support": technicals.get("support"),
                "resistance": technicals.get("resistance"),
            }

            profile = fundamentals.get("profile", {})
            if profile:
                snapshot["sector"] = profile.get("sector")
                snapshot["industry"] = profile.get("industry")
                snapshot["mkt_cap"] = profile.get("mktCap")

            ratios = fundamentals.get("ratios", {})
            if ratios:
                snapshot["pe_ratio"] = ratios.get("peRatioTTM")
                snapshot["profit_margin"] = ratios.get("netProfitMarginTTM")

            etf_info = fundamentals.get("etf_info", {})
            is_etf = profile.get("isEtf", False) or bool(etf_info)
            if is_etf:
                snapshot["asset_type"] = "ETF"
                if etf_info:
                    snapshot["aum"] = etf_info.get("totalAssets")
                    snapshot["expense_ratio"] = etf_info.get("expenseRatio")
                    snapshot["etf_yield"] = etf_info.get("yield")
                    snapshot["holdings_count"] = etf_info.get("holdingsCount")
                    snapshot["asset_class"] = etf_info.get("assetClass")
                flow_ctx = compute_flow_context(price_data)
                if flow_ctx:
                    snapshot["fund_flows"] = flow_ctx
                refi_wall = compute_refinancing_wall(settings, symbol)
                if refi_wall:
                    snapshot["refinancing_wall"] = refi_wall

            try:
                from lox.llm.scenarios.quant_scenarios import compute_quant_scenarios
                from lox.regimes.features import build_unified_regime_state

                regime_state = build_unified_regime_state(settings=settings)
                quant = compute_quant_scenarios(
                    historical_prices=price_data.get("historical", []),
                    regime_state=regime_state,
                    current_price=technicals.get("current"),
                )
                snapshot["quant_scenarios"] = quant
            except Exception as exc:
                logger.debug("Quant scenarios unavailable: %s", exc)

        except Exception as e:
            console.print(f"\n[dim]Analysis unavailable: {e}[/dim]")
            return

    # Launch the interactive chat OUTSIDE the spinner so it doesn't
    # conflict with console input/streaming.
    if snapshot is not None:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="growth",
            snapshot=snapshot,
            regime_label=f"{symbol} Analysis",
            regime_description=f"Equity analysis for {symbol}",
            include_news=True,
            include_prices=False,
            include_calendar=True,
            ticker=symbol,
            console=console,
        )
