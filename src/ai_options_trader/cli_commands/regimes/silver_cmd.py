"""
Silver / SLV regime CLI command.
"""

from __future__ import annotations

import typer
from rich import print
from rich.panel import Panel


def _create_bar(value: float, max_val: float = 100, width: int = 30, 
                colors: list = None, thresholds: list = None) -> str:
    """
    Create a visual progress bar with color zones.
    
    Args:
        value: Current value
        max_val: Maximum value for the bar
        width: Character width of the bar
        colors: List of colors for each zone
        thresholds: List of threshold percentages for color zones
    """
    if colors is None:
        colors = ["green", "yellow", "orange1", "red", "bright_red"]
    if thresholds is None:
        thresholds = [20, 40, 60, 80, 100]
    
    pct = min(100, max(0, (value / max_val) * 100))
    filled = int((pct / 100) * width)
    empty = width - filled
    
    # Determine color based on value
    color = colors[0]
    for i, thresh in enumerate(thresholds):
        if pct <= thresh:
            color = colors[i]
            break
    else:
        color = colors[-1]
    
    # Build the bar with gradient effect
    bar_chars = ""
    for i in range(filled):
        char_pct = (i / width) * 100
        char_color = colors[0]
        for j, thresh in enumerate(thresholds):
            if char_pct <= thresh:
                char_color = colors[j]
                break
        else:
            char_color = colors[-1]
        bar_chars += f"[{char_color}]█[/{char_color}]"
    
    bar_chars += f"[dim]{'░' * empty}[/dim]"
    
    return bar_chars


def _create_gauge(value: float, label: str, width: int = 40) -> str:
    """Create a labeled gauge with the value indicator."""
    pct = min(100, max(0, value))
    
    # Color based on intensity
    if pct < 25:
        color = "green"
        status = "Normal"
    elif pct < 50:
        color = "yellow"
        status = "Elevated"
    elif pct < 75:
        color = "orange1"
        status = "High"
    else:
        color = "red"
        status = "Extreme"
    
    bar = _create_bar(pct, 100, width)
    
    return f"{label}: [{color}]{pct:.0f}%[/{color}] ({status})\n{bar}"


def _show_bubble_tracker(inp, console) -> None:
    """Display the visual bubble tracker."""
    from rich.table import Table
    from rich.text import Text
    
    bubble_score = inp.bubble_score or 0
    mr_pressure = inp.mean_reversion_pressure or 0
    extension = inp.extension_pct or 0
    days_extreme = inp.days_at_extreme or 0
    
    # Determine overall bubble status
    if bubble_score >= 80:
        bubble_status = "🔴 EXTREME BUBBLE"
        bubble_color = "bright_red"
        bubble_emoji = "🫧🫧🫧🫧🫧"
    elif bubble_score >= 60:
        bubble_status = "🟠 HIGH BUBBLE RISK"
        bubble_color = "orange1"
        bubble_emoji = "🫧🫧🫧🫧"
    elif bubble_score >= 40:
        bubble_status = "🟡 ELEVATED"
        bubble_color = "yellow"
        bubble_emoji = "🫧🫧🫧"
    elif bubble_score >= 20:
        bubble_status = "🟢 MILD EXTENSION"
        bubble_color = "green"
        bubble_emoji = "🫧🫧"
    else:
        bubble_status = "⚪ NORMAL"
        bubble_color = "dim"
        bubble_emoji = "🫧"
    
    # Build the visual output
    lines = []
    
    # Header with big status
    lines.append(f"[bold {bubble_color}]{bubble_status}[/bold {bubble_color}]  {bubble_emoji}")
    lines.append("")
    
    # Main bubble gauge
    lines.append("[bold]BUBBLE INTENSITY[/bold]")
    lines.append(_create_bar(bubble_score, 100, 50))
    lines.append(f"[dim]0%{'─' * 15}25%{'─' * 14}50%{'─' * 14}75%{'─' * 13}100%[/dim]")
    lines.append(f"              Normal      Elevated       High        [red]Extreme[/red]")
    lines.append("")
    
    # Mean reversion pressure gauge
    lines.append("[bold]MEAN REVERSION PRESSURE[/bold]")
    mr_bar = _create_bar(mr_pressure, 100, 50, 
                         colors=["dim", "cyan", "blue", "magenta", "bright_magenta"],
                         thresholds=[20, 40, 60, 80, 100])
    lines.append(mr_bar)
    
    if mr_pressure >= 70:
        lines.append(f"[bright_magenta]⚠ HIGH reversion pressure - snap-back risk elevated[/bright_magenta]")
    elif mr_pressure >= 50:
        lines.append(f"[magenta]Mean reversion building - watch for momentum shift[/magenta]")
    else:
        lines.append(f"[dim]Reversion pressure moderate[/dim]")
    lines.append("")
    
    # Extension from fair value
    ext_color = "red" if extension > 50 else "orange1" if extension > 25 else "yellow" if extension > 10 else "green"
    lines.append(f"[bold]EXTENSION FROM FAIR VALUE (200MA):[/bold] [{ext_color}]+{extension:.1f}%[/{ext_color}]")
    
    # Visual extension bar (centered at 0)
    ext_bar_width = 50
    ext_normalized = min(100, extension)  # Cap at 100% for display
    ext_filled = int((ext_normalized / 100) * ext_bar_width)
    lines.append(f"[dim]Fair Value[/dim] │{'█' * ext_filled}[dim]{'░' * (ext_bar_width - ext_filled)}[/dim]│ [dim]+100%[/dim]")
    lines.append("")
    
    # Days at extreme
    if days_extreme > 0:
        urgency = "🔥" * min(5, days_extreme // 5 + 1)
        lines.append(f"[bold]DAYS AT BUBBLE (>50):[/bold] [yellow]{days_extreme}[/yellow] {urgency}")
        if days_extreme > 20:
            lines.append(f"[red]Extended duration increases snap-back severity[/red]")
        elif days_extreme > 10:
            lines.append(f"[orange1]Prolonged extreme - reversion typically follows[/orange1]")
    else:
        lines.append(f"[bold]DAYS AT BUBBLE (>50):[/bold] [green]0[/green]")
    lines.append("")
    
    # Component breakdown
    lines.append("[bold]BUBBLE COMPONENTS:[/bold]")
    
    components = [
        ("Return Z-score", inp.slv_zscore_20d or 0, 4, "Momentum extremity"),
        ("Volatility Z", inp.slv_vol_zscore or 0, 4, "Market stress"),
        ("Dist from 200MA", extension / 5, 20, "Price extension"),  # Scale to 0-20
        ("GSR extremity", abs(inp.gsr_zscore or 0), 4, "vs Gold relative"),
    ]
    
    for name, val, max_v, desc in components:
        contrib = min(20, (abs(val) / max_v) * 20)
        mini_bar = _create_bar(contrib, 20, 15)
        lines.append(f"  {name:15} {mini_bar} [dim]{desc}[/dim]")
    
    lines.append("")
    
    # Mean reversion targets
    lines.append("[bold]MEAN REVERSION TARGETS:[/bold]")
    if inp.slv_price and inp.slv_ma_200:
        target_50 = inp.slv_ma_200 * 1.5  # 50% premium to 200MA
        target_25 = inp.slv_ma_200 * 1.25  # 25% premium
        target_fair = inp.slv_ma_200  # Fair value
        
        lines.append(f"  [green]Fair value (200MA):[/green] ${target_fair:.2f} ([red]{((inp.slv_price / target_fair) - 1) * 100:+.0f}% away[/red])")
        lines.append(f"  [yellow]25% premium:[/yellow] ${target_25:.2f} ([orange1]{((inp.slv_price / target_25) - 1) * 100:+.0f}% away[/orange1])")
        lines.append(f"  [orange1]50% premium:[/orange1] ${target_50:.2f} ({((inp.slv_price / target_50) - 1) * 100:+.0f}% away)")
    
    print(Panel("\n".join(lines), title="🫧 SLV Bubble Tracker 🫧", expand=False))


def _show_reversion_forecast(inp, forecast, console) -> None:
    """Display the reversion forecast with timing and severity estimates."""
    from rich.table import Table
    
    lines = []
    
    # Confidence indicator
    conf = forecast.forecast_confidence or 0
    if conf >= 70:
        conf_color = "green"
        conf_label = "HIGH"
    elif conf >= 50:
        conf_color = "yellow"
        conf_label = "MODERATE"
    else:
        conf_color = "red"
        conf_label = "LOW"
    
    lines.append(f"[bold]Forecast Confidence:[/bold] [{conf_color}]{conf:.0f}% ({conf_label})[/{conf_color}]")
    lines.append(f"[dim]Based on {len(forecast.analog_outcomes)} historical analogs[/dim]")
    lines.append("")
    
    # Leading indicators
    lines.append("[bold underline]LEADING INDICATORS[/bold underline]")
    
    # RSI
    rsi = inp.rsi_14 or 50
    rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "yellow"
    rsi_status = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
    lines.append(f"  RSI (14): [{rsi_color}]{rsi:.1f}[/{rsi_color}] ({rsi_status})")
    
    # Momentum divergence
    mom_div = inp.momentum_divergence
    if mom_div:
        lines.append(f"  Momentum Divergence: [green]✓ DETECTED[/green] (bearish signal)")
    else:
        lines.append(f"  Momentum Divergence: [dim]Not detected[/dim]")
    
    # Volume exhaustion
    vol_ex = inp.volume_exhaustion
    if vol_ex:
        lines.append(f"  Volume Exhaustion: [green]✓ DETECTED[/green] (momentum fading)")
    else:
        lines.append(f"  Volume Exhaustion: [dim]Not detected[/dim]")
    
    lines.append("")
    
    # Timing estimates
    lines.append("[bold underline]TIMING ESTIMATES[/bold underline]")
    lines.append(f"  Days to significant reversion (20%+ drop):")
    
    # Visual timeline
    days_low = forecast.days_to_reversion_low or 10
    days_mid = forecast.days_to_reversion_mid or 30
    days_high = forecast.days_to_reversion_high or 60
    
    timeline_width = 60
    
    # Create timeline with markers
    def pos(days, max_days=90):
        return int((min(days, max_days) / max_days) * timeline_width)
    
    timeline = list("─" * timeline_width)
    timeline[pos(days_low)] = "[green]◄[/green]"
    timeline[pos(days_mid)] = "[yellow]●[/yellow]"
    timeline[pos(days_high)] = "[red]►[/red]"
    
    # Add month markers
    timeline[pos(30)] = "│" if timeline[pos(30)] == "─" else timeline[pos(30)]
    timeline[pos(60)] = "│" if timeline[pos(60)] == "─" else timeline[pos(60)]
    
    lines.append(f"  Today {''.join(timeline)} 90d")
    lines.append(f"        [green]◄ Best: {days_low}d[/green]   [yellow]● Likely: {days_mid}d[/yellow]   [red]► Worst: {days_high}d[/red]")
    lines.append("")
    
    # Probability table
    lines.append(f"  [bold]Reversion Probability:[/bold]")
    prob_30 = forecast.reversion_probability_30d or 0
    prob_60 = forecast.reversion_probability_60d or 0
    prob_90 = forecast.reversion_probability_90d or 0
    
    # Visual probability bars
    def prob_bar(pct, width=20):
        filled = int((pct / 100) * width)
        if pct >= 70:
            color = "green"
        elif pct >= 40:
            color = "yellow"
        else:
            color = "red"
        return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim] {pct:.0f}%"
    
    lines.append(f"    30 days: {prob_bar(prob_30)}")
    lines.append(f"    60 days: {prob_bar(prob_60)}")
    lines.append(f"    90 days: {prob_bar(prob_90)}")
    lines.append("")
    
    # March vs September put analysis
    today = inp.slv_price or 0
    if today > 0:
        lines.append(f"  [bold cyan]YOUR PUTS:[/bold cyan]")
        # Assuming March is ~45 days out, September ~225 days
        lines.append(f"    March expiry (~45d):  {prob_bar(min(prob_60, prob_30 + 15), 15)} chance of being ITM")
        lines.append(f"    September (~225d): {prob_bar(min(99, prob_90 + 5), 15)} chance of being ITM")
    lines.append("")
    
    # Severity estimates
    lines.append("[bold underline]SEVERITY ESTIMATES[/bold underline]")
    lines.append(f"  Expected drawdown from current price (${inp.slv_price:.2f}):")
    
    dd_10 = forecast.drawdown_10th_pct or -20
    dd_50 = forecast.drawdown_median or -40
    dd_90 = forecast.drawdown_90th_pct or -60
    
    # Price targets
    if inp.slv_price:
        price_mild = inp.slv_price * (1 + dd_10/100)
        price_expected = inp.slv_price * (1 + dd_50/100)
        price_severe = inp.slv_price * (1 + dd_90/100)
        
        lines.append(f"")
        lines.append(f"  [green]Mild (10th %ile):[/green]     {dd_10:+.0f}%  →  ${price_mild:.2f}")
        lines.append(f"  [yellow]Expected (median):[/yellow]   {dd_50:+.0f}%  →  ${price_expected:.2f}")
        lines.append(f"  [red]Severe (90th %ile):[/red]   {dd_90:+.0f}%  →  ${price_severe:.2f}")
    lines.append("")
    
    # Severity visualization
    current = 0
    mild_pos = int(abs(dd_10) / 2)
    expected_pos = int(abs(dd_50) / 2)
    severe_pos = int(abs(dd_90) / 2)
    
    severity_bar = list(" " * 55)
    severity_bar[0] = "│"
    severity_bar[mild_pos] = "[green]▼[/green]"
    severity_bar[expected_pos] = "[yellow]▼[/yellow]"
    severity_bar[min(severe_pos, 54)] = "[red]▼[/red]"
    
    lines.append(f"  $0 {''.join(severity_bar)}")
    lines.append(f"  Current: ${inp.slv_price:.2f}")
    lines.append("")
    
    # Key trigger levels
    lines.append("[bold underline]TRIGGER WATCHLIST[/bold underline]")
    lines.append(f"  Watch for these signals to confirm reversion is starting:")
    lines.append("")
    
    if forecast.trigger_50ma:
        pct_away = ((inp.slv_price or 0) / forecast.trigger_50ma - 1) * 100 if inp.slv_price else 0
        lines.append(f"  ⚡ Break below 50-day MA: ${forecast.trigger_50ma:.2f} ({pct_away:+.0f}% away)")
    
    if forecast.trigger_gsr_expansion:
        current_gsr = inp.gsr or 0
        lines.append(f"  ⚡ GSR expansion above: {forecast.trigger_gsr_expansion:.1f} (currently {current_gsr:.1f})")
    
    lines.append(f"  ⚡ Momentum divergence confirmation: {'[green]Active[/green]' if inp.momentum_divergence else '[dim]Waiting[/dim]'}")
    lines.append(f"  ⚡ Volume exhaustion: {'[green]Active[/green]' if inp.volume_exhaustion else '[dim]Waiting[/dim]'}")
    lines.append("")
    
    # Historical analogs
    if forecast.analog_outcomes:
        lines.append("[bold underline]HISTORICAL ANALOGS[/bold underline]")
        lines.append(f"  Similar bubble peaks and what happened after:")
        lines.append("")
        
        for i, analog in enumerate(forecast.analog_outcomes[:5], 1):
            date_str = analog.get("date", "Unknown")
            peak_price = analog.get("peak_price", 0)
            days_to_rev = analog.get("days_to_reversion", "N/A")
            max_dd = analog.get("max_drawdown_90d", 0)
            dd_30 = analog.get("drawdown_30d", 0)
            similarity = analog.get("similarity", 0)
            
            # Color code by similarity
            if similarity >= 0.8:
                sim_color = "green"
            elif similarity >= 0.6:
                sim_color = "yellow"
            else:
                sim_color = "dim"
            
            lines.append(f"  [{sim_color}]{i}. {date_str}[/{sim_color}] (similarity: {similarity:.0%})")
            lines.append(f"     Peak: ${peak_price:.2f} → Days to -20%: {days_to_rev if days_to_rev else '>90'}")
            lines.append(f"     30d return: {dd_30:+.1f}%  |  Max drawdown: {max_dd:.1f}%")
            lines.append("")
    
    # Summary recommendation
    lines.append("[bold underline]SUMMARY[/bold underline]")
    
    if prob_60 >= 70 and (inp.momentum_divergence or inp.volume_exhaustion):
        lines.append(f"  [green bold]⚠ HIGH PROBABILITY reversion within 60 days[/green bold]")
        lines.append(f"  Leading indicators are active. Your September puts look well-positioned.")
        lines.append(f"  March puts are a coin flip - consider rolling if deep OTM.")
    elif prob_60 >= 50:
        lines.append(f"  [yellow bold]MODERATE PROBABILITY reversion within 60 days[/yellow bold]")
        lines.append(f"  Bubble is extreme but no leading indicator confirmation yet.")
        lines.append(f"  Watch for momentum divergence or 50-day MA break.")
    else:
        lines.append(f"  [red bold]TIMING UNCERTAIN - Bubble can extend further[/red bold]")
        lines.append(f"  Despite extreme readings, reversion timing is unclear.")
        lines.append(f"  Consider hedging or reducing March put exposure.")
    
    print(Panel("\n".join(lines), title="📉 SLV Reversion Forecast 📉", expand=False))


def _show_comex_inventory(settings, console) -> None:
    """Display COMEX silver inventory analysis with ASCII chart."""
    import numpy as np
    from ai_options_trader.silver.comex import (
        get_current_comex_inventory,
        get_inventory_analysis_summary,
        build_inventory_vs_price_dataset,
    )
    
    try:
        comex = get_current_comex_inventory()
        summary = get_inventory_analysis_summary(settings)
        df = build_inventory_vs_price_dataset(settings, start_date="2020-01-01")
    except Exception as e:
        console.print(f"[red]Error loading COMEX data: {e}[/red]")
        return
    
    lines = []
    
    # Header with current inventory
    signal_color = {
        "bullish": "green",
        "bearish": "red",
        "neutral": "yellow",
    }.get(comex.signal, "white")
    
    trend_icon = {
        "rising": "📈",
        "falling": "📉",
        "stable": "➡️",
    }.get(comex.trend, "")
    
    lines.append(f"[bold]Current COMEX Silver Inventory:[/bold] {comex.inventory_moz:.0f} Moz {trend_icon}")
    lines.append(f"[bold]As of:[/bold] {comex.date}")
    lines.append(f"[bold]Signal:[/bold] [{signal_color}]{comex.signal.upper()}[/{signal_color}]")
    lines.append("")
    
    # Changes
    lines.append("[bold]Inventory Changes:[/bold]")
    
    def fmt_change(val, pct_val):
        if val is None:
            return "[dim]—[/dim]"
        color = "red" if val > 0 else "green" if val < 0 else "white"
        pct_str = f" ({pct_val:+.1f}%)" if pct_val else ""
        return f"[{color}]{val:+.0f} Moz{pct_str}[/{color}]"
    
    lines.append(f"  1-Month: {fmt_change(comex.change_1m_moz, comex.change_1m_pct)}")
    lines.append(f"  3-Month: {fmt_change(comex.change_3m_moz, comex.change_3m_pct)}")
    lines.append(f"  1-Year:  {fmt_change(comex.change_1y_moz, comex.change_1y_pct)}")
    lines.append("")
    
    # 5-year percentile
    if comex.percentile_5y is not None:
        pct = comex.percentile_5y
        bar_width = 30
        filled = int((pct / 100) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        if pct > 80:
            level_desc = "[red]HIGH (near 5Y highs)[/red]"
        elif pct > 60:
            level_desc = "[yellow]ELEVATED[/yellow]"
        elif pct > 40:
            level_desc = "[white]NORMAL[/white]"
        elif pct > 20:
            level_desc = "[cyan]LOW[/cyan]"
        else:
            level_desc = "[green]VERY LOW (near 5Y lows)[/green]"
        
        lines.append(f"[bold]5-Year Percentile:[/bold] {pct:.0f}% {level_desc}")
        lines.append(f"[dim]Low[/dim] {bar} [dim]High[/dim]")
        lines.append("")
    
    # ASCII chart: Inventory vs Price (last 2 years)
    lines.append("[bold underline]COMEX Inventory vs SLV Price (2-year view)[/bold underline]")
    lines.append("")
    
    if not df.empty and "inventory_normalized" in df.columns and "price_normalized" in df.columns:
        # Get last 2 years of data, sampled monthly
        recent = df.last("730D")
        if len(recent) > 24:
            # Sample to ~24 points
            step = len(recent) // 24
            sampled = recent.iloc[::step]
        else:
            sampled = recent
        
        # Build ASCII chart
        chart_height = 10
        chart_width = min(50, len(sampled))
        
        # Normalize data for chart
        inv_data = sampled["inventory_normalized"].values[-chart_width:]
        price_data = sampled["price_normalized"].values[-chart_width:]
        
        # Create chart grid
        for row in range(chart_height, -1, -1):
            threshold = (row / chart_height) * 100
            line = ""
            for col in range(chart_width):
                inv_val = inv_data[col] if col < len(inv_data) and not np.isnan(inv_data[col]) else 0
                price_val = price_data[col] if col < len(price_data) and not np.isnan(price_data[col]) else 0
                
                inv_here = inv_val >= threshold and inv_val < threshold + (100 / chart_height)
                price_here = price_val >= threshold and price_val < threshold + (100 / chart_height)
                
                if inv_here and price_here:
                    line += "[magenta]●[/magenta]"  # Overlap
                elif inv_here:
                    line += "[cyan]█[/cyan]"  # Inventory
                elif price_here:
                    line += "[yellow]▪[/yellow]"  # Price
                else:
                    line += " "
            
            # Y-axis label
            if row == chart_height:
                lines.append(f"High │{line}│")
            elif row == 0:
                lines.append(f"Low  │{line}│")
            else:
                lines.append(f"     │{line}│")
        
        lines.append(f"     └{'─' * chart_width}┘")
        lines.append(f"      {'2Y ago':<{chart_width//2}}{'Today':>{chart_width//2}}")
        lines.append("")
        lines.append("[cyan]█ Inventory[/cyan]  [yellow]▪ SLV Price[/yellow]  [magenta]● Overlap[/magenta]")
    
    lines.append("")
    
    # Divergence interpretation
    div = summary.get("divergence_score", 0) or 0
    interp = summary.get("divergence_interpretation", "")
    
    if div > 20:
        lines.append(f"[bold green]⚡ BULLISH DIVERGENCE[/bold green]: Inventory falling while price rising")
        lines.append("[dim]Physical tightness suggests more upside[/dim]")
    elif div < -20:
        lines.append(f"[bold red]⚠ BEARISH DIVERGENCE[/bold red]: Inventory rising while price weak")
        lines.append("[dim]Supply building suggests caution[/dim]")
    else:
        lines.append(f"[bold yellow]No significant divergence[/bold yellow]")
    
    lines.append("")
    
    # Key insight
    lines.append("[bold underline]KEY INSIGHT[/bold underline]")
    if comex.trend == "falling" and comex.change_3m_pct and comex.change_3m_pct < -10:
        lines.append("[green]Rapid inventory drawdown signals physical tightness.[/green]")
        lines.append("Historically bullish for silver prices.")
    elif comex.trend == "rising" and comex.change_3m_pct and comex.change_3m_pct > 10:
        lines.append("[red]Inventory build-up signals supply surplus.[/red]")
        lines.append("May cap near-term price appreciation.")
    else:
        lines.append("Inventory levels are within normal range.")
        lines.append("Watch for trend changes as leading indicator.")
    
    lines.append("")
    lines.append("[dim]Data source: CME COMEX warehouse reports (interpolated)[/dim]")
    lines.append("[dim]Update: Run 'lox labs silver --inventory' to refresh[/dim]")
    
    print(Panel("\n".join(lines), title="📦 COMEX Silver Inventory Analysis 📦", expand=False))


def _show_breakdown_levels(inp, df, console) -> None:
    """Display technical breakdown levels that would trigger selling pressure."""
    from ai_options_trader.silver.signals import get_breakdown_levels_summary
    
    current_price = inp.slv_price or 0
    if current_price <= 0:
        console.print("[red]Cannot compute breakdown levels - no price data[/red]")
        return
    
    levels_list = get_breakdown_levels_summary(df, current_price)
    
    if not levels_list:
        console.print("[yellow]No breakdown levels computed[/yellow]")
        return
    
    lines = []
    
    lines.append(f"[bold]Current Price: ${current_price:.2f}[/bold]")
    lines.append("")
    lines.append("[dim]Levels below current price that would trigger technical selling if broken:[/dim]")
    lines.append("")
    
    # Group levels by proximity
    immediate = [l for l in levels_list if l["pct_away"] >= -5]
    near = [l for l in levels_list if -15 <= l["pct_away"] < -5]
    medium = [l for l in levels_list if -30 <= l["pct_away"] < -15]
    far = [l for l in levels_list if l["pct_away"] < -30]
    
    def format_level(level):
        sig = level["significance"]
        if sig >= 5:
            color = "red"
            stars = "★★★★★"
        elif sig >= 4:
            color = "orange1"
            stars = "★★★★"
        elif sig >= 3:
            color = "yellow"
            stars = "★★★"
        else:
            color = "dim"
            stars = "★★"
        
        return f"  [{color}]${level['level']:.2f}[/{color}] ({level['pct_away']:+.1f}%) - {level['name']} {stars}\n    [dim]{level['description']}[/dim]"
    
    if immediate:
        lines.append("[bold red]⚠️  IMMEDIATE SUPPORT (within 5%)[/bold red]")
        for l in immediate:
            lines.append(format_level(l))
        lines.append("")
    
    if near:
        lines.append("[bold orange1]🎯 NEAR-TERM LEVELS (5-15% below)[/bold orange1]")
        for l in near:
            lines.append(format_level(l))
        lines.append("")
    
    if medium:
        lines.append("[bold yellow]📍 MEDIUM-TERM SUPPORT (15-30% below)[/bold yellow]")
        for l in medium:
            lines.append(format_level(l))
        lines.append("")
    
    if far:
        lines.append("[bold cyan]🏁 MAJOR SUPPORT ZONES (30%+ below)[/bold cyan]")
        for l in far:
            lines.append(format_level(l))
        lines.append("")
    
    # Add cascade analysis
    lines.append("[bold underline]CASCADE TRIGGERS[/bold underline]")
    lines.append("")
    
    # Find the first major level (significance >= 4)
    first_major = next((l for l in levels_list if l["significance"] >= 4), None)
    if first_major:
        lines.append(f"[bold]First Major Test:[/bold] ${first_major['level']:.2f} ({first_major['name']})")
        lines.append(f"  If this breaks, expect stops to trigger at:")
        
        # Find subsequent levels within 10% of first major
        subsequent = [l for l in levels_list 
                     if l["level"] < first_major["level"] 
                     and l["level"] >= first_major["level"] * 0.9]
        for s in subsequent[:3]:
            lines.append(f"    → ${s['level']:.2f} ({s['name']})")
    
    lines.append("")
    
    # Daily close significance
    lines.append("[bold underline]CLOSE BELOW = CONFIRMATION[/bold underline]")
    lines.append("")
    lines.append("A [bold]daily close[/bold] below these levels is more significant than intraday:")
    
    # Find most significant levels
    top_levels = sorted(levels_list, key=lambda x: (-x["significance"], x["pct_away"]))[:5]
    for l in top_levels:
        lines.append(f"  • ${l['level']:.2f} - {l['name']}")
    
    lines.append("")
    lines.append("[dim]★★★★★ = Highest significance (institutional/algo trigger)[/dim]")
    lines.append("[dim]★★★★ = High significance (momentum traders exit)[/dim]")
    lines.append("[dim]★★★ = Moderate significance (retail stops)[/dim]")
    
    print(Panel("\n".join(lines), title="🎯 Technical Breakdown Levels 🎯", expand=False))


def _run_silver_snapshot(
    start: str = "2011-01-01",
    refresh: bool = False,
    llm: bool = False,
    features: bool = False,
    json_out: bool = False,
    delta: str = "",
    alert: bool = False,
    puts: bool = False,
    bubble: bool = False,
    forecast: bool = False,
    levels: bool = False,
    inventory: bool = False,
):
    """Shared implementation for silver snapshot."""
    import numpy as np
    import pandas as pd
    from rich.console import Console
    from rich.table import Table

    from ai_options_trader.config import load_settings
    from ai_options_trader.silver.signals import build_silver_state, build_silver_dataset, get_breakdown_levels_summary
    from ai_options_trader.silver.regime import classify_silver_regime, get_regime_color, get_put_outlook
    from ai_options_trader.silver.forecast import build_reversion_forecast
    from ai_options_trader.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        show_alert_output,
    )

    console = Console()
    settings = load_settings()
    state = build_silver_state(settings=settings, start_date=start, refresh=refresh)
    regime = classify_silver_regime(state.inputs)
    regime_color = get_regime_color(regime)

    # Build snapshot data
    inp = state.inputs
    snapshot_data = {
        "slv_price": inp.slv_price,
        "slv_ma_50": inp.slv_ma_50,
        "slv_ma_200": inp.slv_ma_200,
        "slv_ret_5d_pct": inp.slv_ret_5d_pct,
        "slv_ret_20d_pct": inp.slv_ret_20d_pct,
        "slv_ret_60d_pct": inp.slv_ret_60d_pct,
        "slv_zscore_20d": inp.slv_zscore_20d,
        "slv_zscore_60d": inp.slv_zscore_60d,
        "slv_above_50ma": inp.slv_above_50ma,
        "slv_above_200ma": inp.slv_above_200ma,
        "slv_50ma_above_200ma": inp.slv_50ma_above_200ma,
        "gsr": inp.gsr,
        "gsr_zscore": inp.gsr_zscore,
        "gsr_expanding": inp.gsr_expanding,
        "slv_vol_20d_ann_pct": inp.slv_vol_20d_ann_pct,
        "trend_score": inp.trend_score,
        "momentum_score": inp.momentum_score,
        "relative_value_score": inp.relative_value_score,
        "bubble_score": inp.bubble_score,
        "mean_reversion_pressure": inp.mean_reversion_pressure,
        "extension_pct": inp.extension_pct,
        "days_at_extreme": inp.days_at_extreme,
        "regime": regime.label,
    }

    feature_dict = {
        "slv_price": inp.slv_price,
        "slv_ret_20d_pct": inp.slv_ret_20d_pct,
        "slv_ret_60d_pct": inp.slv_ret_60d_pct,
        "slv_zscore_20d": inp.slv_zscore_20d,
        "slv_zscore_60d": inp.slv_zscore_60d,
        "gsr": inp.gsr,
        "gsr_zscore": inp.gsr_zscore,
        "slv_vol_20d_ann_pct": inp.slv_vol_20d_ann_pct,
        "trend_score": inp.trend_score,
        "momentum_score": inp.momentum_score,
        "relative_value_score": inp.relative_value_score,
        "bubble_score": inp.bubble_score,
        "mean_reversion_pressure": inp.mean_reversion_pressure,
        "extension_pct": inp.extension_pct,
    }

    # Handle --features and --json flags
    if handle_output_flags(
        domain="silver",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=regime.label,
        regime_description=regime.description,
        asof=state.asof if hasattr(state, 'asof') else None,
        output_json=json_out,
        output_features=features,
    ):
        return

    # Handle --alert flag (silent unless extreme)
    if alert:
        show_alert_output("silver", regime.label, snapshot_data, regime.description)
        return

    # Handle --delta flag
    if delta:
        from ai_options_trader.cli_commands.shared.labs_utils import get_delta_metrics

        delta_days = parse_delta_period(delta)
        metric_keys = [
            "SLV:slv_price:$",
            "GSR:gsr:",
            "20d Return:slv_ret_20d_pct:%",
            "Trend Score:trend_score:",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("silver", snapshot_data, metric_keys, delta_days)
        show_delta_summary("silver", regime.label, prev_regime, metrics_for_delta, delta_days)

        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox labs silver` daily to build history.[/dim]")
        return

    # Helper for safe float formatting
    def _fmt(v, decimals=2, pct=False):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        suffix = "%" if pct else ""
        return f"{v:+.{decimals}f}{suffix}" if pct else f"{v:.{decimals}f}{suffix}"

    def _z(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:+.2f}"

    def _bool(v):
        if v is True:
            return "[green]Yes[/green]"
        elif v is False:
            return "[red]No[/red]"
        return "n/a"

    # Build main output
    body = (
        f"[b]Regime:[/b] [{regime_color}]{regime.label}[/{regime_color}]\n\n"
        f"[b]SLV Price:[/b] ${_fmt(inp.slv_price)}\n"
        f"[b]50-day MA:[/b] ${_fmt(inp.slv_ma_50)}  [b]Above:[/b] {_bool(inp.slv_above_50ma)}  [b]Dist:[/b] {_fmt(inp.slv_pct_from_50ma, 1, True)}\n"
        f"[b]200-day MA:[/b] ${_fmt(inp.slv_ma_200)}  [b]Above:[/b] {_bool(inp.slv_above_200ma)}  [b]Dist:[/b] {_fmt(inp.slv_pct_from_200ma, 1, True)}\n"
        f"[b]Golden Cross:[/b] {_bool(inp.slv_50ma_above_200ma)}\n\n"
        f"[b]Returns:[/b]  5d: {_fmt(inp.slv_ret_5d_pct, 1, True)}  |  20d: {_fmt(inp.slv_ret_20d_pct, 1, True)}  |  60d: {_fmt(inp.slv_ret_60d_pct, 1, True)}\n"
        f"[b]Z-scores:[/b]  20d: {_z(inp.slv_zscore_20d)}  |  60d: {_z(inp.slv_zscore_60d)}\n\n"
        f"[b]Gold/Silver Ratio (GSR):[/b] {_fmt(inp.gsr, 1)}\n"
        f"[b]GSR Z-score:[/b] {_z(inp.gsr_zscore)}  [b]Expanding:[/b] {_bool(inp.gsr_expanding)}\n\n"
        f"[b]Volatility (20d ann):[/b] {_fmt(inp.slv_vol_20d_ann_pct, 1)}%  [b]Vol z:[/b] {_z(inp.slv_vol_zscore)}\n\n"
        f"[b]Trend Score:[/b] {_fmt(inp.trend_score, 0)} / 100\n"
        f"[b]Momentum Score:[/b] {_fmt(inp.momentum_score, 0)} / 100\n"
        f"[b]Relative Value Score:[/b] {_fmt(inp.relative_value_score, 0)} / 100\n\n"
        f"[dim]{regime.description}[/dim]"
    )

    print(Panel(body, title="Silver / SLV Regime", expand=False))

    # Show bubble tracker if requested
    if bubble:
        _show_bubble_tracker(inp, console)

    # Show reversion forecast if requested
    if forecast:
        try:
            df = build_silver_dataset(settings=settings, start_date=start, refresh=refresh)
            reversion_forecast = build_reversion_forecast(settings=settings, state=state, df=df)
            _show_reversion_forecast(inp, reversion_forecast, console)
        except Exception as e:
            console.print(f"[red]Error building forecast: {e}[/red]")

    # Show breakdown levels if requested
    if levels:
        try:
            df = build_silver_dataset(settings=settings, start_date=start, refresh=refresh)
            _show_breakdown_levels(inp, df, console)
        except Exception as e:
            console.print(f"[red]Error computing breakdown levels: {e}[/red]")

    # Show COMEX inventory analysis if requested
    if inventory:
        _show_comex_inventory(settings, console)

    # Show put outlook if requested
    if puts:
        put_outlook = get_put_outlook(regime, inp)
        
        bias_color = {
            "favorable": "green",
            "unfavorable": "red", 
            "neutral": "yellow",
        }.get(put_outlook["bias"], "white")
        
        put_body = (
            f"[b]Put Bias:[/b] [{bias_color}]{put_outlook['bias'].upper()}[/{bias_color}]\n"
            f"[b]Confidence:[/b] {put_outlook['confidence']}%\n\n"
            f"[b]Notes:[/b]\n"
        )
        for note in put_outlook["notes"]:
            put_body += f"  • {note}\n"
        
        # Add key levels from features
        from ai_options_trader.silver.features import get_put_bias_features
        bias_features = get_put_bias_features(state, regime)
        
        if bias_features["key_levels"]:
            put_body += f"\n[b]Key Levels:[/b]\n"
            for level_name, level_val in bias_features["key_levels"].items():
                put_body += f"  • {level_name}: ${level_val:.2f}\n"
        
        if bias_features["catalysts"]:
            put_body += f"\n[b]Potential Catalysts:[/b]\n"
            for cat in bias_features["catalysts"]:
                put_body += f"  • {cat}\n"
        
        put_body += f"\n[b]Bearish Score:[/b] {bias_features['bearish_score']:+.2f} (range: -1 to +1)"
        
        print(Panel(put_body, title="SLV Put Outlook", expand=False))

    if llm:
        from ai_options_trader.llm.core.analyst import llm_analyze_regime
        from rich.markdown import Markdown

        print("\n[bold cyan]Generating LLM analysis...[/bold cyan]\n")

        # Add put context to LLM if puts flag is set
        extra_context = ""
        if puts:
            extra_context = "\n\nUser context: Holding long-dated SLV puts (March and September expiry). Focus analysis on downside scenarios and key support levels."

        analysis = llm_analyze_regime(
            settings=settings,
            domain="silver",
            snapshot=snapshot_data,
            regime_label=regime.label,
            regime_description=regime.description + extra_context,
        )

        print(Panel(Markdown(analysis), title="Analysis", expand=False))


def register(silver_app: typer.Typer) -> None:
    """Register silver commands."""

    @silver_app.callback(invoke_without_command=True)
    def silver_default(
        ctx: typer.Context,
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh of market data"),
        llm: bool = typer.Option(False, "--llm", help="Get PhD-level LLM analysis"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme"),
        puts: bool = typer.Option(False, "--puts", help="Show put position outlook and key levels"),
        bubble: bool = typer.Option(False, "--bubble", help="Show visual bubble tracker with mean reversion metrics"),
        forecast: bool = typer.Option(False, "--forecast", help="Show reversion timing/severity forecast"),
        levels: bool = typer.Option(False, "--levels", help="Show technical breakdown levels that would trigger selling"),
        inventory: bool = typer.Option(False, "--inventory", help="Show COMEX silver inventory vs price analysis"),
    ):
        """Silver / SLV regime tracker - price, technicals, gold/silver ratio."""
        if ctx.invoked_subcommand is None:
            _run_silver_snapshot(refresh=refresh, llm=llm, features=features, json_out=json_out, delta=delta, alert=alert, puts=puts, bubble=bubble, forecast=forecast, levels=levels, inventory=inventory)

    @silver_app.command("snapshot")
    def snapshot(
        start: str = typer.Option("2011-01-01", "--start"),
        refresh: bool = typer.Option(False, "--refresh"),
        llm: bool = typer.Option(False, "--llm", help="Get PhD-level LLM analysis"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme"),
        puts: bool = typer.Option(False, "--puts", help="Show put position outlook and key levels"),
        bubble: bool = typer.Option(False, "--bubble", help="Show visual bubble tracker with mean reversion metrics"),
        forecast: bool = typer.Option(False, "--forecast", help="Show reversion timing/severity forecast"),
        levels: bool = typer.Option(False, "--levels", help="Show technical breakdown levels that would trigger selling"),
        inventory: bool = typer.Option(False, "--inventory", help="Show COMEX silver inventory vs price analysis"),
    ):
        """Silver snapshot: SLV price, technicals, GSR, regime."""
        _run_silver_snapshot(start=start, refresh=refresh, llm=llm, features=features, json_out=json_out, delta=delta, alert=alert, puts=puts, bubble=bubble, forecast=forecast, levels=levels, inventory=inventory)
