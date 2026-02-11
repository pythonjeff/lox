"""LOX Crypto CLI — data, research, and trading for crypto perpetual futures.

Commands:
    lox crypto data           Fetch and display perps data
    lox crypto research       Research analysis with LLM (macro regime)
    lox crypto analyze        Perps trading analysis (alpha-arena framework)
    lox crypto trade          Place a trade on Aster DEX
    lox crypto positions      Show open DEX positions
    lox crypto balance        Show DEX account balance
    lox crypto close          Close a position
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lox.config import load_settings

# ---------------------------------------------------------------------------
# Perps trading analysis prompt — adapted from alpha-arena-clone framework
# ---------------------------------------------------------------------------
PERPS_ANALYSIS_PROMPT = """\
# ROLE & OBJECTIVE
You are a senior crypto quantitative analyst providing trading guidance for \
perpetual futures. Analyze the provided market data and give actionable \
recommendations for manual trading.

# DATA YOU WILL RECEIVE
Multi-timeframe data for each coin including:
- **Short timeframe**: Last 10 data points with Price, EMA20, MACD, RSI(7), RSI(14)
- **Long timeframe**: EMA20 vs EMA50, ATR(14), Volume vs Average, MACD, RSI(14)
- **Market structure**: Open Interest, Funding Rate

All arrays are ordered OLDEST → NEWEST. The LAST value is MOST CURRENT.

# ANALYSIS FRAMEWORK

## Step 1: 4H Trend Direction (Primary Filter)
- **Bullish**: EMA20 > EMA50 → favor LONG setups
- **Bearish**: EMA20 < EMA50 → favor SHORT setups
- **Neutral**: EMAs within 0.5% → no strong directional bias

## Step 2: Short-TF Execution Setup
- **LONG Setup**: Price > short EMA20 AND short EMA20 trending above long EMA20
- **SHORT Setup**: Price < short EMA20 AND short EMA20 trending below long EMA20

## Step 3: Conviction Assessment
**Tier 1 (Required — all must pass for a trade idea):**
1. Trade direction matches 4H EMA20/EMA50 trend
2. Proper EMA stack alignment
3. Volume confirmation (current > 1.0x average)

**Tier 2 (Confirming — 2/3 builds conviction):**
1. RSI not in extreme conflicting zone (no RSI>70 for shorts, no RSI<30 for longs)
2. MACD alignment across timeframes
3. ATR context (volatility regime)

**Tier 3 (Supporting):**
- Funding rate favorable for trade direction
- Open Interest trending with price

## Step 4: Risk Parameters
- **Stop-loss**: 2x ATR(14) from entry
- **Profit target**: 2.5x ATR(14) from entry
- **Position sizing**: 3-5% of capital at risk per trade

# OUTPUT FORMAT
For each coin, provide:

1. **Trend**: Bullish/Bearish/Neutral with key levels
2. **Setup**: Whether a valid entry exists right now
3. **Conviction**: High/Medium/Low with tier breakdown
4. **Trade Idea** (if conviction is medium+):
   - Direction (LONG/SHORT)
   - Entry zone
   - Stop-loss level
   - Profit target
   - Risk/reward ratio
5. **Key Levels to Watch**: Support/resistance, EMA levels

End with a brief **Market Context** section covering cross-coin themes \
(correlation, funding rate divergences, volume patterns).

Be specific with numbers. No vague statements — every recommendation should \
reference actual indicator values from the data.
"""


def register(crypto_app: typer.Typer) -> None:
    """Register all crypto subcommands."""

    # ------------------------------------------------------------------
    # lox crypto data
    # ------------------------------------------------------------------
    @crypto_app.command("data")
    def data_cmd(
        coins: str = typer.Option("BTC,ETH,SOL", "--coins", "-c", help="Comma-separated coins"),
        exchange: str = typer.Option("", "--exchange", "-e", help="Override CCXT exchange (okx, bybit, binance)"),
        short_tf: str = typer.Option("15m", "--short-tf", help="Short timeframe"),
        long_tf: str = typer.Option("4h", "--long-tf", help="Long timeframe"),
    ):
        """Fetch current crypto perps data (prices, OI, funding, indicators)."""
        console = Console()
        settings = load_settings()
        if exchange:
            settings.CCXT_EXCHANGE = exchange

        coin_list = [c.strip().upper() for c in coins.split(",")]

        from lox.data.crypto_perps import CryptoPerpsData

        console.print(f"\n[bold cyan]Fetching perps data from {settings.CCXT_EXCHANGE.upper()}...[/bold cyan]\n")

        fetcher = CryptoPerpsData(settings)
        snapshots = fetcher.multi_snapshot(coins=coin_list, short_tf=short_tf, long_tf=long_tf)

        if not snapshots:
            console.print("[yellow]No data returned. Check exchange/coin availability.[/yellow]")
            return

        _render_overview_table(console, snapshots)
        _render_technicals_table(console, snapshots, short_tf, long_tf)

    # ------------------------------------------------------------------
    # lox crypto research
    # ------------------------------------------------------------------
    @crypto_app.command("research")
    def research_cmd(
        coins: str = typer.Option("BTC,ETH,SOL", "--coins", "-c", help="Comma-separated coins"),
        exchange: str = typer.Option("", "--exchange", "-e", help="Override CCXT exchange"),
        short_tf: str = typer.Option("15m", "--short-tf", help="Short timeframe"),
        long_tf: str = typer.Option("4h", "--long-tf", help="Long timeframe"),
        no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM analysis, data only"),
    ):
        """Crypto perps research — data + LLM analysis."""
        console = Console()
        settings = load_settings()
        if exchange:
            settings.CCXT_EXCHANGE = exchange

        coin_list = [c.strip().upper() for c in coins.split(",")]

        from lox.data.crypto_perps import CryptoPerpsData

        console.print(f"\n[bold cyan]Fetching perps data from {settings.CCXT_EXCHANGE.upper()}...[/bold cyan]\n")

        fetcher = CryptoPerpsData(settings)
        snapshots = fetcher.multi_snapshot(coins=coin_list, short_tf=short_tf, long_tf=long_tf)

        if not snapshots:
            console.print("[yellow]No data returned.[/yellow]")
            return

        # Show data tables (reuse data_cmd display logic)
        _render_overview_table(console, snapshots)
        _render_technicals_table(console, snapshots, short_tf, long_tf)

        if no_llm:
            return

        # LLM analysis
        if not settings.openai_api_key:
            console.print("[yellow]OPENAI_API_KEY not set — skipping LLM analysis[/yellow]")
            return

        console.print("\n[bold cyan]Generating LLM analysis...[/bold cyan]\n")

        llm_data = CryptoPerpsData.format_multi_for_llm(snapshots)

        from lox.llm.core.analyst import llm_analyze_regime
        from rich.markdown import Markdown

        snapshot_for_llm = {
            "coins": list(snapshots.keys()),
            "perps_data": llm_data,
        }
        # Add summary metrics for the LLM
        for coin, snap in snapshots.items():
            snapshot_for_llm[f"{coin}_price"] = snap["price"]
            if snap.get("funding") and snap["funding"].get("funding_rate") is not None:
                snapshot_for_llm[f"{coin}_funding"] = snap["funding"]["funding_rate"]

        analysis = llm_analyze_regime(
            settings=settings,
            domain="crypto",
            snapshot=snapshot_for_llm,
            regime_label="Crypto Perps Analysis",
            regime_description=f"Real-time perpetual futures data for {', '.join(snapshots.keys())}",
        )

        console.print(Panel(Markdown(analysis), title="Crypto Research", expand=False))

    # ------------------------------------------------------------------
    # lox crypto trade
    # ------------------------------------------------------------------
    @crypto_app.command("trade")
    def trade_cmd(
        coin: str = typer.Argument(..., help="Coin (BTC, ETH, SOL)"),
        side: str = typer.Argument(..., help="BUY or SELL"),
        quantity: float = typer.Argument(..., help="Quantity in coin units"),
        order_type: str = typer.Option("MARKET", "--type", "-t", help="MARKET, LIMIT, STOP_MARKET"),
        price: float = typer.Option(None, "--price", "-p", help="Limit price"),
        stop_price: float = typer.Option(None, "--stop", "-s", help="Stop/trigger price"),
        leverage: int = typer.Option(None, "--leverage", "-l", help="Set leverage (1-125)"),
    ):
        """Place a trade on Aster DEX."""
        console = Console()
        settings = load_settings()

        from lox.trading.aster import AsterClient

        client = AsterClient(settings)
        if not client.is_configured:
            console.print(Panel(
                "[red]Aster DEX not configured.[/red]\n\n"
                "Set these env vars in .env:\n"
                "  ASTER_USER_ADDRESS\n"
                "  ASTER_SIGNER_ADDRESS\n"
                "  ASTER_PRIVATE_KEY",
                title="Error", expand=False,
            ))
            raise typer.Exit(code=1)

        coin = coin.upper()
        side = side.upper()
        if side not in ("BUY", "SELL"):
            console.print("[red]Side must be BUY or SELL[/red]")
            raise typer.Exit(code=1)

        # Confirmation
        current_price = client.get_price(coin)
        console.print(Panel(
            f"[bold]{side} {quantity} {coin}[/bold]\n"
            f"Type: {order_type.upper()}\n"
            f"Current price: ${current_price:,.2f}\n"
            f"Leverage: {leverage or 'unchanged'}\n"
            f"Est. notional: ${quantity * current_price:,.2f}",
            title="Order Preview", expand=False,
        ))

        if not typer.confirm("Execute this trade?"):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit()

        try:
            result = client.place_order(
                coin=coin,
                side=side,
                quantity=quantity,
                order_type=order_type.upper(),
                price=price,
                stop_price=stop_price,
                leverage=leverage,
            )
            console.print(Panel(
                f"[green]Order placed successfully[/green]\n\n"
                f"Order ID: {result.get('orderId', 'N/A')}\n"
                f"Status: {result.get('status', 'N/A')}",
                title="Trade Result", expand=False,
            ))
        except Exception as e:
            console.print(Panel(f"[red]Trade failed[/red]\n\n{e}", title="Error", expand=False))
            raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # lox crypto positions
    # ------------------------------------------------------------------
    @crypto_app.command("positions")
    def positions_cmd():
        """Show current Aster DEX positions."""
        console = Console()
        settings = load_settings()

        from lox.trading.aster import AsterClient

        client = AsterClient(settings)
        if not client.is_configured:
            console.print("[red]Aster DEX not configured. Set ASTER_* env vars.[/red]")
            raise typer.Exit(code=1)

        try:
            positions = client.get_positions()
        except Exception as e:
            console.print(Panel(f"[red]Failed to fetch positions[/red]\n\n{e}", title="Error", expand=False))
            raise typer.Exit(code=1)

        if not positions:
            console.print("\n[yellow]No open positions[/yellow]\n")
            return

        table = Table(title="Open Positions", show_header=True, expand=False)
        table.add_column("Symbol", style="cyan bold")
        table.add_column("Side", justify="center")
        table.add_column("Size", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Mark", justify="right")
        table.add_column("PnL", justify="right")
        table.add_column("Leverage", justify="right")

        for pos in positions:
            amt = float(pos.get("positionAmt", 0))
            side_str = "[green]LONG[/green]" if amt > 0 else "[red]SHORT[/red]"
            entry = float(pos.get("entryPrice", 0))
            mark = float(pos.get("markPrice", 0))
            pnl = float(pos.get("unRealizedProfit", 0))
            lev = pos.get("leverage", "?")
            pnl_color = "green" if pnl >= 0 else "red"

            table.add_row(
                pos.get("symbol", "?"),
                side_str,
                f"{abs(amt)}",
                f"${entry:,.2f}",
                f"${mark:,.2f}",
                f"[{pnl_color}]${pnl:,.2f}[/{pnl_color}]",
                f"{lev}x",
            )

        console.print(table)
        console.print()

    # ------------------------------------------------------------------
    # lox crypto balance
    # ------------------------------------------------------------------
    @crypto_app.command("balance")
    def balance_cmd():
        """Show Aster DEX account balance."""
        console = Console()
        settings = load_settings()

        from lox.trading.aster import AsterClient

        client = AsterClient(settings)
        if not client.is_configured:
            console.print("[red]Aster DEX not configured. Set ASTER_* env vars.[/red]")
            raise typer.Exit(code=1)

        try:
            state = client.get_account_state()
        except Exception as e:
            console.print(Panel(f"[red]Failed to fetch balance[/red]\n\n{e}", title="Error", expand=False))
            raise typer.Exit(code=1)

        upnl = state["totalUnrealizedProfit"]
        pnl_color = "green" if upnl >= 0 else "red"

        console.print(Panel(
            f"Wallet Balance:    ${state['totalWalletBalance']:,.2f}\n"
            f"Available:         ${state['availableBalance']:,.2f}\n"
            f"Margin Balance:    ${state['totalMarginBalance']:,.2f}\n"
            f"Unrealized PnL:    [{pnl_color}]${upnl:,.2f}[/{pnl_color}]",
            title="Aster DEX Account", expand=False,
        ))

    # ------------------------------------------------------------------
    # lox crypto close
    # ------------------------------------------------------------------
    @crypto_app.command("close")
    def close_cmd(
        coin: str = typer.Argument(..., help="Coin to close position for (BTC, ETH, SOL)"),
    ):
        """Close a position on Aster DEX."""
        console = Console()
        settings = load_settings()

        from lox.trading.aster import AsterClient

        client = AsterClient(settings)
        if not client.is_configured:
            console.print("[red]Aster DEX not configured. Set ASTER_* env vars.[/red]")
            raise typer.Exit(code=1)

        coin = coin.upper()

        if not typer.confirm(f"Close {coin} position?"):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit()

        try:
            result = client.close_position(coin)
            if result.get("status") == "no_position":
                console.print(f"[yellow]No open position for {coin}[/yellow]")
            else:
                console.print(f"[green]Position closed for {coin}[/green]")
        except Exception as e:
            console.print(Panel(f"[red]Failed to close position[/red]\n\n{e}", title="Error", expand=False))
            raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # lox crypto analyze
    # ------------------------------------------------------------------
    @crypto_app.command("analyze")
    def analyze_cmd(
        coins: str = typer.Option("BTC,ETH,SOL", "--coins", "-c", help="Comma-separated coins"),
        exchange: str = typer.Option("", "--exchange", "-e", help="Override CCXT exchange"),
        short_tf: str = typer.Option("15m", "--short-tf", help="Short timeframe"),
        long_tf: str = typer.Option("4h", "--long-tf", help="Long timeframe"),
    ):
        """Perps trading analysis — alpha-arena framework with trade ideas."""
        console = Console()
        settings = load_settings()
        if exchange:
            settings.CCXT_EXCHANGE = exchange

        if not settings.openai_api_key:
            console.print("[red]OPENAI_API_KEY (or OpenRouter key) required for analyze.[/red]")
            raise typer.Exit(code=1)

        coin_list = [c.strip().upper() for c in coins.split(",")]

        from lox.data.crypto_perps import CryptoPerpsData

        console.print(f"\n[bold cyan]Fetching perps data from {settings.CCXT_EXCHANGE.upper()}...[/bold cyan]\n")

        fetcher = CryptoPerpsData(settings)
        snapshots = fetcher.multi_snapshot(coins=coin_list, short_tf=short_tf, long_tf=long_tf)

        if not snapshots:
            console.print("[yellow]No data returned.[/yellow]")
            return

        # Show data tables first
        _render_overview_table(console, snapshots)
        _render_technicals_table(console, snapshots, short_tf, long_tf)

        # Build the formatted market data (alpha-arena style)
        llm_data = CryptoPerpsData.format_multi_for_llm(snapshots)

        console.print("[bold cyan]Running perps trading analysis...[/bold cyan]\n")

        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package required") from e

        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.OPENAI_BASE_URL)

        user_msg = (
            f"Analyze the following crypto perpetual futures data. "
            f"Short timeframe: {short_tf}, Long timeframe: {long_tf}.\n\n"
            f"CURRENT MARKET STATE\n{llm_data}"
        )

        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": PERPS_ANALYSIS_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=4000,
        )

        analysis = resp.choices[0].message.content or ""

        from rich.markdown import Markdown

        console.print(Panel(Markdown(analysis), title="Perps Trading Analysis", expand=False))


# ------------------------------------------------------------------
# Shared display helpers
# ------------------------------------------------------------------

def _render_overview_table(console: Console, snapshots: dict) -> None:
    """Render the price overview table."""
    table = Table(title="Crypto Perps Overview", show_header=True, expand=False)
    table.add_column("Coin", style="cyan bold")
    table.add_column("Price", justify="right")
    table.add_column("24h %", justify="right")
    table.add_column("24h Vol", justify="right")
    table.add_column("Funding", justify="right")
    table.add_column("Open Interest", justify="right")

    for coin, snap in snapshots.items():
        price = snap["price"]
        pd_ = 2 if price >= 100 else (4 if price >= 1 else 5)

        chg = vol = ""
        if snap.get("ticker"):
            t = snap["ticker"]
            if t.get("change_pct_24h") is not None:
                pct = t["change_pct_24h"]
                color = "green" if pct >= 0 else "red"
                chg = f"[{color}]{pct:+.2f}%[/{color}]"
            if t.get("volume_24h") is not None:
                vol = f"{t['volume_24h']:,.0f}"

        fr = ""
        if snap.get("funding") and snap["funding"].get("funding_rate") is not None:
            fr_val = snap["funding"]["funding_rate"] * 100
            color = "green" if fr_val >= 0 else "red"
            fr = f"[{color}]{fr_val:.4f}%[/{color}]"

        oi = ""
        if snap.get("open_interest") and snap["open_interest"].get("open_interest"):
            oi = f"{snap['open_interest']['open_interest']:,.0f}"

        table.add_row(coin, f"${price:,.{pd_}f}", chg, vol, fr, oi)

    console.print(table)


def _render_technicals_table(console: Console, snapshots: dict, short_tf: str, long_tf: str) -> None:
    """Render short-term and long-term technicals tables with aligned shared columns."""
    #                Coin | RSI(7) | RSI(14) | MACD | EMA20 | EMA50 | EMA9  | BB Pos
    #                Coin | Trend  | RSI(14) | MACD | EMA20 | EMA50 | ATR   | Vol vs Avg

    # --- Short timeframe (momentum / entries) ---
    st = Table(title=f"Short-Term Technicals ({short_tf})", show_header=True, expand=False)
    st.add_column("Coin", style="cyan bold")
    st.add_column("RSI(7)", justify="right")
    st.add_column("RSI(14)", justify="right")
    st.add_column("MACD", justify="right")
    st.add_column("EMA20", justify="right")
    st.add_column("EMA50", justify="right")
    st.add_column("EMA9", justify="right")
    st.add_column("BB Pos", justify="right")

    for coin, snap in snapshots.items():
        s = snap["short_tf"]["latest"]
        price = snap["price"]
        pd_ = 2 if price >= 100 else (4 if price >= 1 else 5)

        bb_range = s["bb_upper"] - s["bb_lower"]
        bb_pos = ((price - s["bb_lower"]) / bb_range * 100) if bb_range > 0 else 50

        rsi7_c = "red" if s["rsi_7"] > 70 else ("green" if s["rsi_7"] < 30 else "white")
        rsi14_c = "red" if s["rsi_14"] > 70 else ("green" if s["rsi_14"] < 30 else "white")

        st.add_row(
            coin,
            f"[{rsi7_c}]{s['rsi_7']:.1f}[/{rsi7_c}]",
            f"[{rsi14_c}]{s['rsi_14']:.1f}[/{rsi14_c}]",
            f"{s['macd']:.2f}",
            f"{s['ema_20']:.{pd_}f}",
            f"{s['ema_50']:.{pd_}f}",
            f"{s['ema_9']:.{pd_}f}",
            f"{bb_pos:.0f}%",
        )

    console.print(st)

    # --- Long timeframe (trend context) — columns aligned with short-term ---
    lt = Table(title=f"Trend Context ({long_tf})", show_header=True, expand=False)
    lt.add_column("Coin", style="cyan bold")
    lt.add_column("Trend", justify="center")
    lt.add_column("RSI(14)", justify="right")
    lt.add_column("MACD", justify="right")
    lt.add_column("EMA20", justify="right")
    lt.add_column("EMA50", justify="right")
    lt.add_column("ATR(14)", justify="right")
    lt.add_column("Vol vs Avg", justify="right")

    for coin, snap in snapshots.items():
        l = snap["long_tf"]["latest"]
        price = snap["price"]
        pd_ = 2 if price >= 100 else (4 if price >= 1 else 5)

        if l["ema_20"] > l["ema_50"]:
            trend = "[green]Bullish[/green]"
        else:
            trend = "[red]Bearish[/red]"

        rsi_c = "red" if l["rsi_14"] > 70 else ("green" if l["rsi_14"] < 30 else "white")

        vol_ratio = l["volume"] / l["volume_ma"] if l["volume_ma"] > 0 else 0
        vol_c = "green" if vol_ratio > 1.2 else ("red" if vol_ratio < 0.8 else "white")

        lt.add_row(
            coin,
            trend,
            f"[{rsi_c}]{l['rsi_14']:.1f}[/{rsi_c}]",
            f"{l['macd']:.2f}",
            f"{l['ema_20']:.{pd_}f}",
            f"{l['ema_50']:.{pd_}f}",
            f"{l['atr_14']:.{pd_}f}",
            f"[{vol_c}]{vol_ratio:.1f}x[/{vol_c}]",
        )

    console.print(lt)
    console.print()
