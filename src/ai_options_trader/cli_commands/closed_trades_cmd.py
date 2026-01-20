"""
Closed Trades Command
Shows realized P&L from completed round-trip trades.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from ai_options_trader.config import load_settings
from ai_options_trader.data.alpaca import make_clients


def register(app: typer.Typer) -> None:
    @app.command("closed-trades")
    def closed_trades(
        limit: int = typer.Option(500, "--limit", "-l", help="Max orders to fetch"),
        details: bool = typer.Option(False, "--details", "-d", help="Show individual trade details"),
    ):
        """Show realized P&L from closed (round-trip) trades."""
        console = Console()
        
        try:
            settings = load_settings()
            trading, _ = make_clients(settings)
        except Exception as e:
            console.print(f"[red]Error loading settings: {e}[/red]")
            raise typer.Exit(1)
        
        # Fetch orders
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            
            req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
            orders = trading.get_orders(req) or []
        except Exception as e:
            console.print(f"[red]Error fetching orders: {e}[/red]")
            raise typer.Exit(1)
        
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
            is_option = _is_option_symbol(sym)
            mult = 100 if is_option else 1
            
            trade = {
                'qty': filled_qty,
                'price': price,
                'date': str(filled_at)[:10] if filled_at else '?',
                'value': filled_qty * price * mult
            }
            
            if side == 'buy':
                trades_by_symbol[sym]['buys'].append(trade)
            else:
                trades_by_symbol[sym]['sells'].append(trade)
        
        # Calculate closed trades
        closed_trades = []
        
        for sym, data in trades_by_symbol.items():
            buys = data['buys']
            sells = data['sells']
            
            total_bought_qty = sum(b['qty'] for b in buys)
            total_sold_qty = sum(s['qty'] for s in sells)
            total_bought_value = sum(b['value'] for b in buys)
            total_sold_value = sum(s['value'] for s in sells)
            
            display_sym, is_option = _parse_option_symbol(sym)
            
            # Only include if we have both buys and sells (round trip)
            if total_bought_qty > 0 and total_sold_qty > 0:
                # Calculate P&L
                if total_bought_qty == total_sold_qty:
                    realized_pnl = total_sold_value - total_bought_value
                    fully_closed = True
                else:
                    # Partial close - prorate
                    closed_qty = min(total_bought_qty, total_sold_qty)
                    close_ratio = closed_qty / max(total_bought_qty, total_sold_qty)
                    realized_pnl = (total_sold_value - total_bought_value) * close_ratio
                    fully_closed = False
                
                closed_trades.append({
                    'symbol': display_sym,
                    'raw_sym': sym,
                    'is_option': is_option,
                    'bought_qty': total_bought_qty,
                    'sold_qty': total_sold_qty,
                    'cost': total_bought_value,
                    'proceeds': total_sold_value,
                    'pnl': realized_pnl,
                    'buys': buys,
                    'sells': sells,
                    'fully_closed': fully_closed
                })
        
        if not closed_trades:
            console.print("[yellow]No closed trades found.[/yellow]")
            raise typer.Exit(0)
        
        # Sort by P&L (best first)
        closed_trades.sort(key=lambda x: x['pnl'], reverse=True)
        
        # Calculate totals
        total_realized = sum(t['pnl'] for t in closed_trades)
        total_wins = sum(1 for t in closed_trades if t['pnl'] >= 0)
        total_losses = sum(1 for t in closed_trades if t['pnl'] < 0)
        win_rate = total_wins / (total_wins + total_losses) * 100 if (total_wins + total_losses) > 0 else 0
        
        # Build summary table
        table = Table(title="Closed Trades - Realized P&L", show_header=True, header_style="bold")
        table.add_column("Position", style="cyan", width=40)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Cost", justify="right", width=12)
        table.add_column("Proceeds", justify="right", width=12)
        table.add_column("P&L", justify="right", width=12)
        
        for t in closed_trades:
            status = "CLOSED" if t['fully_closed'] else "PARTIAL"
            pnl_style = "green" if t['pnl'] >= 0 else "red"
            pnl_str = f"${t['pnl']:+,.2f}"
            
            table.add_row(
                t['symbol'],
                status,
                f"${t['cost']:,.2f}",
                f"${t['proceeds']:,.2f}",
                f"[{pnl_style}]{pnl_str}[/{pnl_style}]"
            )
        
        # Add total row
        table.add_section()
        pnl_style = "green bold" if total_realized >= 0 else "red bold"
        table.add_row(
            "[bold]TOTAL[/bold]",
            "",
            "",
            "",
            f"[{pnl_style}]${total_realized:+,.2f}[/{pnl_style}]"
        )
        
        console.print()
        console.print(table)
        
        # Summary panel
        summary_text = Text()
        summary_text.append(f"Win/Loss: ", style="dim")
        summary_text.append(f"{total_wins}W", style="green bold")
        summary_text.append(" / ", style="dim")
        summary_text.append(f"{total_losses}L", style="red bold")
        summary_text.append(f"  ({win_rate:.0f}% win rate)", style="dim")
        
        console.print()
        console.print(Panel(summary_text, title="Performance", border_style="blue"))
        
        # Show details if requested
        if details:
            console.print()
            console.print("[bold]Trade Details:[/bold]")
            
            for t in closed_trades:
                console.print()
                pnl_style = "green" if t['pnl'] >= 0 else "red"
                console.print(f"[cyan]{t['symbol']}[/cyan]")
                
                # Show buys
                for b in sorted(t['buys'], key=lambda x: x['date']):
                    qty_str = f"{b['qty']:.2f}" if b['qty'] % 1 else f"{b['qty']:.0f}"
                    console.print(f"  {b['date']} | [green]BUY [/green] {qty_str} @ ${b['price']:.2f} = ${b['value']:,.2f}")
                
                # Show sells
                for s in sorted(t['sells'], key=lambda x: x['date']):
                    qty_str = f"{s['qty']:.2f}" if s['qty'] % 1 else f"{s['qty']:.0f}"
                    console.print(f"  {s['date']} | [red]SELL[/red] {qty_str} @ ${s['price']:.2f} = ${s['value']:,.2f}")
                
                console.print(f"  [{pnl_style}]→ Realized: ${t['pnl']:+,.2f}[/{pnl_style}]")


def _is_option_symbol(sym: str) -> bool:
    """Check if symbol is an option."""
    if '/' in sym:  # Crypto
        return False
    if len(sym) <= 6:  # Stock/ETF
        return False
    # Options have format: TICKER + YYMMDD + C/P + STRIKE (e.g., TLT260717C00105000)
    return len(sym) > 10 and any(c.isdigit() for c in sym[-8:])


def _parse_option_symbol(sym: str) -> tuple[str, bool]:
    """Parse option symbol for display. Returns (display_name, is_option)."""
    if '/' in sym:  # Crypto
        return sym, False
    if len(sym) <= 6:  # Stock/ETF
        return sym, False
    
    # Try to parse OCC format: TICKER + YYMMDD + C/P + STRIKE
    try:
        # Find where digits start
        i = 0
        while i < len(sym) and not sym[i].isdigit():
            i += 1
        
        if i == 0 or i >= len(sym):
            return sym, False
        
        ticker = sym[:i]
        rest = sym[i:]
        
        if len(rest) >= 15:
            exp = f"20{rest[:2]}-{rest[2:4]}-{rest[4:6]}"
            opt_type = "CALL" if rest[6] == 'C' else "PUT"
            strike = int(rest[7:]) / 1000
            return f"{ticker} ${strike:.0f} {opt_type} {exp}", True
    except (ValueError, IndexError):
        pass
    
    return sym, len(sym) > 10
