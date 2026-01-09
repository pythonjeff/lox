from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="AI Options Trader CLI")
macro_app = typer.Typer(add_completion=False, help="Macro signals and datasets")
app.add_typer(macro_app, name="macro")
tariff_app = typer.Typer(add_completion=False, help="Tariff / cost-push regime signals")
app.add_typer(tariff_app, name="tariff")
options_app = typer.Typer(add_completion=False, help="Options scanners (chains, volume/OI ranking)")
app.add_typer(options_app, name="options")
funding_app = typer.Typer(add_completion=False, help="Funding regime (secured rates: SOFR/TGCR/BGCR corridor)")
app.add_typer(funding_app, name="funding")
# Back-compat alias; will be removed once callers migrate.
liquidity_app = typer.Typer(add_completion=False, help="(deprecated) Alias for funding regime")
app.add_typer(liquidity_app, name="liquidity")
usd_app = typer.Typer(add_completion=False, help="USD strength/weakness regime")
app.add_typer(usd_app, name="usd")
monetary_app = typer.Typer(add_completion=False, help="Monetary regime (reserves, Fed balance sheet, RRP, policy rate)")
app.add_typer(monetary_app, name="monetary")
rates_app = typer.Typer(add_completion=False, help="Rates / yield curve regime (UST level/slope/momentum)")
app.add_typer(rates_app, name="rates")
fiscal_app = typer.Typer(add_completion=False, help="US fiscal regime (deficits, issuance mix, auctions, TGA)")
app.add_typer(fiscal_app, name="fiscal")
vol_app = typer.Typer(add_completion=False, help="Volatility regime (VIX: level/momentum/term structure)")
app.add_typer(vol_app, name="volatility")
commod_app = typer.Typer(add_completion=False, help="Commodities regime (oil/gold/broad index)")
app.add_typer(commod_app, name="commodities")
crypto_app = typer.Typer(add_completion=False, help="Crypto snapshots and LLM outlooks")
app.add_typer(crypto_app, name="crypto")
ticker_app = typer.Typer(add_completion=False, help="Ticker snapshots and LLM outlooks")
app.add_typer(ticker_app, name="ticker")
ideas_app = typer.Typer(add_completion=False, help="Trade idea generation from thesis + regimes")
app.add_typer(ideas_app, name="ideas")
track_app = typer.Typer(add_completion=False, help="Track recommendations, executions, and performance")
app.add_typer(track_app, name="track")
nav_app = typer.Typer(add_completion=False, help="NAV sheet + cashflow tracking (returns over time)")
app.add_typer(nav_app, name="nav")
autopilot_app = typer.Typer(add_completion=False, help="Macro autopilot (run-once workflows)")
app.add_typer(autopilot_app, name="autopilot")

_COMMANDS_REGISTERED = False

def _register_commands() -> None:
    global _COMMANDS_REGISTERED
    if _COMMANDS_REGISTERED:
        return
    # Import here to keep `ai_options_trader.cli` lightweight at import time.
    from ai_options_trader.cli_commands.select_cmd import register as register_select
    from ai_options_trader.cli_commands.macro_cmd import register as register_macro
    from ai_options_trader.cli_commands.tariff_cmd import register as register_tariff
    from ai_options_trader.cli_commands.options_cmd import register as register_options
    from ai_options_trader.cli_commands.funding_cmd import register as register_funding
    from ai_options_trader.cli_commands.liquidity_cmd import register as register_liquidity
    from ai_options_trader.cli_commands.usd_cmd import register as register_usd
    from ai_options_trader.cli_commands.monetary_cmd import register as register_monetary
    from ai_options_trader.cli_commands.rates_cmd import register as register_rates
    from ai_options_trader.cli_commands.fiscal_cmd import register as register_fiscal
    from ai_options_trader.cli_commands.volatility_cmd import register as register_volatility
    from ai_options_trader.cli_commands.commodities_cmd import register as register_commodities
    from ai_options_trader.cli_commands.crypto_cmd import register as register_crypto
    from ai_options_trader.cli_commands.ticker_cmd import register as register_ticker
    from ai_options_trader.cli_commands.regimes_cmd import register as register_regimes
    from ai_options_trader.cli_commands.ideas_cmd import register as register_ideas
    from ai_options_trader.cli_commands.macro_model_cmd import register as register_macro_model
    from ai_options_trader.cli_commands.track_cmd import register as register_track
    from ai_options_trader.cli_commands.nav_cmd import register as register_nav
    from ai_options_trader.cli_commands.autopilot_cmd import register as register_autopilot
    from ai_options_trader.cli_commands.portfolio_cmd import register as register_portfolio
    from ai_options_trader.cli_commands.account_cmd import register as register_account

    register_select(app)
    register_macro(macro_app)
    register_tariff(tariff_app)
    register_options(options_app)
    register_funding(funding_app)
    register_liquidity(liquidity_app)
    register_usd(usd_app)
    register_monetary(monetary_app)
    register_rates(rates_app)
    register_fiscal(fiscal_app)
    register_volatility(vol_app)
    register_commodities(commod_app)
    register_crypto(crypto_app)
    register_ticker(ticker_app)
    register_portfolio(app)
    register_regimes(app)
    register_ideas(ideas_app)
    register_macro_model(ideas_app)
    register_track(track_app)
    register_nav(nav_app)
    register_autopilot(autopilot_app)
    register_account(app)
    _COMMANDS_REGISTERED = True


def main():
    _register_commands()
    app()

# Register commands when imported as a console-script entrypoint (`pyproject.toml` uses `ai_options_trader.cli:app`).
_register_commands()


if __name__ == "__main__":
    main()
