from __future__ import annotations

import pandas as pd
import typer
from rich import print

from ai_options_trader.config import load_settings


def register(tariff_app: typer.Typer) -> None:
    @tariff_app.command("baskets")
    def tariff_baskets():
        """List available tariff baskets."""
        from ai_options_trader.tariff.universe import BASKETS

        for name, b in BASKETS.items():
            print(f"- {name}: {b.description} (tickers={','.join(b.tickers)})")

    @tariff_app.command("snapshot")
    def tariff_snapshot(
        basket: str = typer.Option("import_retail_apparel", "--basket"),
        start: str = typer.Option("2011-01-01", "--start"),
        benchmark: str = typer.Option("XLY", "--benchmark", help="Sector or market benchmark (e.g., XLY, SPY)"),
        refresh: bool = typer.Option(False, "--refresh"),
    ):
        """
        Compute tariff/cost-push regime snapshot for an import-exposed basket.
        """
        from ai_options_trader.data.fred import FredClient
        from ai_options_trader.data.market import fetch_equity_daily_closes
        from ai_options_trader.tariff.universe import BASKETS
        from ai_options_trader.tariff.proxies import DEFAULT_COST_PROXY_SERIES
        from ai_options_trader.tariff.signals import build_tariff_regime_state

        settings = load_settings()

        if basket not in BASKETS:
            raise typer.BadParameter(f"Unknown basket: {basket}. Choose from: {list(BASKETS.keys())}")

        universe = BASKETS[basket].tickers

        # --- Cost proxies (FRED) ---
        fred = FredClient(api_key=settings.FRED_API_KEY)

        frames = []
        for col, sid in DEFAULT_COST_PROXY_SERIES.items():
            df = fred.fetch_series(sid, start_date=start, refresh=refresh)
            df = df.rename(columns={"value": col}).set_index("date")
            frames.append(df[[col]])

        cost_df = pd.concat(frames, axis=1).sort_index()

        # Align to daily for merging with equities
        cost_df = cost_df.resample("D").ffill()

        # --- Equities (historical closes; default: FMP) ---
        symbols = sorted(set(universe + [benchmark]))
        px = fetch_equity_daily_closes(settings=settings, symbols=symbols, start=start, refresh=bool(refresh))
        px = px.sort_index().ffill().dropna(how="all")

        state = build_tariff_regime_state(
            cost_df=cost_df,
            equity_prices=px,
            universe=universe,
            benchmark=benchmark,
            basket_name=basket,
            start_date=start,
        )

        print(state)


