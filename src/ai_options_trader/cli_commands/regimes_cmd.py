from __future__ import annotations

import pandas as pd
import typer
from rich import print

from ai_options_trader.config import load_settings
from ai_options_trader.macro.regime import classify_macro_regime_from_state
from ai_options_trader.macro.signals import build_macro_state


def register(app: typer.Typer) -> None:
    @app.command("regimes")
    def regimes(
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
        benchmark: str = typer.Option("XLY", "--benchmark", help="Sector or market benchmark (e.g., XLY, SPY)"),
        baskets: str = typer.Option(
            "all",
            "--baskets",
            help="Comma-separated basket names, or 'all' (see: ai-options-trader tariff baskets)",
        ),
        llm: bool = typer.Option(False, "--llm", help="Ask an LLM to summarize regimes + follow-ups"),
        llm_model: str = typer.Option("", "--llm-model", help="Override OPENAI_MODEL (optional)"),
        llm_temperature: float = typer.Option(0.2, "--llm-temperature", help="LLM temperature (0..2)"),
    ):
        """
        Print the current macro regime and all tariff/cost-push regimes (by basket).
        """
        from ai_options_trader.data.fred import FredClient
        from ai_options_trader.data.market import fetch_equity_daily_closes
        from ai_options_trader.funding.signals import build_funding_state
        from ai_options_trader.tariff.universe import BASKETS
        from ai_options_trader.tariff.proxies import DEFAULT_COST_PROXY_SERIES
        from ai_options_trader.tariff.signals import build_tariff_regime_state

        settings = load_settings()

        # --- Macro ---
        macro_state = build_macro_state(settings=settings, start_date=start, refresh=refresh)
        print(macro_state)
        macro_regime = classify_macro_regime_from_state(
            cpi_yoy=macro_state.inputs.cpi_yoy,
            payrolls_3m_annualized=macro_state.inputs.payrolls_3m_annualized,
            inflation_momentum_minus_be5y=macro_state.inputs.inflation_momentum_minus_be5y,
            real_yield_proxy_10y=macro_state.inputs.real_yield_proxy_10y,
            z_inflation_momentum_minus_be5y=macro_state.inputs.components.get("z_infl_mom_minus_be5y") if macro_state.inputs.components else None,
            z_real_yield_proxy_10y=macro_state.inputs.components.get("z_real_yield_proxy_10y") if macro_state.inputs.components else None,
            use_zscores=True,
            cpi_target=3.0,
            infl_thresh=0.0,
            real_thresh=0.0,
        )
        print("\nMACRO REGIME")
        print(macro_regime)

        # --- Liquidity ---
        liquidity_state = build_funding_state(settings=settings, start_date=start, refresh=refresh)
        print("\nLIQUIDITY (CREDIT + RATES)")
        print(liquidity_state)

        # --- Tariff baskets selection ---
        if baskets.strip().lower() == "all":
            basket_names = list(BASKETS.keys())
        else:
            basket_names = [b.strip() for b in baskets.split(",") if b.strip()]

        unknown = [b for b in basket_names if b not in BASKETS]
        if unknown:
            raise typer.BadParameter(f"Unknown basket(s): {unknown}. Choose from: {list(BASKETS.keys())}")

        # --- Cost proxies (FRED) fetched once ---
        if not settings.FRED_API_KEY:
            raise RuntimeError("Missing FRED_API_KEY in environment / .env")
        fred = FredClient(api_key=settings.FRED_API_KEY)

        frames = []
        for col, sid in DEFAULT_COST_PROXY_SERIES.items():
            df = fred.fetch_series(sid, start_date=start, refresh=refresh)
            df = df.rename(columns={"value": col}).set_index("date")
            frames.append(df[[col]])
        cost_df = pd.concat(frames, axis=1).sort_index().resample("D").ffill()

        # --- Equities (historical closes; default: FMP) fetched once ---
        all_universe = sorted({sym for b in basket_names for sym in BASKETS[b].tickers})
        symbols = sorted(set(all_universe + [benchmark.strip().upper()]))
        px = fetch_equity_daily_closes(settings=settings, symbols=symbols, start=start, refresh=bool(refresh))
        px = px.sort_index().ffill().dropna(how="all")

        print("\nTARIFF / COST-PUSH REGIMES")
        tariff_results = []
        for b in basket_names:
            basket = BASKETS[b]
            print(f"\n[b]{basket.name}[/b] — {basket.description}")
            state = build_tariff_regime_state(
                cost_df=cost_df,
                equity_prices=px,
                universe=basket.tickers,
                benchmark=benchmark,
                basket_name=basket.name,
                start_date=start,
            )
            print(state)
            tariff_results.append(
                {
                    "basket": basket.name,
                    "description": basket.description,
                    "benchmark": benchmark,
                    "state": state,
                }
            )

        if llm:
            from ai_options_trader.llm.regime_summary import llm_regime_summary

            print("\nLLM SUMMARY")
            summary = llm_regime_summary(
                settings=settings,
                macro_state=macro_state,
                macro_regime=macro_regime,
                tariff_regimes=tariff_results,
                model=llm_model.strip() or None,
                temperature=float(llm_temperature),
            )
            print(summary)

    @app.command("regime-features")
    def regime_features(
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
        benchmark: str = typer.Option("XLY", "--benchmark", help="Sector or market benchmark (e.g., XLY, SPY)"),
        baskets: str = typer.Option(
            "all",
            "--baskets",
            help="Comma-separated basket names, or 'all' (see: ai-options-trader tariff baskets)",
        ),
        pretty: bool = typer.Option(True, "--pretty/--compact", help="Pretty-print JSON"),
    ):
        """
        Print a single merged, ML-friendly feature vector (floats only) for:
        - macro regime
        - liquidity regime (credit + rates)
        - tariff/cost-push regimes (per basket + aggregates)
        """
        import json

        from ai_options_trader.data.fred import FredClient
        from ai_options_trader.data.market import fetch_equity_daily_closes
        from ai_options_trader.funding.features import funding_feature_vector
        from ai_options_trader.funding.signals import build_funding_state
        from ai_options_trader.macro.features import macro_feature_vector
        from ai_options_trader.rates.features import rates_feature_vector
        from ai_options_trader.rates.regime import classify_rates_regime
        from ai_options_trader.rates.signals import build_rates_state
        from ai_options_trader.regimes.schema import merge_feature_dicts
        from ai_options_trader.tariff.features import tariff_feature_vector
        from ai_options_trader.tariff.proxies import DEFAULT_COST_PROXY_SERIES
        from ai_options_trader.tariff.signals import build_tariff_regime_state
        from ai_options_trader.tariff.universe import BASKETS
        from ai_options_trader.usd.features import usd_feature_vector
        from ai_options_trader.usd.signals import build_usd_state
        from ai_options_trader.volatility.features import volatility_feature_vector
        from ai_options_trader.volatility.regime import classify_volatility_regime
        from ai_options_trader.volatility.signals import build_volatility_state
        from ai_options_trader.commodities.features import commodities_feature_vector
        from ai_options_trader.commodities.regime import classify_commodities_regime
        from ai_options_trader.commodities.signals import build_commodities_state

        settings = load_settings()

        # Macro
        macro_state = build_macro_state(settings=settings, start_date=start, refresh=refresh)
        macro_regime = classify_macro_regime_from_state(
            cpi_yoy=macro_state.inputs.cpi_yoy,
            payrolls_3m_annualized=macro_state.inputs.payrolls_3m_annualized,
            inflation_momentum_minus_be5y=macro_state.inputs.inflation_momentum_minus_be5y,
            real_yield_proxy_10y=macro_state.inputs.real_yield_proxy_10y,
            z_inflation_momentum_minus_be5y=macro_state.inputs.components.get("z_infl_mom_minus_be5y") if macro_state.inputs.components else None,
            z_real_yield_proxy_10y=macro_state.inputs.components.get("z_real_yield_proxy_10y") if macro_state.inputs.components else None,
            use_zscores=True,
            cpi_target=3.0,
            infl_thresh=0.0,
            real_thresh=0.0,
        )
        macro_vec = macro_feature_vector(macro_state=macro_state, macro_regime=macro_regime)

        # Liquidity
        liq_state = build_funding_state(settings=settings, start_date=start, refresh=refresh)
        liq_vec = funding_feature_vector(liq_state)

        # USD
        usd_state = build_usd_state(settings=settings, start_date=start, refresh=refresh)
        usd_vec = usd_feature_vector(usd_state)

        # Rates
        rates_state = build_rates_state(settings=settings, start_date=start, refresh=refresh)
        rates_regime = classify_rates_regime(rates_state.inputs)
        rates_vec = rates_feature_vector(rates_state, rates_regime)

        # Volatility
        vol_state = build_volatility_state(settings=settings, start_date=start, refresh=refresh)
        vol_regime = classify_volatility_regime(vol_state.inputs)
        vol_vec = volatility_feature_vector(vol_state, vol_regime)

        # Commodities
        commod_state = build_commodities_state(settings=settings, start_date=start, refresh=refresh)
        commod_regime = classify_commodities_regime(commod_state.inputs)
        commod_vec = commodities_feature_vector(commod_state, commod_regime)

        # Tariff basket selection
        if baskets.strip().lower() == "all":
            basket_names = list(BASKETS.keys())
        else:
            basket_names = [b.strip() for b in baskets.split(",") if b.strip()]
        unknown = [b for b in basket_names if b not in BASKETS]
        if unknown:
            raise typer.BadParameter(f"Unknown basket(s): {unknown}. Choose from: {list(BASKETS.keys())}")

        # Cost proxies (FRED) fetched once
        if not settings.FRED_API_KEY:
            raise RuntimeError("Missing FRED_API_KEY in environment / .env")
        fred = FredClient(api_key=settings.FRED_API_KEY)
        frames = []
        for col, sid in DEFAULT_COST_PROXY_SERIES.items():
            df = fred.fetch_series(sid, start_date=start, refresh=refresh)
            df = df.rename(columns={"value": col}).set_index("date")
            frames.append(df[[col]])
        cost_df = pd.concat(frames, axis=1).sort_index().resample("D").ffill()

        # Equities (historical closes; default: FMP) fetched once
        all_universe = sorted({sym for b in basket_names for sym in BASKETS[b].tickers})
        symbols = sorted(set(all_universe + [benchmark.strip().upper()]))
        px = fetch_equity_daily_closes(settings=settings, symbols=symbols, start=start, refresh=bool(refresh))
        px = px.sort_index().ffill().dropna(how="all")

        tariff_states = []
        for b in basket_names:
            basket = BASKETS[b]
            state = build_tariff_regime_state(
                cost_df=cost_df,
                equity_prices=px,
                universe=basket.tickers,
                benchmark=benchmark,
                basket_name=basket.name,
                start_date=start,
            )
            tariff_states.append(state)
        tariff_vec = tariff_feature_vector(tariff_states, asof=tariff_states[-1].asof if tariff_states else macro_state.asof)

        # Merge into one flat mapping (floats only)
        merged = merge_feature_dicts(
            macro_vec.features,
            liq_vec.features,
            usd_vec.features,
            rates_vec.features,
            vol_vec.features,
            commod_vec.features,
            tariff_vec.features,
        )
        out = {"asof": macro_state.asof, **{k: merged[k] for k in sorted(merged.keys())}}

        if pretty:
            print(json.dumps(out, indent=2, sort_keys=False))
        else:
            print(json.dumps(out, separators=(",", ":"), sort_keys=False))


