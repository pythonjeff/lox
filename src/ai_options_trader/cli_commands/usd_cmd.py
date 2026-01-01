from __future__ import annotations

import typer
from rich import print

from ai_options_trader.config import load_settings


def register(usd_app: typer.Typer) -> None:
    @usd_app.command("snapshot")
    def usd_snapshot(
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
        features: bool = typer.Option(False, "--features", help="Print ML-friendly scalar feature vector too"),
    ):
        """Compute USD strength/weakness regime snapshot."""
        settings = load_settings()

        from ai_options_trader.usd.signals import build_usd_state

        state = build_usd_state(settings=settings, start_date=start, refresh=refresh)
        print(state)

        if features:
            import json
            from ai_options_trader.usd.features import usd_feature_vector

            vec = usd_feature_vector(state)
            out = {"asof": vec.asof, **{k: vec.features[k] for k in sorted(vec.features.keys())}}
            print("\nUSD FEATURES (floats)")
            print(json.dumps(out, indent=2))

    @usd_app.command("outlook")
    def usd_outlook(
        year: int = typer.Option(2026, "--year", help="Focus year for the outlook (e.g., 2026)"),
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
        with_context: bool = typer.Option(True, "--with-context/--usd-only", help="Include macro+liquidity context"),
        llm_model: str = typer.Option("", "--llm-model", help="Override OPENAI_MODEL (optional)"),
        llm_temperature: float = typer.Option(0.2, "--llm-temperature", help="LLM temperature (0..2)"),
    ):
        """Ask an LLM for a scenario-style USD outlook (grounded in the USD regime snapshot)."""
        settings = load_settings()

        from ai_options_trader.usd.signals import build_usd_state
        from ai_options_trader.llm.usd_outlook import llm_usd_outlook

        usd_state = build_usd_state(settings=settings, start_date=start, refresh=refresh)

        macro_state = None
        liquidity_state = None
        if with_context:
            from ai_options_trader.macro.signals import build_macro_state
            from ai_options_trader.liquidity.signals import build_liquidity_state

            macro_state = build_macro_state(settings=settings, start_date=start, refresh=refresh)
            liquidity_state = build_liquidity_state(settings=settings, start_date=start, refresh=refresh)

        text = llm_usd_outlook(
            settings=settings,
            usd_state=usd_state,
            macro_state=macro_state,
            liquidity_state=liquidity_state,
            year=int(year),
            model=llm_model.strip() or None,
            temperature=float(llm_temperature),
        )
        print(text)


