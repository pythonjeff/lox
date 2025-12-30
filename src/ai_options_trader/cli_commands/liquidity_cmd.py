from __future__ import annotations

import typer
from rich import print

from ai_options_trader.config import load_settings


def register(liquidity_app: typer.Typer) -> None:
    @liquidity_app.command("snapshot")
    def liquidity_snapshot(
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
        features: bool = typer.Option(False, "--features", help="Print ML-friendly scalar feature vector too"),
    ):
        """
        Compute liquidity regime snapshot (corporate credit + government bond market).
        """
        settings = load_settings()
        from ai_options_trader.liquidity.signals import build_liquidity_state

        state = build_liquidity_state(settings=settings, start_date=start, refresh=refresh)
        print(state)

        if features:
            from ai_options_trader.liquidity.features import liquidity_feature_vector
            import json

            vec = liquidity_feature_vector(state)
            out = {"asof": vec.asof, **{k: vec.features[k] for k in sorted(vec.features.keys())}}
            print("\nLIQUIDITY FEATURES (floats)")
            print(json.dumps(out, indent=2))


