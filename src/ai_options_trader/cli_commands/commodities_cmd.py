from __future__ import annotations

import typer
from rich import print
from rich.panel import Panel


def register(commod_app: typer.Typer) -> None:
    @commod_app.command("snapshot")
    def snapshot(
        start: str = typer.Option("2011-01-01", "--start"),
        refresh: bool = typer.Option(False, "--refresh"),
    ):
        """
        Commodities snapshot: oil/gold (+ broad commodities index when available).
        """
        import numpy as np
        import pandas as pd

        from ai_options_trader.config import load_settings
        from ai_options_trader.data.market import fetch_equity_daily_closes
        from ai_options_trader.commodities.signals import build_commodities_state
        from ai_options_trader.commodities.regime import classify_commodities_regime

        settings = load_settings()
        state = build_commodities_state(settings=settings, start_date=start, refresh=refresh)
        regime = classify_commodities_regime(state.inputs)

        # Daily tradable proxies for trend context (more reliable than low-frequency FRED series)
        proxy = {
            "Gold (proxy)": "GLDM",
            "Broad commod (proxy)": "DBC",
            "Copper (proxy)": "CPER",
            "Oil (proxy)": "USO",
        }

        trend_lines: list[str] = []
        try:
            api_key = settings.alpaca_data_key or settings.alpaca_api_key
            api_secret = settings.alpaca_data_secret or settings.alpaca_api_secret
            px = fetch_equity_daily_closes(api_key=api_key, api_secret=api_secret, symbols=list(proxy.values()), start=start)
            px = px.sort_index().ffill()

            def _ret(s: pd.Series, d: int) -> float | None:
                s = pd.to_numeric(s, errors="coerce").dropna()
                if s.shape[0] <= d:
                    return None
                v = (s.iloc[-1] / s.iloc[-1 - d] - 1.0) * 100.0
                return float(v) if np.isfinite(v) else None

            def _rv(s: pd.Series, d: int = 20) -> float | None:
                s = pd.to_numeric(s, errors="coerce").dropna()
                if s.shape[0] <= d + 2:
                    return None
                r = s.pct_change().dropna()
                v = r.tail(d).std(ddof=0) * np.sqrt(252) * 100.0
                return float(v) if np.isfinite(v) else None

            for label, sym in proxy.items():
                if sym not in px.columns:
                    continue
                last = float(pd.to_numeric(px[sym], errors="coerce").dropna().iloc[-1])
                r20 = _ret(px[sym], 20)
                r60 = _ret(px[sym], 60)
                r252 = _ret(px[sym], 252)
                rv20 = _rv(px[sym], 20)
                trend_lines.append(
                    f"- [b]{label}[/b] {sym}: px={last:.2f}  20d={r20:+.1f}%  60d={r60:+.1f}%  1y={r252:+.1f}%  rv20≈{(rv20 if rv20 is not None else float('nan')):.1f}%"
                )
        except Exception:
            # Keep snapshot usable even if Alpaca data fetch fails.
            trend_lines = []

        body = (
            f"[b]Regime:[/b] {regime.label}\n"
            f"[b]WTI:[/b] {state.inputs.wti}  [b]20d%:[/b] {state.inputs.wti_ret_20d_pct}  [b]Z:[/b] {state.inputs.z_wti_ret_20d}\n"
            f"[b]Gold:[/b] {state.inputs.gold}  [b]20d%:[/b] {state.inputs.gold_ret_20d_pct}  [b]Z:[/b] {state.inputs.z_gold_ret_20d}\n"
            f"[b]Copper:[/b] {state.inputs.copper}  [b]60d%:[/b] {state.inputs.copper_ret_60d_pct}  [b]Z:[/b] {state.inputs.z_copper_ret_60d}\n"
            f"[b]Broad idx:[/b] {state.inputs.broad_index}  [b]60d%:[/b] {state.inputs.broad_ret_60d_pct}  [b]Z:[/b] {state.inputs.z_broad_ret_60d}\n"
            f"[b]Pressure score:[/b] {state.inputs.commodity_pressure_score}\n"
            f"[b]Energy shock:[/b] {state.inputs.energy_shock}\n\n"
            f"[b]Metals impulse:[/b] {state.inputs.metals_impulse}\n\n"
        )
        if trend_lines:
            body += "[b]Proxy price trends (daily):[/b]\n" + "\n".join(trend_lines) + "\n\n"
        body += f"[dim]{regime.description}[/dim]"

        print(Panel(body, title="Commodities snapshot", expand=False))


