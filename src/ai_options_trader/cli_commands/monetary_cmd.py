from __future__ import annotations

import typer
from rich import print
from rich.panel import Panel

from ai_options_trader.config import load_settings
from ai_options_trader.monetary.models import MonetaryInputs
from ai_options_trader.monetary.regime import classify_monetary_regime
from ai_options_trader.monetary.signals import build_monetary_page_data
from ai_options_trader.utils.formatting import fmt_usd_from_millions


def register(monetary_app: typer.Typer) -> None:
    def _ctx_z(z: float | None, *, low: float = -0.75, high: float = 0.75) -> str:
        if not isinstance(z, (int, float)):
            return "n/a"
        v = float(z)
        if v <= low:
            return f"{v:+.2f} → Lower vs recent history (not a scarcity signal by itself)"
        if v >= high:
            return f"{v:+.2f} → Higher vs recent history"
        return f"{v:+.2f} → Around recent average"

    def _ctx_qt(z_qt: float | None) -> str:
        # z-score of 13w change in Fed assets: negative means shrinking.
        if not isinstance(z_qt, (int, float)):
            return "n/a"
        v = float(z_qt)
        if v <= -0.75:
            return f"{v:+.2f} → Fast shrink (QT pressure)"
        if v >= 0.75:
            return f"{v:+.2f} → Balance sheet expanding (liquidity support / QT offset)"
        return f"{v:+.2f} → Normal pace"

    @monetary_app.command("snapshot")
    def monetary_snapshot(
        lookback_years: int = typer.Option(5, "--lookback-years", help="How many years of history to load."),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
    ):
        """
        Print monetary regime snapshot (lean MVP).

        MVP series:
        1) EFFR (DFF)
        2) Total reserves (TOTRESNS)
        3) Fed balance sheet total assets and Δ (WALCL)
        4) ON RRP usage (RRPONTSYD)
        """
        settings = load_settings()
        d = build_monetary_page_data(settings=settings, lookback_years=lookback_years, refresh=refresh)

        effr = d.get("effr")
        effr_disp = f"{float(effr):.2f}%" if isinstance(effr, (int, float)) else "n/a"

        res = d.get("reserves") if isinstance(d.get("reserves"), dict) else {}
        res_level = fmt_usd_from_millions(res.get("level"))
        res_chg_13w = fmt_usd_from_millions(res.get("chg_13w"))
        res_z = res.get("z_level")
        res_pct_gdp = res.get("pct_gdp")
        res_pct_gdp_disp = f"{float(res_pct_gdp):.1f}%" if isinstance(res_pct_gdp, (int, float)) else "n/a"

        fa = d.get("fed_assets") if isinstance(d.get("fed_assets"), dict) else {}
        fa_level = fmt_usd_from_millions(fa.get("level"))
        fa_chg_13w = fmt_usd_from_millions(fa.get("chg_13w"))
        fa_z = fa.get("z_chg_13w")

        rrp = d.get("on_rrp") if isinstance(d.get("on_rrp"), dict) else {}
        rrp_level = fmt_usd_from_millions(rrp.get("level"))
        rrp_chg_13w = fmt_usd_from_millions(rrp.get("chg_13w"))
        rrp_z = rrp.get("z_level")

        # Classifier uses z context only (lean).
        regime = classify_monetary_regime(
            MonetaryInputs(
                z_total_reserves=float(res_z) if isinstance(res_z, (int, float)) else None,
                z_on_rrp=float(rrp_z) if isinstance(rrp_z, (int, float)) else None,
                z_fed_assets_chg_13w=float(fa_z) if isinstance(fa_z, (int, float)) else None,
            )
        )

        series = d.get("series_used") if isinstance(d.get("series_used"), list) else []
        series_disp = ", ".join(str(x) for x in series) if series else "n/a"

        body = "\n".join(
            [
                f"As of: [bold]{d.get('asof','n/a')}[/bold]",
                f"EFFR (DFF): [bold]{effr_disp}[/bold]",
                "Total reserves (TOTRESNS):",
                f"  Level: [bold]{res_level}[/bold]  [dim](z={_ctx_z(float(res_z)) if isinstance(res_z,(int,float)) else 'n/a'})[/dim]",
                f"  Reserves / GDP: [bold]{res_pct_gdp_disp}[/bold]  [dim](normalization anchor; helps avoid 'lower than QE peak' false alarms)[/dim]",
                f"  Δ 13-week: [bold]{res_chg_13w}[/bold]",
                "Fed balance sheet (WALCL):",
                f"  Level: [bold]{fa_level}[/bold]",
                f"  Δ 13-week: [bold]{fa_chg_13w}[/bold]  [dim](z={_ctx_qt(float(fa_z)) if isinstance(fa_z,(int,float)) else 'n/a'})[/dim]",
                f"  [dim]Qualifier: reserves Δ13w={res_chg_13w}; ON RRP Δ13w={rrp_chg_13w} (TGA/funding regime can still offset; not included yet).[/dim]",
                "ON RRP usage (RRPONTSYD):",
                f"  Level: [bold]{rrp_level}[/bold]  [dim](z={_ctx_z(float(rrp_z)) if isinstance(rrp_z,(int,float)) else 'n/a'})[/dim]",
                f"  Δ 13-week: [bold]{rrp_chg_13w}[/bold]",
                f"Regime: [bold]{regime.label or regime.name}[/bold]",
                f"Answer: {regime.description}",
                f"History loaded: {d.get('lookback_years', lookback_years)}y",
                f"Series (FRED): [dim]{series_disp}[/dim]",
            ]
        )
        print(Panel.fit(body, title="US Monetary (MVP)", border_style="cyan"))


