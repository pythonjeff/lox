"""CLI command for the AI bubble regime (v0 — price-only)."""
from __future__ import annotations

from rich import print
from rich.console import Group
from rich.table import Table
from rich.text import Text

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


def _fmt_pct(x: float | None, signed: bool = True) -> str:
    if x is None:
        return "n/a"
    return (f"{x:+.1f}%" if signed else f"{x:.1f}%")


def _ytd_ctx(v: float | None) -> str:
    if v is None:
        return "basket YTD vs SPY"
    if v > 40:
        return "blow-off — basket smoking the market"
    if v > 20:
        return "extreme outperformance — late-stage"
    if v > 10:
        return "strong AI bid"
    if v > 0:
        return "modest excess return"
    if v > -10:
        return "AI trade flat"
    return "AI trade unwinding"


def _breadth_ctx(v: float | None) -> str:
    if v is None:
        return "% of basket above MA"
    if v >= 80:
        return "broad participation — healthy"
    if v >= 60:
        return "decent breadth"
    if v >= 40:
        return "narrowing — leaders carrying"
    return "thin — only a handful holding up"


def _dd_ctx(v: float | None) -> str:
    if v is None:
        return "avg drawdown from 52w high"
    if v > -3:
        return "at/near highs"
    if v > -8:
        return "shallow pullback"
    if v > -15:
        return "meaningful drawdown — cracks forming"
    return "deep drawdown — already correcting"


def _divergence_ctx(v: float | None) -> str:
    if v is None:
        return "chips minus power 3M"
    if v > 25:
        return "power rolling, chips ignoring — early warning"
    if v > 10:
        return "power lagging chips — watch"
    if v > -10:
        return "chips and power in sync"
    return "power ripping while chips lag — unusual"


def _capex_yoy_ctx(v: float | None) -> str:
    if v is None:
        return "TTM capex YoY across spenders"
    if v > 60:
        return "parabolic capex — late-cycle spend"
    if v > 35:
        return "rapid capex ramp"
    if v > 15:
        return "elevated capex"
    if v > 0:
        return "modest capex growth"
    return "capex contracting — trade rolling"


def _capex_ocf_ctx(v: float | None) -> str:
    if v is None:
        return "capex / operating cash flow"
    if v > 95:
        return "capex consuming all OCF — debt-funded"
    if v > 75:
        return "capex stretched vs cash flow"
    if v > 55:
        return "elevated capex intensity"
    return "capex comfortably covered by OCF"


def _vol_ctx(v: float | None) -> str:
    if v is None:
        return "basket vs SPY 20d vol"
    if v > 2.5:
        return "vol blow-off — overheated"
    if v > 1.8:
        return "elevated vol — frothy"
    if v > 1.3:
        return "normal AI-vs-market vol"
    return "vol compressed — complacent"


def ai_snapshot(*, llm: bool = False, ticker: str = "", refresh: bool = False) -> None:
    """Entry point for `lox regime ai`."""
    settings = load_settings()

    from lox.ai.signals import compute_ai_signals, BASKET
    from lox.ai.regime import classify_ai

    sig = compute_ai_signals(settings=settings, refresh=refresh)
    cx = sig.capex
    result = classify_ai(
        basket_ytd_excess=sig.basket_ytd_excess,
        basket_3m_excess=sig.basket_3m_excess,
        pct_above_50dma=sig.pct_above_50dma,
        pct_above_200dma=sig.pct_above_200dma,
        avg_drawdown_from_52w=sig.avg_drawdown_from_52w,
        chip_vs_power_spread=sig.chip_vs_power_spread,
        vol_ratio_vs_spy=sig.vol_ratio_vs_spy,
        hyperscaler_capex_yoy=cx.aggregate_ttm_capex_yoy_pct if cx else None,
        capex_to_ocf_pct=cx.aggregate_capex_to_ocf_pct if cx else None,
    )

    metrics = [
        {"name": "Basket YTD vs SPY",
         "value": _fmt_pct(sig.basket_ytd_excess) + "pp" if sig.basket_ytd_excess is not None else "n/a",
         "context": _ytd_ctx(sig.basket_ytd_excess)},
        {"name": "Basket 3M vs SPY",
         "value": _fmt_pct(sig.basket_3m_excess) + "pp" if sig.basket_3m_excess is not None else "n/a",
         "context": _ytd_ctx(sig.basket_3m_excess)},
        {"name": "% above 50dma",
         "value": _fmt_pct(sig.pct_above_50dma, signed=False),
         "context": _breadth_ctx(sig.pct_above_50dma)},
        {"name": "% above 200dma",
         "value": _fmt_pct(sig.pct_above_200dma, signed=False),
         "context": _breadth_ctx(sig.pct_above_200dma)},
        {"name": "Avg DD from 52w high",
         "value": _fmt_pct(sig.avg_drawdown_from_52w),
         "context": _dd_ctx(sig.avg_drawdown_from_52w)},
        {"name": "Chips − Power (3M)",
         "value": (_fmt_pct(sig.chip_vs_power_spread) + "pp") if sig.chip_vs_power_spread is not None else "n/a",
         "context": _divergence_ctx(sig.chip_vs_power_spread)},
        {"name": "Basket/SPY 20d vol",
         "value": (f"{sig.vol_ratio_vs_spy:.2f}x" if sig.vol_ratio_vs_spy is not None else "n/a"),
         "context": _vol_ctx(sig.vol_ratio_vs_spy)},
    ]

    from lox.regimes.trend import get_domain_trend
    trend = get_domain_trend("ai", result.score, result.label)

    print(render_regime_panel(
        domain="AI Bubble",
        asof=sig.asof,
        regime_label=result.label,
        score=result.score,
        description=result.description,
        metrics=metrics,
        trend=trend,
    ))

    # ── Sub-group returns ──────────────────────────────────────────────
    gt = Table(title="Sub-group returns", show_header=True, header_style="bold dim",
               box=None, padding=(0, 2), title_style="bold")
    gt.add_column("Group", style="bold")
    gt.add_column("1M", justify="right")
    gt.add_column("3M", justify="right")
    gt.add_column("YTD", justify="right")
    label_map = {"chips": "Chips", "hyperscale": "Hyperscalers", "power": "AI Power"}
    for g, label in label_map.items():
        r = sig.group_returns.get(g, {})
        gt.add_row(
            label,
            _fmt_pct(r.get("1m")),
            _fmt_pct(r.get("3m")),
            _fmt_pct(r.get("ytd")),
        )
    print(Text(""))
    print(gt)

    # ── Per-name table ─────────────────────────────────────────────────
    pt = Table(title="Per-name", show_header=True, header_style="bold dim",
               box=None, padding=(0, 2), title_style="bold")
    pt.add_column("Ticker", style="bold")
    pt.add_column("Group", style="dim")
    pt.add_column("3M", justify="right")
    pt.add_column("YTD", justify="right")
    pt.add_column("DD 52w", justify="right")
    pt.add_column("> 50d", justify="center")
    pt.add_column("> 200d", justify="center")
    order = {g: i for i, g in enumerate(BASKET.keys())}
    rows = sorted(sig.per_name, key=lambda r: (order.get(r["group"], 99), r["ticker"]))
    for r in rows:
        def _yn(b):
            if b is None:
                return "—"
            return "[green]✓[/green]" if b else "[red]✗[/red]"
        pt.add_row(
            r["ticker"],
            label_map.get(r["group"], r["group"]),
            _fmt_pct(r.get("3m")),
            _fmt_pct(r.get("ytd")),
            _fmt_pct(r.get("dd_52w")),
            _yn(r.get("above_50")),
            _yn(r.get("above_200")),
        )
    print(Text(""))
    print(pt)

    # ── Hyperscaler capex panel (v1) ───────────────────────────────────
    if sig.capex and sig.capex.per_name:
        cx = sig.capex
        header_bits = []
        if cx.aggregate_ttm_capex_bn is not None:
            header_bits.append(f"TTM capex ${cx.aggregate_ttm_capex_bn:.0f}B")
        if cx.aggregate_ttm_capex_yoy_pct is not None:
            header_bits.append(f"YoY {cx.aggregate_ttm_capex_yoy_pct:+.0f}%  — {_capex_yoy_ctx(cx.aggregate_ttm_capex_yoy_pct)}")
        if cx.aggregate_capex_to_ocf_pct is not None:
            header_bits.append(f"capex/OCF {cx.aggregate_capex_to_ocf_pct:.0f}%  — {_capex_ocf_ctx(cx.aggregate_capex_to_ocf_pct)}")
        ct = Table(title="Hyperscaler capex (quarterly, FMP)", show_header=True,
                   header_style="bold dim", box=None, padding=(0, 2),
                   title_style="bold",
                   caption=" | ".join(header_bits) if header_bits else None,
                   caption_style="dim")
        ct.add_column("Ticker", style="bold")
        ct.add_column("Period", style="dim")
        ct.add_column("Q capex ($B)", justify="right")
        ct.add_column("YoY", justify="right")
        ct.add_column("Capex/OCF (TTM)", justify="right")
        ct.add_column("TTM capex ($B)", justify="right")
        for s in sorted(cx.per_name, key=lambda r: -(r.ttm_capex_bn or 0)):
            ct.add_row(
                s.symbol,
                s.latest_period or "—",
                f"{s.latest_capex_bn:.1f}" if s.latest_capex_bn is not None else "—",
                _fmt_pct(s.capex_yoy_pct),
                f"{s.capex_to_ocf_pct:.0f}%" if s.capex_to_ocf_pct is not None else "—",
                f"{s.ttm_capex_bn:.1f}" if s.ttm_capex_bn is not None else "—",
            )
        print(Text(""))
        print(ct)

    # ── News pulse (display-only, v1) ──────────────────────────────────
    if sig.news and sig.news.total_articles:
        n = sig.news
        print(Text(""))
        summary = (f"[bold]News pulse[/bold] (5d, {n.total_articles} articles)  "
                   f"bubble-language hits: [yellow]{n.bubble_hits}[/yellow]   "
                   f"cracks-language hits: [red]{n.cracks_hits}[/red]")
        print(Text.from_markup(summary))
        if n.top_headlines:
            for h in n.top_headlines[:5]:
                tag_str = " ".join(f"[{('red' if t == 'cracks' else 'yellow')}]{t}[/]" for t in h.get("tags", []))
                print(Text.from_markup(f"  • {tag_str}  [dim]{h.get('source', '')}[/dim]  {h.get('title', '')}"))

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="ai",
            snapshot={
                "ytd_excess": sig.basket_ytd_excess,
                "3m_excess": sig.basket_3m_excess,
                "breadth_50": sig.pct_above_50dma,
                "breadth_200": sig.pct_above_200dma,
                "avg_dd": sig.avg_drawdown_from_52w,
                "chip_minus_power_3m": sig.chip_vs_power_spread,
                "vol_ratio": sig.vol_ratio_vs_spy,
                "groups": sig.group_returns,
            },
            regime_label=result.label,
            regime_description=result.description,
            ticker=ticker,
        )
