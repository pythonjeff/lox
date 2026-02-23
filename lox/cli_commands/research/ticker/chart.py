"""Ticker charting — institutional-grade candlestick + volume + RSI chart."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def generate_chart(symbol: str, price_data: dict, technicals: dict) -> str | None:
    """Generate institutional-grade price chart (candlestick + volume + RSI)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.ticker as mticker
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        import numpy as np
        import pandas as pd
        from datetime import datetime
        import tempfile
        from pathlib import Path

        df = technicals.get("df")
        if df is None or df.empty:
            return None

        # Use last 6 months of trading days
        df = df.tail(126).copy().reset_index(drop=True)

        if len(df) < 20:
            return None

        # ── Compute indicators ─────────────────────────────────────────────
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        opens = df["open"].values.astype(float)
        volumes = df["volume"].values.astype(float)

        # Moving averages
        df["sma20"] = df["close"].rolling(20).mean()
        df["sma50"] = df["close"].rolling(50).mean()

        # Bollinger Bands (20, 2) — subtle envelope only
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["sma20"] + 2 * bb_std
        df["bb_lower"] = df["sma20"] - 2 * bb_std

        # RSI (14-day, Wilder smoothing)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Volume MA for reference
        df["vol_ma20"] = df["volume"].rolling(20).mean()

        # ── Color palette (institutional dark) ─────────────────────────────
        BG = "#0a0e17"
        PANEL_BG = "#0f1318"
        GRID = "#1a2030"
        TEXT = "#c9d1d9"
        TEXT_DIM = "#6e7681"
        GREEN = "#00d26a"
        RED = "#ff4757"
        BLUE = "#3b82f6"
        CYAN = "#22d3ee"
        ORANGE = "#f59e0b"
        WHITE = "#e6edf3"

        # ── Figure layout: 3 panels (price, volume, RSI) ──────────────────
        fig = plt.figure(figsize=(14, 9), facecolor=BG)
        gs = fig.add_gridspec(
            3, 1,
            height_ratios=[5, 1.2, 1.5],
            hspace=0.03,
            left=0.07, right=0.93, top=0.92, bottom=0.06,
        )
        ax_price = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
        ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)

        for ax in (ax_price, ax_vol, ax_rsi):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_DIM, labelsize=9)
            ax.grid(True, alpha=0.15, color=GRID, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_color(GRID)
                spine.set_linewidth(0.5)

        dates = mdates.date2num(df["date"])

        # ── Panel 1: Candlestick + Overlays ────────────────────────────────
        up = closes >= opens
        dn = ~up
        candle_width = 0.6

        # Candle bodies
        ax_price.bar(
            dates[up], (closes - opens)[up], candle_width,
            bottom=opens[up], color=GREEN, alpha=0.9, linewidth=0, zorder=3,
        )
        ax_price.bar(
            dates[dn], (opens - closes)[dn], candle_width,
            bottom=closes[dn], color=RED, alpha=0.9, linewidth=0, zorder=3,
        )

        # Candle wicks
        ax_price.vlines(
            dates[up], lows[up], highs[up],
            color=GREEN, linewidth=0.7, zorder=2,
        )
        ax_price.vlines(
            dates[dn], lows[dn], highs[dn],
            color=RED, linewidth=0.7, zorder=2,
        )

        # Moving averages
        ax_price.plot(dates, df["sma20"], color=BLUE, linewidth=1.3, alpha=0.85, label="SMA 20")
        if len(df) >= 50:
            ax_price.plot(dates, df["sma50"], color=ORANGE, linewidth=1.3, alpha=0.85, label="SMA 50")

        # Bollinger Bands — subtle envelope (no edge lines, just fill)
        bb_valid = df["bb_upper"].notna()
        if bb_valid.any():
            ax_price.fill_between(
                dates[bb_valid],
                df["bb_upper"][bb_valid],
                df["bb_lower"][bb_valid],
                color=BLUE, alpha=0.06, label="BBand (2σ)",
            )

        # Support / Resistance levels
        support = technicals.get("support")
        resistance = technicals.get("resistance")
        if support:
            ax_price.axhline(y=support, color=GREEN, linestyle=":", linewidth=0.8, alpha=0.6, zorder=1)
            ax_price.text(
                dates[-1] + 1, support, f" S ${support:.2f}",
                color=GREEN, fontsize=8, va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=BG, edgecolor=GREEN, alpha=0.8, linewidth=0.5),
            )
        if resistance:
            ax_price.axhline(y=resistance, color=RED, linestyle=":", linewidth=0.8, alpha=0.6, zorder=1)
            ax_price.text(
                dates[-1] + 1, resistance, f" R ${resistance:.2f}",
                color=RED, fontsize=8, va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=BG, edgecolor=RED, alpha=0.8, linewidth=0.5),
            )

        # Last price annotation
        last_price = closes[-1]
        last_color = GREEN if closes[-1] >= opens[-1] else RED
        ax_price.axhline(y=last_price, color=last_color, linewidth=0.5, alpha=0.4, linestyle="-", zorder=1)
        ax_price.text(
            dates[-1] + 1.5, last_price, f" ${last_price:.2f}",
            color=last_color, fontsize=9, va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=last_color, edgecolor="none", alpha=0.15),
        )

        # Y-axis: tight to price range with 2% padding
        price_min = float(df["low"].min())
        price_max = float(df["high"].max())
        price_range = price_max - price_min
        pad = price_range * 0.05
        ax_price.set_ylim(price_min - pad, price_max + pad)
        ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

        # Legend — compact, out of the way
        legend_elements = [
            Line2D([0], [0], color=BLUE, lw=1.3, label="SMA 20"),
            Line2D([0], [0], color=ORANGE, lw=1.3, label="SMA 50"),
            mpatches.Patch(color=BLUE, alpha=0.15, label="BBand 2σ"),
        ]
        ax_price.legend(
            handles=legend_elements, loc="upper left", fontsize=7.5,
            framealpha=0.7, facecolor=BG, edgecolor=GRID, labelcolor=TEXT_DIM,
            ncol=3, columnspacing=1.2, handlelength=1.5,
        )
        ax_price.tick_params(labelbottom=False)
        ax_price.set_ylabel("Price", fontsize=9, color=TEXT_DIM, labelpad=8)

        # ── Panel 2: Volume ────────────────────────────────────────────────
        vol_colors = np.where(up, GREEN, RED)
        ax_vol.bar(dates, volumes, candle_width, color=vol_colors, alpha=0.6, zorder=2)
        vol_ma = df["vol_ma20"]
        vol_ma_valid = vol_ma.notna()
        if vol_ma_valid.any():
            ax_vol.plot(dates[vol_ma_valid], vol_ma[vol_ma_valid], color=BLUE, linewidth=1, alpha=0.7)

        ax_vol.set_ylabel("Vol", fontsize=8, color=TEXT_DIM, labelpad=8)
        ax_vol.tick_params(labelbottom=False)

        # Format volume axis (M/B)
        def _vol_fmt(x, _):
            if x >= 1e9:
                return f"{x / 1e9:.1f}B"
            if x >= 1e6:
                return f"{x / 1e6:.0f}M"
            if x >= 1e3:
                return f"{x / 1e3:.0f}K"
            return str(int(x))
        ax_vol.yaxis.set_major_formatter(mticker.FuncFormatter(_vol_fmt))

        # ── Panel 3: RSI ───────────────────────────────────────────────────
        rsi_valid = df["rsi"].notna()
        if rsi_valid.any():
            ax_rsi.plot(dates[rsi_valid], df["rsi"][rsi_valid], color=CYAN, linewidth=1.2)
            ax_rsi.axhline(70, color=RED, linewidth=0.6, alpha=0.5, linestyle="--")
            ax_rsi.axhline(30, color=GREEN, linewidth=0.6, alpha=0.5, linestyle="--")
            ax_rsi.axhline(50, color=TEXT_DIM, linewidth=0.4, alpha=0.3, linestyle="-")

            # Shade overbought/oversold zones
            ax_rsi.fill_between(
                dates[rsi_valid], 70, df["rsi"][rsi_valid],
                where=df["rsi"][rsi_valid] >= 70,
                color=RED, alpha=0.1, interpolate=True,
            )
            ax_rsi.fill_between(
                dates[rsi_valid], 30, df["rsi"][rsi_valid],
                where=df["rsi"][rsi_valid] <= 30,
                color=GREEN, alpha=0.1, interpolate=True,
            )

            # Current RSI value label
            rsi_last = float(df["rsi"].iloc[-1]) if pd.notna(df["rsi"].iloc[-1]) else None
            if rsi_last is not None:
                rsi_color = RED if rsi_last >= 70 else GREEN if rsi_last <= 30 else CYAN
                ax_rsi.text(
                    dates[-1] + 1, rsi_last, f" {rsi_last:.1f}",
                    color=rsi_color, fontsize=8, va="center", fontweight="bold",
                )

        ax_rsi.set_ylim(10, 90)
        ax_rsi.set_ylabel("RSI", fontsize=8, color=TEXT_DIM, labelpad=8)
        ax_rsi.set_yticks([30, 50, 70])

        # X-axis formatting
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax_rsi.xaxis.set_major_locator(mdates.MonthLocator())
        ax_rsi.tick_params(axis="x", labelsize=8, rotation=0)

        # ── Title bar ─────────────────────────────────────────────────────
        current = technicals.get("current", 0)
        change_pct = price_data.get("quote", {}).get("changesPercentage", 0)
        change_val = price_data.get("quote", {}).get("change", 0)
        chg_color = GREEN if change_pct >= 0 else RED
        chg_sign = "+" if change_pct >= 0 else ""

        # Main title
        fig.text(
            0.07, 0.96, symbol,
            fontsize=20, fontweight="bold", color=WHITE,
            fontfamily="monospace",
        )
        fig.text(
            0.07 + len(symbol) * 0.018, 0.96,
            f"   ${current:,.2f}   {chg_sign}{change_val:,.2f} ({chg_sign}{change_pct:.2f}%)",
            fontsize=13, color=chg_color,
            fontfamily="monospace",
        )

        # Info bar (right-aligned)
        rsi_val = technicals.get("rsi")
        vol_val = technicals.get("volatility")
        info_parts = []
        if rsi_val is not None:
            info_parts.append(f"RSI {rsi_val:.1f}")
        if vol_val is not None:
            info_parts.append(f"Vol {vol_val:.1f}%")
        info_parts.append(datetime.now().strftime("%b %d, %Y"))
        fig.text(
            0.93, 0.96, "  |  ".join(info_parts),
            fontsize=9, color=TEXT_DIM, ha="right",
            fontfamily="monospace",
        )

        # Branding
        fig.text(
            0.93, 0.015, "LOX FUND",
            fontsize=8, color=TEXT_DIM, ha="right", fontweight="bold",
            fontfamily="monospace", alpha=0.5,
        )

        # ── Save ───────────────────────────────────────────────────────────
        output_dir = Path(tempfile.gettempdir()) / "lox_charts"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{symbol}_{timestamp}.png"

        fig.savefig(
            output_path, dpi=180, facecolor=BG, edgecolor="none",
            bbox_inches="tight", pad_inches=0.15,
        )
        plt.close(fig)

        return str(output_path)
    except Exception:
        logger.debug("Failed to generate chart for %s", symbol, exc_info=True)
        return None


def open_chart(path: str):
    """Open chart in system viewer."""
    import subprocess
    import sys

    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        elif sys.platform == "win32":
            subprocess.run(["start", path], shell=True, check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception:
        logger.debug("Failed to open chart at %s", path, exc_info=True)
