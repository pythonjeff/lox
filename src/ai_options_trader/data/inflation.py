"""
Wage growth vs inflation (YoY) using FRED.

This script intentionally avoids `pandas_datareader` because it breaks on Python 3.12
(it imports `distutils`, removed from stdlib).

Requirements:
- `FRED_API_KEY` in `.env` (or environment)
- `matplotlib` installed if you want plots

Run:
  PYTHONPATH=src python src/ai_options_trader/data/inflation.py
"""

from __future__ import annotations

import pandas as pd

from ai_options_trader.config import load_settings
from ai_options_trader.data.fred import FredClient


def main(months: int = 12, refresh: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing matplotlib. Install it with: pip install matplotlib") from e

    settings = load_settings()
    if not settings.FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY in environment / .env")

    fred = FredClient(api_key=settings.FRED_API_KEY)

    # Fetch enough history to compute YoY (12-month change) and then display only last `months`.
    # We pull ~ (months + 18) months as a buffer for missing observations.
    calc_months = max(int(months) + 18, 30)
    start = (
        (pd.Timestamp.today().to_period("M").to_timestamp() - pd.DateOffset(months=calc_months))
        .date()
        .isoformat()
    )

    # FRED series:
    # - CPIAUCSL: CPI (All Urban Consumers), SA, index
    # - CES0500000003: Avg Hourly Earnings of All Employees: Total Private, SA, $
    cpi = fred.fetch_series("CPIAUCSL", start_date=start, refresh=refresh).rename(columns={"value": "CPI"})
    wage = fred.fetch_series("CES0500000003", start_date=start, refresh=refresh).rename(
        columns={"value": "AvgHourlyEarnings"}
    )

    # Ensure `date` is a proper datetime index (otherwise matplotlib can render nonsense like 1971-10)
    cpi["date"] = pd.to_datetime(cpi["date"], errors="coerce")
    wage["date"] = pd.to_datetime(wage["date"], errors="coerce")

    cpi = cpi.dropna(subset=["date"]).set_index("date").sort_index()
    wage = wage.dropna(subset=["date"]).set_index("date").sort_index()

    df = cpi.join(wage, how="inner").dropna().sort_index()

    # Year-over-year % change (i.e., "past 12 months" growth)
    df["Inflation_YoY_%"] = df["CPI"].pct_change(12) * 100
    df["WageGrowth_YoY_%"] = df["AvgHourlyEarnings"].pct_change(12) * 100

    # Ensure a clean monthly index for plotting, then show only last `months`.
    monthly = df[["Inflation_YoY_%", "WageGrowth_YoY_%"]].dropna().resample("MS").last()
    plot_df = monthly.tail(int(months))

    fig, ax = plt.subplots(figsize=(11, 6))
    x = plot_df.index.to_pydatetime()
    ax.plot(x, plot_df["Inflation_YoY_%"].to_numpy(), linewidth=2, label="Inflation_YoY_%")
    ax.plot(x, plot_df["WageGrowth_YoY_%"].to_numpy(), linewidth=2, label="WageGrowth_YoY_%")
    ax.set_title(f"Wage Growth vs Inflation (YoY %, last {months} months)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent (%)")
    ax.axhline(0, linewidth=1)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.get_offset_text().set_visible(False)  # hide weird epoch-like offset text
    fig.autofmt_xdate()
    plt.show()

    df["RealWageGrowth_Approx_%"] = df["WageGrowth_YoY_%"] - df["Inflation_YoY_%"]
    monthly2 = (
        df[["Inflation_YoY_%", "WageGrowth_YoY_%", "RealWageGrowth_Approx_%"]]
        .dropna()
        .resample("MS")
        .last()
    )
    plot_df2 = monthly2.tail(int(months))

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    x2 = plot_df2.index.to_pydatetime()
    ax2.plot(x2, plot_df2["Inflation_YoY_%"].to_numpy(), linewidth=2, label="Inflation_YoY_%")
    ax2.plot(x2, plot_df2["WageGrowth_YoY_%"].to_numpy(), linewidth=2, label="WageGrowth_YoY_%")
    ax2.plot(
        x2,
        plot_df2["RealWageGrowth_Approx_%"].to_numpy(),
        linewidth=2,
        label="RealWageGrowth_Approx_%",
    )
    ax2.set_title(f"Wage Growth vs Inflation (YoY %) + Real Wage Growth (Approx.), last {months} months")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Percent (%)")
    ax2.axhline(0, linewidth=1)
    ax2.grid(True, linestyle="--", linewidth=0.6)
    ax2.legend()
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.get_offset_text().set_visible(False)
    fig2.autofmt_xdate()
    plt.show()


if __name__ == "__main__":
    main()