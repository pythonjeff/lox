"""
Historical bubble reference points for context comparisons.

Each entry holds peak-ish readings near the headline top.  Values are
approximate (sources vary) and chosen to be the figures commonly cited in
the literature so readers can do a sanity check.  All ratios are
unit-consistent with the live metrics (Buffett indicator = total mkt-cap /
GDP × 100; top-10 share = % of cap-weighted index; margin debt/GDP × 100).

Sources / notes:

  1929 — Great Depression peak (Sept 1929)
    Buffett indicator: ~75-80% (Goldsmith estimates; Z.1 series begins 1945)
    Top-10 share: ~30% (Dow/Cowles index; Steel, GM, Standard Oil dominated)
    Margin/GDP: ~9% (broker loans ~$8.5B vs GNP ~$103B; the legendary one)

  2000 — Dot-com peak (March 2000)
    Buffett indicator: ~145%
    Top-10 share: ~25-27% (MSFT, CSCO, INTC, GE, etc.)
    Margin/GDP: ~2.6% (peak margin $278B; GDP ~$10.6T)

  2007 — Pre-GFC peak (Oct 2007)
    Buffett indicator: ~105%
    Top-10 share: ~20%
    Margin/GDP: ~2.7% (margin $381B; GDP ~$14.5T)

  2021 — Post-COVID peak (Nov 2021)
    Buffett indicator: ~200%
    Top-10 share: ~30%
    Margin/GDP: ~3.6% (margin $936B; GDP ~$26T — record nominal)

The 2021 row is the most relevant base rate today — the only modern peak
with similar concentration and a real-money-money margin ramp.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BubblePeak:
    name: str
    year: int
    buffett_pct: float | None
    top10_share_pct: float | None
    margin_to_gdp_pct: float | None
    note: str


HISTORICAL_BUBBLES: tuple[BubblePeak, ...] = (
    BubblePeak("1929 — Great Depression",   1929,  75.0, 30.0, 9.0,
               "10x margin retail mania"),
    BubblePeak("2000 — Dot-com",            2000, 145.0, 26.0, 2.6,
               "TMT-led, no broad leverage"),
    BubblePeak("2007 — Pre-GFC",            2007, 105.0, 20.0, 2.7,
               "valuation tame, credit was the bubble"),
    BubblePeak("2021 — Post-COVID",         2021, 200.0, 30.0, 3.6,
               "broad mania, record margin"),
)
