"""
Regime Pillars - Individual economic/market factors.

Each pillar provides:
1. Raw metrics with context
2. Z-scores for ML
3. Composite score
4. Regime classification
5. Deep-dive analysis

Available pillars:
- inflation: CPI, PCE, breakevens, stickiness
- growth: GDP, payrolls, unemployment, leading indicators
- rates: Yields, curve, real rates, term premium
- liquidity: Fed funds, SOFR, RRP, TGA, reserves
- volatility: VIX, term structure, skew, realized vol
- credit: IG/HY spreads, default rates
- housing: Mortgage spreads, home prices, REIT performance
- commodities: Oil, gold, copper, broad index
"""
from lox.regimes.pillars.inflation import InflationPillar
from lox.regimes.pillars.growth import GrowthPillar
from lox.regimes.pillars.liquidity import LiquidityPillar
from lox.regimes.pillars.volatility import VolatilityPillar

__all__ = [
    "InflationPillar",
    "GrowthPillar",
    "LiquidityPillar",
    "VolatilityPillar",
]
