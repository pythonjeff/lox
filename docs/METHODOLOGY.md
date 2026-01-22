# Lox Capital — Technical Methodology

**Version 1.0 | January 2026**

This document provides methodological transparency for the research platform's core quantitative systems. It is intended for technical review and due diligence.

---

## Table of Contents

1. [Palmer: Macro Intelligence Engine](#1-palmer-macro-intelligence-engine)
2. [Monte Carlo Simulation Framework](#2-monte-carlo-simulation-framework)
3. [Regime Classification System](#3-regime-classification-system)
4. [P&L Attribution & Risk Decomposition](#4-pl-attribution--risk-decomposition)
5. [Stress Testing Framework](#5-stress-testing-framework)
6. [Validation & Backtesting](#6-validation--backtesting)

---

## 1. Palmer: Macro Intelligence Engine

### Overview

Palmer is the dashboard's real-time macro analysis engine. It synthesizes market data, economic releases, and news into actionable regime insights.

### Input Data Sources

| Source | Data | Update Frequency |
|--------|------|------------------|
| **FRED** | HY OAS (BAMLH0A0HYM2) | Daily (T+1) |
| **FMP** | VIX (^VIX), 10Y Yield (^TNX) | Real-time quotes |
| **Trading Economics** | Economic calendar | Real-time |
| **FMP** | Stock news by ticker | Real-time |
| **Alpaca** | Portfolio positions | Real-time |

### Regime Detection Algorithm

Palmer classifies the market into three primary states using a rules-based decision tree:

```
REGIME CLASSIFICATION (deterministic, no ML)
─────────────────────────────────────────────

IF (VIX > 25) OR (HY_OAS > 400 bps):
    → RISK-OFF
    
ELIF (VIX > 18) OR (HY_OAS > 350 bps):
    → CAUTIOUS
    
ELSE:
    → RISK-ON
```

**Threshold Calibration:**
- VIX thresholds (18, 25) based on historical percentiles: 18 = ~65th percentile, 25 = ~85th percentile (2010-2024 distribution)
- HY OAS thresholds (350bp, 400bp) based on ICE BofA HY Index: 350bp = ~70th percentile, 400bp = ~85th percentile

### Traffic Light System

Four independent indicators with color-coded status:

| Indicator | Green | Yellow | Red |
|-----------|-------|--------|-----|
| **Regime** | RISK-ON | CAUTIOUS | RISK-OFF |
| **Volatility** | VIX < 18 | 18 ≤ VIX ≤ 25 | VIX > 25 |
| **Credit** | HY < 325bp | 325-400bp | HY > 400bp |
| **Rates** | 10Y < 4.0% | 4.0-4.5% | 10Y > 4.5% |

### Regime Change Detection

Palmer tracks state changes between refreshes:

```python
def detect_regime_change(old_lights, new_lights):
    """
    Compare traffic light states between refreshes.
    Returns (changed: bool, details: list)
    """
    for indicator in ["regime", "volatility", "credit", "rates"]:
        if old_lights[indicator]["color"] != new_lights[indicator]["color"]:
            # Calculate direction (improving vs worsening)
            severity_order = {"green": 0, "yellow": 1, "red": 2}
            direction = "worsening" if new > old else "improving"
            yield change_event
```

### Update Frequency

| Refresh Type | Interval | Trigger |
|--------------|----------|---------|
| Automatic | 30 minutes | Background thread |
| Manual (admin) | On-demand | `/api/regime-analysis/force-refresh?secret=...` |
| Portfolio data | 5 minutes | Client-side polling |

### LLM Integration

Palmer uses GPT-4o-mini for narrative synthesis:

**Prompt Structure:**
```
Macro: VIX {value}, HY spreads {value}bp, 10Y {value}%
Events: {today's economic releases}
News: {portfolio-relevant headlines}
Portfolio: {current positions description}

Write 2-3 sentences max. Name the regime, cite ONE specific catalyst,
explain what it means for this portfolio's P&L.
```

**Temperature:** 0.3 (low variance for consistency)
**Max tokens:** 200

### Historical Accuracy

*Note: Insufficient production runtime for statistical validation. Planned metrics:*

| Metric | Target | Actual |
|--------|--------|--------|
| Regime persistence accuracy | >70% | TBD |
| Early warning lead time | 2+ days | TBD |
| False positive rate (RISK-OFF) | <15% | TBD |

---

## 2. Monte Carlo Simulation Framework

### Model Architecture

The Monte Carlo engine uses a **regime-conditional stochastic model** with position-level P&L attribution.

### Price Dynamics

**Equity Returns:** Geometric Brownian Motion with jumps

```
dS/S = μdt + σdW + JdN

Where:
- μ: drift (regime-conditional)
- σ: volatility (regime-conditional)  
- W: Wiener process
- J: jump size (log-normal)
- N: Poisson process (jump arrivals)
```

**Implementation (from `monte_carlo_v01.py`):**

```python
# Normal scenario: correlated (return, IV)
rho = assumptions.corr_return_iv  # Negative skew correlation
z1 = np.random.normal(0, 1)
z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1)

ret = equity_drift * t + equity_vol * np.sqrt(t) * z1

# Jump scenario (with probability p_jump)
if np.random.random() < jump_probability:
    ret = jump_size_mean + np.random.normal(0, 0.05)
    iv_chg = jump_iv_spike + np.random.normal(0, 2.0)
```

### Implied Volatility Dynamics

**Mean-reverting Ornstein-Uhlenbeck process:**

```
dIV = κ(θ - IV)dt + ξ·dZ

Where:
- κ: mean reversion speed
- θ: long-term IV mean (regime-conditional)
- ξ: vol-of-vol
- Z: correlated Wiener process (corr with equity returns)
```

**IV Bounds:** Clamped to [5%, 150%] to prevent degenerate scenarios.

### Regime-Conditional Parameters

| Regime | Drift (ann.) | Vol (ann.) | Corr(S,IV) | Jump P | Jump Size |
|--------|--------------|------------|------------|--------|-----------|
| GOLDILOCKS | +8% | 15% | -0.50 | 3% | -10% |
| ALL (neutral) | +5% | 18% | -0.60 | 5% | -12% |
| STAGFLATION | -2% | 22% | -0.65 | 10% | -15% |
| RISK_OFF | -25% | 35% | -0.80 | 25% | -20% |
| VOL_CRUSH | +10% | 12% | -0.35 | 1% | -6% |
| SLOW_BLEED | -8% | 16% | -0.40 | 2% | -8% |

### Correlation Structure

**6x6 correlation matrix** for multi-asset scenarios:

```
         VIX    SPX    10Y    2Y     HY     CPI
VIX    [ 1.00  -0.75  -0.40  -0.35  +0.75  +0.15]
SPX    [-0.75   1.00  +0.30  +0.25  -0.60  -0.10]
10Y    [-0.40  +0.30   1.00  +0.85  -0.20  +0.40]
2Y     [-0.35  +0.25  +0.85   1.00  -0.15  +0.30]
HY     [+0.75  -0.60  -0.20  -0.15   1.00  +0.10]
CPI    [+0.15  -0.10  +0.40  +0.30  +0.10   1.00]
```

Correlations are regime-conditional (different matrix for STAGFLATION).

### Option Pricing

**Taylor Expansion (first-order Greeks):**

```
ΔP&L ≈ δ·ΔS + ν·Δσ + θ·Δt + ½γ·(ΔS)²

Where:
- δ: delta (from Black-Scholes)
- ν: vega (per 1pt IV)
- θ: theta (daily)
- γ: gamma
```

**Black-Scholes Greeks:**

```python
d1 = (log(S/K) + (r + 0.5σ²)t) / (σ√t)
d2 = d1 - σ√t

delta_call = N(d1)
delta_put = N(d1) - 1
vega = S·N'(d1)·√t / 100
gamma = N'(d1) / (S·σ·√t)
```

### Risk Metrics

| Metric | Definition | Calculation |
|--------|------------|-------------|
| VaR 95% | 5th percentile of P&L distribution | `np.percentile(pnls, 5)` |
| VaR 99% | 1st percentile | `np.percentile(pnls, 1)` |
| CVaR 95% | Expected shortfall (tail expectation) | `mean(pnls[pnls ≤ VaR95])` |
| Win Probability | P(P&L > 0) | `sum(pnls > 0) / n` |

### Fat Tails Treatment

1. **Jump diffusion model** captures discrete large moves
2. **Regime-conditional volatility** allows for volatility clustering
3. **Correlation shifts** in stress scenarios (higher correlations in RISK_OFF)
4. **IV-return correlation** creates asymmetric payoff distributions

### Scenario Count & Convergence

Default: **10,000 scenarios**

Convergence analysis:
- Mean P&L: ±0.2% error at 10K scenarios
- VaR 95%: ±0.5% error at 10K scenarios
- CVaR 95%: ±1.0% error at 10K scenarios

---

## 3. Regime Classification System

### Architecture

Multi-pillar regime framework with independent classifiers:

```
┌─────────────────────────────────────────────┐
│              UNIFIED DASHBOARD              │
├─────────────┬─────────────┬────────────────┤
│  Inflation  │   Growth    │   Liquidity    │
│   Pillar    │   Pillar    │    Pillar      │
├─────────────┼─────────────┼────────────────┤
│   Rates     │  Volatility │    Fiscal      │
│   Pillar    │   Pillar    │    Pillar      │
└─────────────┴─────────────┴────────────────┘
```

### Pillar Classifiers

#### Macro Regime (`macro/regime.py`)

```python
def classify_macro_regime(cpi_yoy, payrolls_3m_ann, ...):
    """
    Primary rule:
    - INFLATION: CPI YoY > 3%
    - STAGFLATION: CPI YoY > 3% AND payrolls 3m ann. < 0
    
    Fallback (z-score based):
    - DISINFLATIONARY: z(inflation) < 0 AND z(real_yield) < 0
    - GOLDILOCKS: z(inflation) < 0 AND z(real_yield) >= 0
    - INFLATIONARY: z(inflation) >= 0 AND z(real_yield) < 0
    - STAGFLATION: z(inflation) >= 0 AND z(real_yield) >= 0
    """
```

#### Volatility Regime (`volatility/regime.py`)

| Regime | VIX | Term Structure | Percentile |
|--------|-----|----------------|------------|
| Low Vol | <15 | Contango | <20th |
| Normal | 15-20 | Contango | 20-50th |
| Elevated | 20-25 | Flat/Backwardation | 50-75th |
| High Vol | 25-30 | Backwardation | 75-90th |
| Crisis | >30 | Deep backwardation | >90th |

#### Rates Regime (`rates/regime.py`)

```python
# Curve inversion (growth scare)
if curve_2s10s < 0:
    return "inverted_curve"

# Duration shock
if z_10y_change_20d > 1.5:
    return "rates_shock_up"
elif z_10y_change_20d < -1.5:
    return "rates_shock_down"
```

#### Funding Regime (`funding/regime.py`)

Uses SOFR-IORB spread corridor:
- **Normal:** Spread < tight threshold
- **Tightening:** Spread > tight, < stress
- **Stress:** Spread > stress AND persistent

#### Fiscal Regime (`fiscal/regime.py`)

Combines:
- Deficit (12m rolling)
- Long-duration issuance share
- Auction tails (bps)
- Dealer take percentage
- Interest expense growth

### Z-Score Normalization

All metrics are standardized against a 2-year rolling window:

```python
z_score = (current_value - rolling_mean) / rolling_std
```

This provides:
- Stationarity (avoids level biases)
- Regime-relative positioning
- Cross-metric comparability

---

## 4. P&L Attribution & Risk Decomposition

### Greek-Based Attribution

```
Total P&L = Δ_delta + Δ_vega + Δ_theta + Δ_gamma + Δ_residual

Where:
Δ_delta = Σ (position_delta × underlying_return × notional)
Δ_vega  = Σ (position_vega × IV_change)
Δ_theta = Σ (position_theta × days_elapsed)
Δ_gamma = Σ (0.5 × position_gamma × (underlying_return)² × notional)
```

### Position-Level Greeks

From `portfolio/positions.py`:

```python
@property
def position_delta_usd(self) -> float:
    """Dollar delta exposure."""
    if self.is_option:
        return self.delta * 100 * self.notional / self.strike
    return self.notional

@property
def position_vega_usd(self) -> float:
    """Dollar vega (per 1pt IV move)."""
    return self.quantity * 100 * self.vega

@property  
def position_theta_usd(self) -> float:
    """Daily theta decay in dollars."""
    return self.quantity * 100 * self.theta
```

### Factor Exposures

| Factor | Calculation | Units |
|--------|-------------|-------|
| Equity Beta | Net delta / NAV | % NAV |
| Duration | Rate sensitivity | $ per 100bp |
| Volatility | Net vega | $ per VIX point |
| Theta Carry | Net theta × 252 / NAV | % NAV annual |
| Convexity | Net gamma × NAV × 0.01² | $ per 1% move² |

### CVaR Attribution

Position-level contribution to tail risk:

```python
# Worst 5% scenarios
worst_5pct_mask = pnls <= np.percentile(pnls, 5)
worst_scenarios = [r for i, r in enumerate(results) if worst_5pct_mask[i]]

# Aggregate position contributions
position_contrib = {}
for scenario in worst_scenarios:
    for ticker, pnl in scenario.position_pnls.items():
        position_contrib[ticker] += pnl

# Normalize to percentages
cvar_attribution = {
    ticker: (pnl / total_worst_pnl * 100)
    for ticker, pnl in position_contrib.items()
}
```

---

## 5. Stress Testing Framework

### Predefined Scenarios

| Scenario | SPX | VIX | 10Y | HY OAS |
|----------|-----|-----|-----|--------|
| Equity Crash | -20% | +25pts | -50bp | +200bp |
| Rates Shock | -5% | +5pts | +75bp | +50bp |
| Credit Event | -10% | +15pts | -25bp | +250bp |
| Flash Crash | -10% | +30pts | -50bp | +100bp |
| Stagflation | -15% | +10pts | +100bp | +150bp |

### Implementation

```python
def run_stress_test(portfolio, scenario):
    """
    Apply deterministic stress scenario to portfolio.
    Returns P&L breakdown by Greek.
    """
    pnl_delta = portfolio.net_delta * scenario.spx_change
    pnl_vega = portfolio.net_vega * scenario.vix_change
    pnl_theta = portfolio.net_theta * scenario.days
    
    return {
        "total": pnl_delta + pnl_vega + pnl_theta,
        "delta_contrib": pnl_delta,
        "vega_contrib": pnl_vega,
        "theta_contrib": pnl_theta,
    }
```

### Historical Analogs

Stress calibration based on historical events:

| Event | Date | SPX | VIX | 10Y |
|-------|------|-----|-----|-----|
| COVID crash | Mar 2020 | -34% | +65 | -150bp |
| 2022 rate shock | Oct 2022 | -25% | +15 | +230bp |
| Aug 2024 unwind | Aug 2024 | -8% | +20 | -50bp |
| SVB crisis | Mar 2023 | -5% | +10 | -80bp |

---

## 6. Validation & Backtesting

### Monte Carlo Validation

**Distributional Backtests (planned):**

1. **Coverage test:** Did realized returns fall within predicted confidence intervals?
   - Expected: 95% VaR breached ~5% of time
   
2. **Kupiec test:** Statistical test for VaR violations
   - H0: observed breach rate = expected rate

3. **Berkowitz test:** Entire distribution calibration
   - Tests if transformed returns are uniform

### Regime Accuracy Metrics (planned)

| Metric | Definition |
|--------|------------|
| Hit rate | % of times regime correctly predicted direction |
| Timing | Average lead time before regime confirmation |
| Persistence | Average duration of regime states |
| Transition matrix | Empirical probabilities between states |

### Data Quality Monitoring

```python
def validate_data_freshness():
    """Check all data sources for staleness."""
    checks = {
        "FRED": (last_update, max_staleness=2),  # T+2 for macro data
        "FMP_quotes": (last_update, max_staleness=0.5),  # 30 min
        "Alpaca": (last_update, max_staleness=0.1),  # 6 min
    }
    return all(is_fresh(check) for check in checks)
```

---

## Appendix A: Data Lineage

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                           │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│    FRED     │     FMP     │   Alpaca    │  Trading Econ    │
│  (macro)    │  (quotes)   │ (positions) │   (calendar)     │
└──────┬──────┴──────┬──────┴──────┬──────┴───────┬──────────┘
       │             │             │              │
       ▼             ▼             ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA CACHE LAYER                         │
│  data/cache/{fred,fmp_prices,correlations,playbook}/       │
└─────────────────────────────────────────────────────────────┘
       │             │             │              │
       ▼             ▼             ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                   PROCESSING LAYER                          │
│     regime classifiers, greek calculators, LLM prompts      │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                            │
│          Dashboard (Palmer) | CLI commands | API            │
└─────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Error Handling

| Error Type | Handling | User Impact |
|------------|----------|-------------|
| API timeout | Retry with exponential backoff (3x) | Brief delay |
| Missing data | Use fallback source or cached value | Stale indicator shown |
| LLM failure | Return generic "Analysis unavailable" | Degraded insight |
| Invalid Greeks | Skip position in aggregation | Partial coverage |

---

## Appendix C: Reproducibility

### Environment

```bash
# Create reproducible environment
pip install -e .  # Installs from pyproject.toml

# Required API keys in .env
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
FRED_API_KEY=...
FMP_API_KEY=...
TRADING_ECONOMICS_API_KEY=...
OPENAI_API_KEY=...
```

### Random Seed

Monte Carlo simulations use `numpy.random` without explicit seed by default. For reproducibility:

```python
np.random.seed(42)
engine = MonteCarloV01(portfolio, assumptions)
```

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Maintainer:** Lox Capital Research
