# Changelog

---

## v4 — Regime Restructuring (2026-02-10)

### 12-Domain Regime System (up from 10)
- **Split** Macro regime into **Growth** (payrolls, ISM, claims, industrial production) and **Inflation** (CPI, Core PCE, breakevens, PPI)
- **New: Credit** regime — HY OAS levels, 30d change, BBB/AAA spreads, HY-IG spread
- **New: Consumer** regime — Michigan Sentiment/Expectations, retail sales, personal spending, mortgage rates (absorbs old Housing regime)
- **New: Positioning** regime — VIX term structure slope, put/call ratio, AAII sentiment
- **Deleted** Crypto regime (persistent data unavailability)
- **Deleted** Housing regime (metrics absorbed into Consumer)

### Macro Quadrant Derivation
- Growth + Inflation scores now produce a derived quadrant: Stagflation, Goldilocks, Reflation, Deflation Risk, or Mixed
- Displayed in the `lox regime unified` header

### Enhanced Unified Output
- Traffic light emojis (🔴 🟡 🟢) based on regime score
- "Key Inputs" column showing actual data values instead of prose descriptions
- Regime Changes (30d) section tracking historical transitions
- Monte Carlo Adjustments table with Base, Adjusted values, and driving regimes
- LLM-generated one-sentence portfolio implication
- Score guide legend

### New CLI Commands
- `lox regime growth` — Growth regime snapshot
- `lox regime inflation` — Inflation regime snapshot
- `lox regime credit` — Credit/spreads regime snapshot
- `lox regime consumer` — Consumer health regime snapshot
- `lox regime positioning` — Market positioning regime snapshot
- `lox regime macro` kept as alias showing Growth + Inflation + quadrant

### New Data Sources
- Trading Economics API client (`lox/altdata/trading_economics.py`) for consumer/macro indicators
- FRED series additions: PCEPILFE, PPIFIS, INDPRO, PCETRIM12M159SFRBDAL, BAMLC0A4CBBB, BAMLC0A1CAAA, UMCSENT, RSXFS, TOTALSL
- Regime history persistence (`~/.lox/regime_history.json`) for change detection

### Updated Weights
- New 12-regime weighted overall score: Growth (15%), Inflation (10%), Volatility (15%), Credit (15%), Rates (10%), Funding (5%), Consumer (10%), Fiscal (5%), Positioning (4%), Monetary (5%), USD (3%), Commodities (3%)

---

## v3 — Architecture Consolidation (2026-01-25)

### 5-Pillar CLI Architecture
- **Fund Info**: NAV, investors, portfolio health
- **Macro & Regimes**: Dashboard, regime analysis with subcommands
- **Portfolio Analysis**: Monte Carlo, ML-enhanced scenarios
- **Ideas Generation**: Trade ideas based on overbought/oversold extremes
- **Research**: Deep ticker/sector research with LLM integration

### Enhanced Research Command
- Momentum Metrics: 1W/1M/3M/6M/1Y returns, RSI, SMA crossovers
- Hedge Fund Metrics: Sharpe, Sortino, Calmar, VaR, CVaR, Max Drawdown, Skewness, Kurtosis
- SEC Filings: Recent 8-K, 10-K, 10-Q, Form 4 insider activity
- Asset-Type Aware: Different reports for ETFs vs stocks

### Code Consolidation
- Reduced `cli.py` from ~1,800 to ~1,000 lines
- Centralized FMP API calls to single `altdata/fmp.py` client
- Merged scenario commands (basic + ML) into single module
- Created `cli_commands/utils/formatting.py` for reusable display helpers
- Consolidated snapshot builders into `data/snapshots.py`

### v3.5 — Real Data Integration
- **Correlation Trainer**: Train Monte Carlo correlations on 10+ years of real FRED data (VIX, rates, spreads, CPI)
- **Smart Fallback**: If regime has insufficient data (e.g., STAGFLATION), falls back to ALL regime correlations, then heuristics
- **Real Position Integration**: Fetches positions from Alpaca, calculates actual greeks
- Key finding: real correlations are less extreme than heuristics, producing tighter and more realistic distributions

---

## v2 — Regime System (2026-01-25)

### Unified Regime Framework
- 12 Regime Domains: Growth, Inflation, Volatility, Credit, Rates, Funding, Consumer, Fiscal, Positioning, Monetary, USD, Commodities
- Unified State Builder: Single `lox labs unified` command shows all regimes with scores
- ML Feature Extraction: Export flat feature vectors with `--json` for model training
- Standardized Interface: All regimes return consistent `RegimeResult` with name, label, score, tags

### Leading Indicator Adjustments
- 7 Warning Signals: Yield curve inversion, VIX elevated/complacent, credit widening, funding stress, rates shock
- Probability Adjustments: Signals adjust transition probabilities (e.g., inverted curve -> risk-off prob x1.8)
- Research-Backed: Each indicator sourced from academic research (Estrella & Mishkin, CBOE, Gilchrist & Zakrajsek)

### Regime Transition Probabilities
- Historical Frequencies: Base probabilities from 2000-2024 market data
- Horizon Scaling: 1-month, 3-month, 6-month horizons with mean-reversion
- Monte Carlo Integration: Scenarios weighted by transition probability
- Path Simulation: `simulate_regime_path()` for multi-step forecasts

---

## v1 — Dashboard Launch (2026-01-24)

### Investor Dashboard (live at [loxfund.com](https://loxfund.com))
- TWR Performance: GIPS-compliant time-weighted returns as headline metric
- Benchmark Comparison: S&P 500 performance with alpha calculation
- Liquidation NAV: Conservative bid/ask marking for accurate portfolio valuation
- Investor Ledger: Individual investor P&L with unitized returns

### Market Intelligence
- Regime Trackers: Visual range bars for VIX, HY Spread, 10Y Yield with smart thresholds
- Regime Domain Grid: Core + extended domains (Growth, Inflation, Volatility, Credit, Rates, Funding, Consumer, Fiscal, Positioning, Monetary, USD, Commodities)
- Palmer AI Analysis: LLM-generated macro context with portfolio impact assessment
- Traffic Light System: Real-time RISK-ON / CAUTIOUS / RISK-OFF classification

### News & Economic Calendar
- Portfolio News Feed: Headlines filtered by active positions
- Economic Calendar: Upcoming high-impact events (Fed speeches, PPI, CPI, NFP)
- Source Attribution: News sources with timestamps

### Position Intelligence
- LLM Position Theories: Macro-aware explanations for each position's profit conditions
- Monte Carlo Forecast: 6-month outlook with VaR 95% and win probability
- Trade Performance: Realized P&L history with win rate tracking

### Deep Ticker Research
- Asset-Type Aware: Identifies ETFs vs stocks, tailors analysis accordingly
- ETF Analysis: AUM, expense ratio, top holdings, fund flows, institutional holders
- Stock Analysis: Company profile, technicals, hedge fund metrics, earnings, SEC filings, analyst targets
- Interactive Ticker Chat: `lox chat -t AAPL` for focused discussion with research context

### Design
- Collapsed Accordions: Clean default view with key metrics visible at a glance
- Mobile-Responsive: Professional interface optimized for all devices
- Standardized CLI Flags: Uniform `--llm`, `--features`, `--json` across all `lox labs` commands
