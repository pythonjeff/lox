# Changelog

---

## v6 — Composite Regime + USD Regime (2026-03-10)

### New: Composite Regime Classification
- **Hedge-fund-style macro regime ID** — compresses 12 pillar scores into 5 named regimes via distance-based prototype matching
- **5 regimes:** RISK-ON / GOLDILOCKS, REFLATION, STAGFLATION, RISK-OFF / DEFLATIONARY, TRANSITION / MIXED
- **Confidence scoring** — softmax probability distribution with TRANSITION override for high-dispersion, low-conviction states
- **Transition outlook** — projects pillar scores 21 days forward using velocity, reclassifies to estimate where the regime is heading
- **Swing factors** — identifies which pillars are closest to flipping the regime, with velocity-based ETAs
- **Canonical playbooks** — positioning guidance per regime (equity, duration, credit, commodity, vol stances + key trade expressions with tickers)
- **CLI:** `lox regime composite` (full dashboard), `lox regime composite --json` (machine-readable)
- **Integrated everywhere:** PM report header, LLM system prompt, `lox research regimes` overview, trend dashboard headers
- New files: `lox/regimes/composite.py`, `lox/cli_commands/shared/composite_display.py`

### New: USD Regime Command
- **Dedicated USD strength analysis** — trade-weighted dollar regime with FX momentum, volatility, and cross-regime implications
- **Dashboard:** Broad index level, 200d MA distance, z-score, 20d/60d/YoY momentum, realized FX volatility
- **Cross-regime signals:** Growth-USD, commodity-USD divergence detection with context-aware interpretation
- **Tail risks:** Dollar surge (EM funding stress, commodity crash) and dollar plunge (confidence crisis, imported inflation) warnings
- **CLI:** `lox regime usd` with `--llm`, `--trades`, `--alert`, `--calendar`, `--features`, `--json`, `-t TICKER` flags
- New file: `lox/cli_commands/regimes/usd_cmd.py`

### PM Report Enhancements
- Composite regime headline in PM report header (regime name + confidence + transition direction)
- Composite regime context injected into LLM system prompt (probabilities, swing factors, playbook, transition outlook)
- JSON output includes full composite regime object

### Regime Engine Updates
- `UnifiedRegimeState` now includes `composite` field (populated automatically after trends/dislocations)
- Scenario count increased from 8 to 10 (added Trade War Escalation, Supply Chain Normalization)
- Dislocation detector: 12 cross-pillar divergence rules (credit-vol, growth-credit, rates-growth, etc.)
- Pillar count: 12 (Growth, Inflation, Volatility, Credit, Rates, Liquidity, Consumer, Fiscal, USD, Commodities, Earnings, Policy)

---

## v5 — CLI Consolidation + PM Morning Report (2026-03-06)

### CLI Cleanup
- **Removed** 12 unused CLI files and ~800 lines of dead code
- **Removed commands**: `lox dashboard`, `lox analyze`, `lox suggest`, `lox run`, `lox regime macro`, `lox regime unified`
- **Removed regime commands**: `lox regime commodities`, `lox regime usd`, `lox regime positioning`, `lox regime monetary`, `lox regime policy`, `lox regime crypto`
- **Removed research commands**: `lox research cvna`, `lox research oi-scan`
- **Moved** `invite-investor` and `create-admin` into `lox account` subgroup
- Backend data modules (growth, inflation, commodities, usd, etc.) remain intact — only CLI surface removed

### New: `lox pm` — PM Morning Report
- Single command daily hedge fund briefing: `lox pm`
- Parallel data fetch (regime state + Greeks + positions via ThreadPoolExecutor)
- 4 sections: Macro Environment (10-pillar heatmap), Active Scenarios, Portfolio (NAV/Greeks/bleeders/winners/risk signals), LLM CIO Briefing
- LLM on by default with streaming output; CIO-grade system prompt (dense, opinionated, 250 words max)
- `--no-llm` for data-only mode, `--json` for machine-readable output
- Graceful degradation if any data source fails

### Documentation
- Updated all docs to reflect current CLI structure
- Removed references to deleted commands throughout

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
