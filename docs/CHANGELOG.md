# Changelog

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
- 10 Regime Domains: Macro, Volatility, Rates, Funding, Fiscal, Commodities, Housing, Monetary, USD, Crypto
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
- Regime Domain Grid: Six macro domains (Funding, USD, Commodities, Volatility, Housing, Crypto)
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
