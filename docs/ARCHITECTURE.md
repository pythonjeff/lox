# Architecture

Lox is organized around **5 core CLI pillars** plus a web dashboard and supporting data/analysis layers.

---

## 5-Pillar CLI Structure

| Pillar | Purpose | Key Commands |
|--------|---------|--------------|
| **1. Fund Info** | NAV, investors, portfolio health | `lox status`, `lox nav` |
| **2. Macro & Regimes** | Dashboard, regime analysis | `lox dashboard`, `lox regime *` |
| **3. Portfolio Analysis** | Monte Carlo, ML scenarios | `lox scenario *` |
| **4. Ideas Generation** | Trade ideas, extreme scanners | `lox scan-extremes`, `lox ideas` |
| **5. Research** | Deep ticker/sector research | `lox research`, `lox chat` |

---

## Directory Layout

```
lox/                          # Main package (installed via pip install -e .)
|-- cli.py                    # Typer entrypoint (lox command)
|-- config.py                 # Settings / .env loader
|-- cli_commands/             # Command implementations
|   |-- core/                 # status, nav, account, dashboard, weekly report
|   |-- regimes/              # vol, fiscal, funding, rates, growth, inflation, credit, consumer, positioning, etc.
|   |-- research/             # ticker, regimes (unified), portfolio
|   +-- utils/                # Shared formatting helpers
|-- altdata/                  # External data clients (FMP, SEC, news, earnings)
|-- data/                     # Market data fetchers (Alpaca, FRED, Polygon)
|-- llm/                      # LLM analysis (analyst, regime summaries, sentiment)
|-- overlay/                  # News/calendar context for LLM payloads
|-- regimes/                  # Regime classification framework
|   |-- pillars/              # Dashboard pillars (growth, inflation, liquidity, vol)
|   |-- features.py           # Unified regime state builder
|   |-- transitions.py        # Regime transition probabilities
|   +-- feature_matrix.py     # ML feature extraction
|-- growth/                   # Growth regime (payrolls, ISM, claims, industrial production)
|-- inflation/                # Inflation regime (CPI, Core PCE, breakevens, PPI)
|-- volatility/               # Volatility regime (VIX, term structure)
|-- credit/                   # Credit regime (HY OAS, BBB/AAA spreads)
|-- rates/                    # Rates/yield curve regime
|-- funding/                  # Funding regime (SOFR, repo)
|-- consumer/                 # Consumer regime (sentiment, spending, mortgage rates)
|-- fiscal/                   # Fiscal regime (deficits, TGA, issuance)
|-- positioning/              # Positioning regime (VIX term structure, put/call, AAII)
|-- monetary/                 # Monetary policy regime
|-- usd/                      # USD regime
|-- commodities/              # Commodities regime (oil, gold, copper)
|-- nav/                      # NAV tracking and investor ledger
+-- utils/                    # Shared utilities (dates, OCC parsing, logging)

dashboard/                    # Flask web dashboard (deployed to Heroku)
|-- app.py                    # Main Flask app
|-- data_fetchers.py          # Portfolio/market data for dashboard
|-- regime_utils.py           # Regime computation for dashboard display
|-- regime_history.py         # Historical regime tracking
|-- static/                   # CSS, JS, images
+-- templates/                # Jinja2 templates

data/                         # Local data (gitignored CSVs, schemas in README)
docs/                         # Documentation
tests/                        # pytest test suite
examples/                     # Usage examples
```

---

## Data Layer

| Source | Module | Data |
|--------|--------|------|
| **FRED** | `lox/data/fred.py` | Macro time series (rates, employment, inflation, credit spreads) |
| **Alpaca** | `lox/data/alpaca.py` | Real-time quotes, positions, options, news |
| **FMP** | `lox/altdata/fmp.py` | Stock news, calendar, quotes, fundamentals (centralized client) |
| **Trading Economics** | `lox/altdata/trading_economics.py` | Consumer/macro indicators (Michigan Sentiment, ISM, AAII, retail sales) |
| **Polygon** | `lox/data/polygon.py` | Options data (OI, greeks) |

---

## Analysis Layer

| Component | Module | Purpose |
|-----------|--------|---------|
| **Regime Classification** | `lox/regimes/` | 12 regime domains with unified state builder |
| **LLM Analyst** | `lox/llm/core/analyst.py` | PhD-level research briefs with news, scenario, and trade synthesis |
| **Monte Carlo** | `lox/regimes/transitions.py` | Regime-conditional scenario simulation with trained correlations |
| **Ticker Research** | `lox/cli_commands/research/ticker_cmd.py` | Fundamentals, technicals, SEC filings, LLM analysis |

---

## Regime System

12 regime domains, each following a consistent pattern:

1. **Signals module** (`signals.py`): Fetches raw data from FRED/Alpaca/FMP/Trading Economics, computes z-scores and indicators.
2. **Regime classifier** (`regime.py`): Maps indicators to a named regime with label, score (0-100), and tags.
3. **Features module** (`features.py`): Exports flat feature vectors for ML.

Domains: Growth, Inflation, Volatility, Credit, Rates, Funding, Consumer, Fiscal, Positioning, Monetary, USD, Commodities.

Score interpretation: 0 = strong risk-on signal, 100 = strong risk-off signal. The unified view derives a "Macro Quadrant" (Stagflation, Goldilocks, Reflation, Deflation Risk, or Mixed) from the Growth + Inflation scores.

The unified state builder (`lox/regimes/features.py`) computes all regimes and produces a single `UnifiedRegimeState` with an overall risk score.

### Monte Carlo Correlation Training

The Monte Carlo engine uses correlations trained on 10+ years of real FRED market data:

- **Correlation Trainer**: Downloads VIX, rates, spreads, CPI; calculates historical and regime-conditional correlations.
- **Smart Fallback**: If a regime has insufficient training data (e.g., STAGFLATION with 0 historical days), falls back to ALL-regime correlations, then to heuristics.
- **Regime-Conditional**: GOLDILOCKS, INFLATIONARY, etc. each have their own correlation matrices when sufficient data exists.

Key finding: real correlations are less extreme than hand-coded heuristics (e.g., VIX-10Y: -0.16 actual vs -0.40 heuristic), producing tighter and more realistic P&L distributions.

---

## Web Dashboard

The Flask dashboard (`dashboard/`) is the primary interface, deployed on Heroku at [loxfund.com](https://loxfund.com).

Key capabilities:
- **Liquidation NAV**: Conservative bid/ask marking
- **Palmer AI**: LLM-generated macro context with traffic light system (RISK-ON / CAUTIOUS / RISK-OFF)
- **Monte Carlo Forecast**: 6-month outlook with VaR 95%
- **Position Intelligence**: LLM theories for each position's profit conditions
- **Economic Calendar**: Upcoming releases with beat/miss indicators
