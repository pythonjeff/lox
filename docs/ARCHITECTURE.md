# Architecture

Lox is organized around a streamlined CLI for daily PM workflows plus a web dashboard and supporting data/analysis layers.

---

## CLI Structure

| Area | Purpose | Key Commands |
|------|---------|--------------|
| **PM Report** | Daily CIO briefing | `lox pm` |
| **Research** | Deep analysis + regime views | `lox research regimes`, `lox research ticker`, `lox research scenario` |
| **Regime Drill-Down** | Per-pillar deep dives | `lox regime vol`, `lox regime credit`, etc. |
| **Portfolio & Risk** | Health + Greeks | `lox status`, `lox risk`, `lox scan` |
| **Accounting** | NAV, investors, reports | `lox nav`, `lox weekly`, `lox account` |
| **Crypto** | Perps data + trading | `lox crypto data`, `lox crypto trade` |

---

## Directory Layout

```
lox/                          # Main package (installed via pip install -e .)
|-- cli.py                    # Typer entrypoint (lox command)
|-- config.py                 # Settings / .env loader
|-- cli_commands/             # Command implementations
|   |-- core/                 # pm, status, account, nav, risk, weekly report
|   |-- regimes/              # vol, fiscal, funding, rates, growth, inflation, credit, consumer, earnings, oil
|   |-- research/             # ticker, regimes (unified), portfolio, scenario, chat
|   +-- shared/               # Shared display components (regime_display, regime_chat, scenario_display, trend_display)
|-- altdata/                  # External data clients (FMP, SEC, news, earnings)
|-- data/                     # Market data fetchers (Alpaca, FRED, Polygon)
|-- llm/                      # LLM analysis (analyst, regime summaries, sentiment)
|-- overlay/                  # News/calendar context for LLM payloads
|-- regimes/                  # Regime classification framework
|   |-- pillars/              # Dashboard pillars (growth, inflation, liquidity, vol)
|   |-- features.py           # Unified regime state builder
|   |-- scenarios.py          # Cross-regime macro scenarios
|   |-- trend.py              # Regime trend/momentum
|   |-- transitions.py        # Regime transition probabilities
|   +-- feature_matrix.py     # ML feature extraction
|-- scenarios/                # Monte Carlo scenario engine
|   +-- engine.py             # Block-bootstrap MC with macro shock translation
|-- risk/                     # Portfolio risk
|   +-- greeks.py             # Portfolio Greeks aggregation
|-- growth/                   # Growth regime data (payrolls, ISM, claims)
|-- inflation/                # Inflation regime data (CPI, Core PCE, breakevens)
|-- volatility/               # Volatility regime data (VIX, term structure)
|-- credit/                   # Credit regime data (HY OAS, BBB/AAA spreads)
|-- rates/                    # Rates/yield curve regime data
|-- funding/                  # Funding regime data (SOFR, repo)
|-- consumer/                 # Consumer regime data (sentiment, spending)
|-- fiscal/                   # Fiscal regime data (deficits, TGA)
|-- earnings/                 # Earnings regime data (S&P 500 beat rate)
|-- commodities/              # Commodities regime data (oil, gold, copper)
|-- positioning/              # Positioning regime data (VIX term, P/C, AAII)
|-- monetary/                 # Monetary policy regime data
|-- usd/                      # USD regime data (DXY)
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
```

---

## Data Layer

| Source | Module | Data |
|--------|--------|------|
| **FRED** | `lox/data/fred.py` | Macro time series (rates, employment, inflation, credit spreads) |
| **Alpaca** | `lox/data/alpaca.py` | Real-time quotes, positions, options, news |
| **FMP** | `lox/altdata/fmp.py` | Stock news, calendar, quotes, fundamentals |
| **Trading Economics** | `lox/altdata/trading_economics.py` | Consumer/macro indicators |
| **Polygon** | `lox/data/polygon.py` | Options data (OI, greeks) |

---

## Analysis Layer

| Component | Module | Purpose |
|-----------|--------|---------|
| **Regime Classification** | `lox/regimes/` | 12 regime domains with unified state builder |
| **LLM Analyst** | `lox/llm/core/analyst.py` | Research briefs with news, scenario, and trade synthesis |
| **Monte Carlo** | `lox/scenarios/engine.py` | Block-bootstrap MC with macro shock translation |
| **Ticker Research** | `lox/cli_commands/research/ticker_cmd.py` | Fundamentals, technicals, SEC filings, LLM analysis |
| **PM Morning Report** | `lox/cli_commands/core/pm_cmd.py` | Daily CIO briefing combining all data sources |

---

## Regime System

12 regime domains, each following a consistent pattern:

1. **Signals module** (`signals.py`): Fetches raw data from FRED/Alpaca/FMP/Trading Economics, computes z-scores and indicators.
2. **Regime classifier** (`regime.py`): Maps indicators to a named regime with label, score (0-100), and tags.
3. **Features module** (`features.py`): Exports flat feature vectors for ML.

CLI exposes 10 pillars directly (growth, inflation, volatility, credit, rates, funding, consumer, fiscal, earnings, oil). All 12 domains (including positioning, monetary, usd) contribute to the unified regime state and overall risk score.

Score interpretation: 0 = strong risk-on signal, 100 = strong risk-off signal. The unified view derives a "Macro Quadrant" (Stagflation, Goldilocks, Reflation, Deflation Risk, or Mixed) from the Growth + Inflation scores.

The unified state builder (`lox/regimes/features.py`) computes all regimes and produces a single `UnifiedRegimeState` with an overall risk score.

### Monte Carlo Correlation Training

The Monte Carlo engine uses correlations trained on 10+ years of real FRED market data:

- **Correlation Trainer**: Downloads VIX, rates, spreads, CPI; calculates historical and regime-conditional correlations.
- **Smart Fallback**: If a regime has insufficient training data, falls back to ALL-regime correlations, then to heuristics.
- **Regime-Conditional**: Each regime state has its own correlation matrix when sufficient data exists.

---

## Web Dashboard

The Flask dashboard (`dashboard/`) is the investor-facing interface, deployed on Heroku at [loxfund.com](https://loxfund.com).

Key capabilities:
- **Liquidation NAV**: Conservative bid/ask marking
- **Palmer AI**: LLM-generated macro context with traffic light system (RISK-ON / CAUTIOUS / RISK-OFF)
- **Monte Carlo Forecast**: 6-month outlook with VaR 95%
- **Position Intelligence**: LLM theories for each position's profit conditions
- **Economic Calendar**: Upcoming releases with beat/miss indicators
