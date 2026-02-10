# CLI Reference

Full command reference for the Lox CLI.

---

## CLI Structure (5 Pillars)

```
lox                           # Help with examples
|
+-- [1] Fund Information
|   +-- status                # Portfolio health
|   +-- nav                   # NAV management
|
+-- [2] Macro & Regimes
|   +-- dashboard             # All regimes at a glance
|   +-- regime                # Economic regime analysis (12 domains)
|       +-- growth            # Growth (payrolls, ISM, claims)
|       +-- inflation         # Inflation (CPI, PCE, breakevens)
|       +-- vol               # Volatility (VIX)
|       +-- credit            # Credit spreads (HY OAS, BBB, AAA)
|       +-- rates             # Yield curve
|       +-- funding           # Funding markets (SOFR)
|       +-- consumer          # Consumer (sentiment, spending, mortgage)
|       +-- fiscal            # Fiscal (deficits, TGA)
|       +-- positioning       # Positioning (VIX term, P/C, AAII)
|       +-- monetary          # Monetary policy (reserves, RRP)
|       +-- usd               # USD (DXY)
|       +-- commodities       # Commodities (oil, gold, copper)
|       +-- macro             # Alias: Growth + Inflation + quadrant
|       +-- unified           # All 12 regimes + MC adjustments
|
+-- [3] Portfolio Analysis
|   +-- scenario              # Portfolio scenarios
|       +-- monte-carlo       # Monte Carlo simulation
|       +-- stress            # Stress testing
|       +-- forward           # Forward-looking scenarios
|       +-- custom            # Custom scenario builder
|
+-- [4] Ideas Generation
|   +-- scan-extremes         # Overbought/oversold scanner
|   +-- ideas                 # Trade ideas
|   +-- scan -t TICKER        # Options chain scanner
|
+-- [5] Research
|   +-- research ticker TICKER # Deep ticker research
|   +-- research regimes      # Unified regime view
|   +-- research portfolio    # Open position outlook
|   +-- chat                  # Interactive research chat
|   +-- labs                  # Advanced tools (power users)
|
+-- trade                     # Trade execution
+-- options                   # Full options toolset
```

---

## Core Commands

| Command | Purpose |
|---------|---------|
| `lox status` | Portfolio health at a glance |
| `lox dashboard` | All regime pillars |
| `lox scan -t TICKER` | Options chain scanner |
| `lox research ticker TICKER` | Deep ticker research (momentum, HF metrics, SEC filings) |
| `lox regime vol` | Volatility regime |
| `lox regime fiscal` | Fiscal regime |

---

## Research Commands

| Command | Purpose |
|---------|---------|
| `lox research ticker AAPL` | Full research: momentum, HF metrics, fundamentals |
| `lox research ticker NVDA --llm` | With LLM synthesis |
| `lox research regimes` | Unified regime overview |
| `lox research regimes --llm` | With AI commentary |
| `lox research regimes -d vol` | Drill into volatility regime |
| `lox research portfolio` | LLM outlook on open positions |
| `lox regime vol --llm` | Volatility with LLM |
| `lox regime rates` | Rates/curve analysis |
| `lox scan-extremes` | Find overbought/oversold tickers |

---

## Regime Commands

```bash
# Core regimes (drive Monte Carlo adjustments)
lox regime growth                 # Growth (payrolls, ISM, claims, industrial production)
lox regime inflation              # Inflation (CPI, Core PCE, breakevens, PPI)
lox regime vol                    # Volatility (VIX, term structure)
lox regime credit                 # Credit spreads (HY OAS, BBB, AAA)
lox regime rates                  # Rates/yield curve
lox regime funding                # Funding markets (SOFR, repo)

# Extended regimes (context)
lox regime consumer               # Consumer (sentiment, spending, mortgage rates)
lox regime fiscal                 # Fiscal (deficits, TGA)
lox regime positioning            # Positioning (VIX term slope, put/call, AAII)
lox regime monetary               # Monetary policy (reserves, RRP)
lox regime usd                    # USD (DXY)
lox regime commodities            # Commodities (oil, gold, copper)

# Aliases and unified views
lox regime macro                  # Alias: shows Growth + Inflation + macro quadrant
lox regime unified                # All 12 regimes + MC adjustments + regime changes

# Add --llm to any command for LLM analysis
lox regime vol --llm
```

---

## Options Commands

| Command | Purpose |
|---------|---------|
| `lox scan -t NVDA` | Options chain (default: puts, 30-365 DTE) |
| `lox scan -t CRWV --want put --min-days 100` | Custom DTE range |
| `lox options best NVDA --budget 500` | Best options under budget |

---

## Scenario Analysis

```bash
lox scenario monte-carlo          # Monte Carlo simulation
lox scenario stress               # Stress testing
```

---

## Interactive Chat

| Command | Purpose |
|---------|---------|
| `lox chat` | Interactive research chat |
| `lox chat -t AAPL` | Ticker-focused chat |
| `lox chat -c fiscal` | Chat with regime context |

---

## Advanced Tools (Power Users)

```bash
lox labs ticker deep -t AAPL --llm    # Full ticker deep dive
lox labs mc-v01 --regime RISK_OFF     # Specific regime scenario
lox labs unified                       # All 12 regimes
lox labs transitions                   # Transition matrix with signal adjustments
lox labs train-correlations            # Train Monte Carlo correlations on real data
```

---

## Daily Workflow

```bash
# Morning: Dashboard review (5 min)
lox status                    # Portfolio health
lox dashboard                 # All regimes at a glance

# Research (as needed)
lox research ticker NVDA      # Full research report
lox regime vol                # Volatility regime
lox regime fiscal             # Fiscal regime

# Options scanning
lox scan -t CRWV --want put   # Options chain
lox scan -t NVDA --min-days 60 --max-days 180

# Ideas and scenarios
lox scan-extremes             # Find overbought/oversold tickers
lox ideas                     # Trade ideas
lox scenario monte-carlo      # Monte Carlo simulation

# EOD: Record NAV
lox nav snapshot
```

---

## Dashboard Access

| Environment | URL | Command |
|------------|-----|---------|
| **Production** | [loxfund.com](https://loxfund.com) | Direct browser access |
| **Local** | http://localhost:5001 | `cd dashboard && python app.py` |
