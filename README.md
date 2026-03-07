# Lox

Discretionary macro research platform with systematic tools for regime-aware portfolio management, risk monitoring, and tail-risk hedging. Built for a PM morning risk meeting workflow.

**Live Dashboard**: [loxfund.com](https://loxfund.com)

## Quick Start

```bash
git clone https://github.com/pythonjeff/lox.git
cd lox
pip install -e ".[dashboard]"
cp .env.example .env
# Edit .env with your API keys (Alpaca, OpenAI, FRED, FMP)
lox --help
```

---

## PM Morning Report

Your daily hedge fund briefing in one command. Combines macro regime state, active scenarios, portfolio Greeks, and a streaming LLM CIO brief.

```bash
lox pm                # Full report with LLM briefing (default on)
lox pm --no-llm       # Data sections only
lox pm --json         # Machine-readable JSON
```

---

## Regime Engine

10-pillar macro regime system scoring 0-100 (higher = more stress/risk). Each pillar uses a multi-layer classifier with weighted sub-scores, cross-signal confirmation, and sector/factor decomposition.

```bash
lox research regimes          # Overview with trend arrows + 7d deltas
lox research regimes --trend  # Full trend dashboard (sparklines, momentum z, velocity)
lox research regimes --detail credit  # Deep dive on one pillar + trend panel
```

### Pillar Commands

Every regime pillar supports `--llm` for LLM chat, `--book` for position impact, and drill-down flags:

```bash
lox regime growth             # GDP, labor, ISM, LEI
lox regime inflation          # CPI, PCE, breakevens, tariff pass-through
lox regime vol                # VIX term structure, realized vs implied
lox regime credit             # IG/HY spreads, credit conditions
lox regime rates              # Yield curve, Fed policy, duration risk
lox regime funding            # Repo, commercial paper, bank reserves
lox regime fiscal             # Deficit, debt/GDP, Treasury issuance
lox regime consumer           # Sentiment, spending, labor
lox regime earnings           # S&P 500 beat rate, revisions, surprises
lox regime oil                # Energy/commodities regime
```

### Enriched Regime Features

- **3-layer scoring**: Weighted sub-scores, sector/factor amplifiers, cross-signal confirmation
- **Sparklines**: Rolling 90-day score history with trend arrows and momentum
- **Delta tracking**: Score changes vs N days ago
- **Cross-regime signals**: Confirmation/divergence detection across pillars
- **Feature vectors**: `--features` exports ML-ready JSON for each pillar
- **Alert mode**: `--alert` suppresses output unless regime is extreme (for cron monitoring)
- **Calendar events**: `--calendar` shows upcoming catalysts
- **Trade expressions**: `--trades` suggests instrument-level trade ideas
- **Book impact**: `--book` maps regime to your open Alpaca positions

### Scenarios + Trends

```bash
lox research regimes --scenarios        # Active macro scenarios (conviction-ranked)
lox research regimes --detail credit    # Deep dive on one pillar + trend panel
```

8 named macro scenarios (e.g., Stagflation, Credit Crunch, Goldilocks) auto-evaluated against live regime state with HIGH/MEDIUM conviction scoring.

---

## Risk Management

Portfolio-level Greeks dashboard with theta breakeven analysis.

```bash
lox risk                      # Full Greeks dashboard + theta breakeven
lox risk --json               # Machine-readable export
```

### What `lox risk` Shows

1. **Account snapshot** — equity, buying power, options BP
2. **Portfolio Greeks** — consolidated net delta, gamma, theta, vega
3. **Exposure by underlying** — per-name delta decomposition
4. **Position detail** — every position with Greeks, IV, and P/L
5. **Risk signals** — auto-generated warnings (exposure, gamma, theta, vol, leverage)
6. **Theta breakeven** — per-name delta breakeven and gamma scalp breakeven
7. **Theta burn analysis** — daily/weekly/monthly/annual projections

---

## Research

```bash
lox research ticker NVDA      # Full hedge-fund-style research report
lox research portfolio        # Outlook on all open positions
lox research scenario SPY     # Monte Carlo macro shock simulation
lox research chat             # Interactive research chat
lox scan -t NVDA --want put   # Options chain scanner with Greek filters
```

---

## Crypto Perps

Real-time crypto perpetual futures data, LLM-powered analysis, and manual trading via Aster DEX.

```bash
lox crypto data                          # BTC, ETH, SOL overview + technicals
lox crypto data --coins BTC,ETH,DOGE     # Custom coin list
lox crypto research                      # Data + macro regime LLM analysis
lox crypto analyze                       # Perps-specific trading analysis

# Trading (requires Aster DEX config in .env)
lox crypto balance                       # Account balance & equity
lox crypto positions                     # Open positions with PnL
lox crypto trade BTC BUY 0.001 --leverage 5
lox crypto close BTC
```

---

## Accounting & Reporting

```bash
lox status                    # Portfolio health (NAV, P&L)
lox account                   # Alpaca account details
lox account summary           # LLM summary with news + calendar + risk watch
lox nav snapshot              # NAV and investor ledger
lox weekly report --share     # Investor-facing weekly report
```

## Dashboard

```bash
cd dashboard && python app.py
# Navigate to http://localhost:5001
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/CLI_REFERENCE.md) | Full command reference and daily workflows |
| [Architecture](docs/ARCHITECTURE.md) | System design and module layout |
| [Methodology](docs/METHODOLOGY.md) | Palmer, Monte Carlo, regime detection algorithms |
| [Technical Spec](docs/TECHNICAL_SPEC.md) | Data lineage, error handling, deployment |
| [Changelog](docs/CHANGELOG.md) | Version history and recent upgrades |

## API Keys Required

| Service | Purpose | Required? |
|---------|---------|-----------|
| [Alpaca](https://app.alpaca.markets/) | Brokerage, positions, options data, Greeks | Yes |
| [OpenAI](https://platform.openai.com/api-keys) or [OpenRouter](https://openrouter.ai/keys) | LLM analysis | For `--llm` / `lox pm` |
| [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) | Macro/economic time series | Yes |
| [FMP](https://financialmodelingprep.com/developer/docs/) | News, calendar, earnings, quotes | Yes |
| [Trading Economics](https://tradingeconomics.com/api/) | Consumer/macro indicators | Optional, falls back to FRED |
| [Aster DEX](https://app.asterdex.com/) | Crypto perps trading | For `crypto trade` only |

## License

[Add your license here]
