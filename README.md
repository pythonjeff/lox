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

## Regime Engine

12-pillar macro regime system scoring 0-100 (higher = more stress/risk). Each pillar uses a multi-layer classifier with weighted sub-scores, cross-signal confirmation, and sector/factor decomposition.

```bash
lox regime unified            # All 12 regimes + Monte Carlo adjustments
lox research regimes          # Overview with trend arrows + 7d deltas
lox research regimes --trend  # Full trend dashboard (sparklines, momentum z, velocity)
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
lox regime commodities        # Energy, metals, agriculture
lox regime usd                # DXY, EM FX stress, trade flows
lox regime positioning        # CFTC, put/call, fund flows
lox regime earnings           # S&P 500 beat rate, revisions, surprises
```

### Enriched Regime Features

Regimes include quant-grade analytics built for systematic discretionary workflows:

- **3-layer scoring**: Weighted sub-scores → sector/factor amplifiers → cross-signal confirmation
- **Sparklines**: Rolling 90-day score history with trend arrows and momentum
- **Delta tracking**: `--delta 7d` shows score changes vs N days ago
- **Cross-regime signals**: Detects confirmation/divergence across pillars (e.g., credit stress + vol compression = warning)
- **Feature vectors**: `--features` exports ML-ready JSON for each pillar
- **Alert mode**: `--alert` suppresses output unless regime is extreme (for cron monitoring)
- **Calendar events**: `--calendar` shows upcoming catalysts that could shift the regime
- **Trade expressions**: `--trades` suggests instrument-level trade ideas for the current regime
- **Book impact**: `--book` shows how the regime maps to your open Alpaca positions

### Earnings Regime + Sector Drill-Down

The earnings pillar tracks S&P 500 earnings season in real-time with sector decomposition:

```bash
lox regime earnings                     # Full earnings dashboard with sector heatmap
lox regime earnings --sector technology # Drill into sector → stock basket + idea generator
lox regime earnings --sector real-estate # Hyphenated names work for multi-word sectors
lox regime earnings --delta 7d          # Score movement over the last week
```

The sector drill-down returns a basket of stocks in that GICS sector with per-stock EPS surprise, revenue surprise, analyst consensus, and performance — designed as a quick idea generator.

### Scenarios + Trends

```bash
lox research regimes --detail credit    # Deep dive on one pillar + trend panel
lox research regimes --scenarios        # Active macro scenarios (conviction-ranked)
```

8 named macro scenarios (e.g., Stagflation, Credit Crunch, Goldilocks) auto-evaluated against live regime state with HIGH/MEDIUM conviction scoring.

---

## Risk Management

Portfolio-level Greeks dashboard with theta breakeven analysis. The foundational "am I hedged?" view for a morning risk meeting.

```bash
lox risk                      # Full Greeks dashboard + theta breakeven
lox risk --json               # Machine-readable export
```

### What `lox risk` Shows

1. **Account snapshot** — equity, buying power, options BP
2. **Portfolio Greeks** — consolidated net delta, gamma, theta, vega with human-readable labels
3. **Exposure by underlying** — per-name delta decomposition (equity Δ + options Δ = net Δ), gamma, theta, vega
4. **Position detail** — every position with Greeks, IV, and P/L
5. **Risk signals** — auto-generated warnings (directional exposure, gamma profile, theta decay, vol exposure, leverage)
6. **Theta breakeven by underlying** — per-name delta breakeven ($/day and %/day) and gamma scalp breakeven
7. **Theta burn analysis** — daily/weekly/monthly/annual projections, equity burn rate, portfolio-level delta and gamma scalp breakevens

### Greek Conventions

- Equity: delta = qty, gamma/theta/vega = 0
- Options: position Greek = per-contract × qty × 100 (standard contract multiplier)
- Gamma scalp breakeven: `move = sqrt(2 × |theta| / gamma)` — daily underlying move where gamma P/L covers theta cost

---

## Research

```bash
lox research ticker NVDA      # Full hedge-fund-style research report
lox research portfolio        # Outlook on all open positions
lox account summary           # LLM summary with risk watch + news + calendar
lox scan -t NVDA --want put   # Options chain scanner with Greek filters
```

---

## Crypto Perps

Real-time crypto perpetual futures data, LLM-powered analysis, and manual trading via Aster DEX. No API keys needed for market data — uses public CCXT endpoints (OKX by default).

```bash
# Market data & technicals
lox crypto data                          # BTC, ETH, SOL overview + technicals
lox crypto data --coins BTC,ETH,DOGE     # Custom coin list

# LLM analysis
lox crypto research                      # Data + macro regime LLM analysis
lox crypto analyze                       # Perps-specific trading analysis
lox crypto analyze --coins BTC           # Single coin deep dive

# Regime
lox regime crypto                        # Crypto regime score (funding, technicals, momentum)

# Trading (requires Aster DEX config in .env)
lox crypto balance                       # Account balance & equity
lox crypto positions                     # Open positions with PnL
lox crypto trade BTC BUY 0.001 --leverage 5   # Place a trade
lox crypto close BTC                     # Close a position
```

To enable trading:
```bash
pip install -e ".[trading]"
# Then add ASTER_USER_ADDRESS, ASTER_SIGNER_ADDRESS, ASTER_PRIVATE_KEY to .env
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
| [Architecture](docs/ARCHITECTURE.md) | System design and 5-pillar CLI structure |
| [Methodology](docs/METHODOLOGY.md) | Palmer, Monte Carlo, regime detection algorithms |
| [Technical Spec](docs/TECHNICAL_SPEC.md) | Data lineage, error handling, deployment |
| [Changelog](docs/CHANGELOG.md) | Version history and recent upgrades |

## API Keys Required

| Service | Purpose | Required? |
|---------|---------|-----------|
| [Alpaca](https://app.alpaca.markets/) | Brokerage, positions, options data, Greeks | Yes |
| [OpenAI](https://platform.openai.com/api-keys) or [OpenRouter](https://openrouter.ai/keys) | LLM analysis | For `--llm` features |
| [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) | Macro/economic time series | Yes |
| [FMP](https://financialmodelingprep.com/developer/docs/) | News, calendar, earnings, quotes | Yes |
| [Trading Economics](https://tradingeconomics.com/api/) | Consumer/macro indicators | Optional, falls back to FRED |
| [Aster DEX](https://app.asterdex.com/) | Crypto perps trading | For `crypto trade` only |

## License

[Add your license here]

<!-- TODO: Run git filter-repo or BFG Repo Cleaner to purge Investor list.xlsx from git history. -->
