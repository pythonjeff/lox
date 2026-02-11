# Lox

Discretionary macro research platform with systematic tools for regime-aware portfolio management and tail-risk hedging.

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

## Core Commands

```bash
lox status                    # Portfolio health
lox dashboard                 # All regimes at a glance
lox regime unified            # All 12 regimes + MC adjustments
lox regime growth             # Growth regime
lox regime vol                # Volatility regime
lox research ticker NVDA      # Full research report
lox scan -t NVDA --want put   # Options chain
lox research regimes          # All regimes overview
```

## Crypto Perps

Real-time crypto perpetual futures data, LLM-powered analysis, and manual trading via Aster DEX.

No API keys needed for market data — uses public CCXT endpoints (OKX by default).

```bash
# Market data & technicals
lox crypto data                          # BTC, ETH, SOL overview + technicals
lox crypto data --coins BTC,ETH,DOGE     # Custom coin list

# LLM analysis
lox crypto research                      # Data + macro regime LLM analysis
lox crypto analyze                       # Perps-specific trading analysis (alpha-arena style)
lox crypto analyze --coins BTC           # Single coin deep dive

# Trading (requires Aster DEX config in .env)
lox crypto balance                       # Account balance & equity
lox crypto positions                     # Open positions with PnL
lox crypto trade BTC BUY 0.001 --leverage 5   # Place a trade
lox crypto close BTC                     # Close a position
```

To enable trading, install with the trading extras and set Aster DEX env vars:
```bash
pip install -e ".[trading]"
# Then add ASTER_USER_ADDRESS, ASTER_SIGNER_ADDRESS, ASTER_PRIVATE_KEY to .env
```

## Dashboard

```bash
cd dashboard && python app.py
# Navigate to http://localhost:5001
```

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
| [Alpaca](https://app.alpaca.markets/) | Brokerage, positions, market data | Yes |
| [OpenAI](https://platform.openai.com/api-keys) or [OpenRouter](https://openrouter.ai/keys) | LLM analysis | For `--llm` features |
| [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) | Macro/economic time series | Yes |
| [FMP](https://financialmodelingprep.com/developer/docs/) | News, calendar, quotes | Yes |
| [Trading Economics](https://tradingeconomics.com/api/) | Consumer/macro indicators | Optional, falls back to FRED |
| [Aster DEX](https://app.asterdex.com/) | Crypto perps trading | For `crypto trade` only |

## License

[Add your license here]

<!-- TODO: Run git filter-repo or BFG Repo Cleaner to purge Investor list.xlsx from git history. -->
