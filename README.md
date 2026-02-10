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

| Service | Purpose |
|---------|---------|
| [Alpaca](https://app.alpaca.markets/) | Brokerage, positions, market data |
| [OpenAI](https://platform.openai.com/api-keys) | LLM analysis |
| [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) | Macro/economic time series |
| [FMP](https://financialmodelingprep.com/developer/docs/) | News, calendar, quotes |
| [Trading Economics](https://tradingeconomics.com/api/) | Consumer/macro indicators (optional, falls back to FRED) |

## License

[Add your license here]

<!-- TODO: Run git filter-repo or BFG Repo Cleaner to purge Investor list.xlsx from git history. -->
