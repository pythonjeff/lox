## Avocado — systematic options idea generation & execution (Alpaca)

**Avocado** is a research and execution CLI designed to turn a macro thesis into **tradeable options expressions**:
- **Regime-aware context** (macro + tariff/cost-push)
- **Deterministic contract selection** (DTE / delta / spread / liquidity)
- **Explainability** (why an idea exists, what inputs drove it, and what to check next)
- **Paper execution + tracking** (log recommendations, link executed orders, track P&L)

**Important**: This is a research tool. Nothing here is financial advice. Use paper trading first.

## Quickstart

```bash
cd /path/to/repo
pip install -e .
cp .env.example .env
```

Minimum environment:
- **Alpaca**: `ALPACA_API_KEY`, `ALPACA_API_SECRET` (and optionally `ALPACA_DATA_KEY`, `ALPACA_DATA_SECRET`)
- **FRED**: `FRED_API_KEY` (for macro/tariff signals)
- **LLM (optional)**: `OPENAI_API_KEY`, `OPENAI_MODEL`
- **FMP (optional, recommended for news/calendar)**: `FMP_API_KEY`

## Core workflow (recommended)

### 1) Inspect regimes (macro + tariff) and optional LLM readout

```bash
avocado regimes
avocado regimes --llm
```

List available tariff baskets:

```bash
avocado tariff baskets
```

### 2) Generate thesis-driven ideas (AI bubble / inflation / tariffs)

```bash
avocado ideas ai-bubble --top 15
```

Pull option chains and select a liquid **starter leg** per idea:

```bash
avocado ideas ai-bubble --with-legs --top 10 --target-dte 45
```

### 3) Review ideas one-by-one and optionally execute (paper)

```bash
avocado ideas ai-bubble --interactive --with-legs --execute --top 10
```

Execution safety:
- Orders are only sent when you confirm interactively
- When `ALPACA_PAPER` is false, execution is refused

## Architecture

High-level pipeline:

```text
                ┌───────────────────────────────────────────────────────────┐
                │                         CLI (Typer)                       │
                │                   src/ai_options_trader/cli.py            │
                └───────────────┬───────────────────────────────┬──────────┘
                                │                               │
                                │                               │
                 ┌──────────────▼───────────────┐  ┌───────────▼───────────┐
                 │     Regimes / Datasets       │  │     Thesis → Ideas     │
                 │  macro/*  tariff/*  data/*   │  │  ideas/ai_bubble.py    │
                 └──────────────┬───────────────┘  └───────────┬───────────┘
                                │                               │
                                │                               │
                  ┌─────────────▼─────────────┐    ┌───────────▼───────────┐
                  │     Data Providers        │    │   Option Leg Selector  │
                  │  FRED (fred.py)           │    │ strategy/selector.py   │
                  │  Alpaca (market.py,       │    │ (filters + scoring +   │
                  │         alpaca.py)         │    │  sizing hooks)         │
                  └─────────────┬─────────────┘    └───────────┬───────────┘
                                │                               │
                                │                               │
                      ┌─────────▼─────────┐           ┌─────────▼─────────┐
                      │  Execution (opt)  │           │ Tracking (SQLite) │
                      │ execution/alpaca  │           │ tracking/store.py │
                      └─────────┬─────────┘           └─────────┬─────────┘
                                │                               │
                                └───────────────┬───────────────┘
                                                │
                                      ┌─────────▼─────────┐
                                      │ Reports / Sync     │
                                      │ `avocado track …`  │
                                      └────────────────────┘
```

Where the “brains” live:
- **Idea engine (direction + ranking + explainability)**: `src/ai_options_trader/ideas/ai_bubble.py`
- **Regimes**:
  - Macro dataset/state: `src/ai_options_trader/macro/signals.py`
  - Macro regime label: `src/ai_options_trader/macro/regime.py`
  - Tariff regime: `src/ai_options_trader/tariff/signals.py`
- **Option selector (filters + scoring)**: `src/ai_options_trader/strategy/selector.py`
- **Order submission (paper/live guardrails)**: `src/ai_options_trader/execution/alpaca.py`
- **Tracker DB**: `src/ai_options_trader/tracking/store.py`

## Tracking & reporting

Avocado logs:
- Every **recommendation** (per run id)
- Every **execution** (links Alpaca order id back to the recommendation)

Storage:
- Default SQLite DB: `data/tracker.sqlite3`
- Override: `AOT_TRACKER_DB=/path/to/tracker.sqlite3`

Commands:

```bash
avocado track recent --limit 20
avocado track sync --limit 50
avocado track report
```

## What to edit (the “brains”)

- **Idea engine (thesis → ranked tickers + direction + why)**: `src/ai_options_trader/ideas/ai_bubble.py`
- **Option selector (filters + scoring + sizing)**: `src/ai_options_trader/strategy/selector.py`
- **Risk sizing constraints**: `src/ai_options_trader/strategy/risk.py` and `src/ai_options_trader/config.py`
- **Data inputs**:
  - FRED: `src/ai_options_trader/data/fred.py`
  - Alpaca option snapshots / greeks: `src/ai_options_trader/data/alpaca.py`

## Command reference (high signal)

### Macro

```bash
avocado macro snapshot
avocado macro news --provider fmp --days 7
avocado macro outlook --provider fmp --days 7
avocado macro equity-sensitivity --tickers NVDA,AMD,MSFT,GOOGL --benchmark QQQ
avocado macro beta-adjusted-sensitivity --tickers NVDA,AMD,MSFT,GOOGL --benchmark QQQ
```

### News & sentiment (positions / tickers)

Summarize recent ticker-specific news (uses Alpaca open positions by default, or pass tickers explicitly):

```bash
avocado track news-brief --provider fmp --days 7 --tickers AAPL,MSFT
avocado track news-brief --provider fmp --days 7
```

Explain mode (includes reasons + evidence indices and prints indexed inputs):

```bash
avocado track news-brief --provider fmp --days 7 --tickers AAPL --explain
```

### Single-name leg selection

```bash
avocado select --ticker AAPL --sentiment positive --target-dte 30 --debug
```

### Tariff

```bash
avocado tariff snapshot --basket import_retail_apparel --benchmark XLY
avocado tariff baskets
```

## Notes & limitations

- **Option snapshot completeness varies**: OI/volume/greeks may be missing; filters are best-effort when fields are absent.
- **This repo is “legs-first”**: verticals/condors/etc. are the next layer (compose legs into spreads).
- **Cache**: FRED data is cached under `data/cache/` (recommended: do not commit cache artifacts).
- **News sources**:
  - Ticker news uses **FMP** `stock_news` (provider `fmp`) and filters to your lookback window.
  - Macro news uses **FMP** `general_news` (provider `fmp`) and topic-tags items heuristically.
- **Economic “What to watch”**: uses **FMP** `economic_calendar` when available for next release dates; falls back to official schedule scraping when needed.

## Roadmap

See `docs/OBJECTIVES.md`.
