## Lox — systematic macro + options research CLI (ML-ready regimes + LLM interface)

**Lox** is a research-grade CLI for building a **systematic macro trader**:
- compute **macro / liquidity / tariff** regimes as **scalar feature vectors**
- (next) train **ML forecasts** on those features
- use an **LLM interface** to summarize regimes, scenario-plan, and draft trade structures
- generate **execution-aware, risk-defined options ideas** (paper-first)

Guiding principles live in:
- `docs/PROJECT_CONSTITUTION.md`

## Quickstart

Install:

```bash
pip install -e .
```

Set environment variables (copy a `.env` if you use one locally):
- **Alpaca**: `ALPACA_API_KEY`, `ALPACA_API_SECRET` (and optionally `ALPACA_DATA_KEY`, `ALPACA_DATA_SECRET`)
- **FRED**: `FRED_API_KEY` (for macro/liquidity/tariff time series)
- **OpenAI**: `OPENAI_API_KEY` (for LLM summaries)

## Core commands (outputs)

Show all commands:

```bash
lox --help
```

### Regimes (dashboard + LLM summary)

Print macro + liquidity + tariff regimes:

```bash
lox regimes
```

Have an LLM summarize regimes + risks + follow-ups:

```bash
lox regimes --llm
```

### ML-friendly regime feature vector (JSON)

Print one merged scalar feature vector (good for ML training/export):

```bash
lox regime-features
```

### Macro

Print macro state + regime label (supports historical “as-of”):

```bash
lox macro snapshot
lox macro snapshot --asof 2022-06-30
```

Macro “stage” logic (high level):
- **inflation** if CPI YoY > target (default 3%)
- **stagflation** only if CPI YoY > target *and* job growth is negative (PAYEMS 3m annualized < 0)

### Liquidity (credit + rates)

```bash
lox liquidity snapshot
lox liquidity snapshot --features
```

Liquidity uses corporate credit spreads + rates dynamics (tightness score).

### Tariff / cost-push

List baskets:

```bash
lox tariff baskets
```

Snapshot for one basket:

```bash
lox tariff snapshot --basket import_retail_apparel
```

### Option leg selection (building block)

Pick a single “best” option leg (call/put) under liquidity + spread constraints:

```bash
lox select --ticker NVDA --sentiment positive --debug
```

### Ideas (thesis-driven)

Generate thesis-driven ideas (macro + tariff + tech sensitivity), optionally selecting option legs:

```bash
lox ideas ai-bubble
lox ideas ai-bubble --with-legs
```

### Tracking

```bash
lox track recent
lox track report
lox track sync
```

## Data baseline (default start date)
By default, regime datasets use a baseline start date of **2011-01-01**.

If you need fresh data:

```bash
lox regimes --refresh
```

## Roadmap
- `docs/OBJECTIVES.md`
