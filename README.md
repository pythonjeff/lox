## Lox — systematic macro → regimes → options research (CLI, reproducible data, ML-ready features)

Lox is a **research-first** command-line toolkit for building a **systematic, regime-aware options workflow**:
- **Regime engines** (macro, liquidity, USD, tariff, fiscal, monetary) that emit **explainable labels + ML-friendly feature vectors**
- **Reproducible data** with on-disk caching for fast iteration
- **Execution-aware option selection** primitives (paper-first) and a lightweight tracking loop
- **LLM “research assistant” hooks** that summarize snapshots and draft scenario-based trade expressions (grounded in the snapshot JSON)

Guiding principles:
- `docs/PROJECT_CONSTITUTION.md`
- `docs/OBJECTIVES.md`

### Non-goals / guardrails
- **Not investment advice.** This is research software; outputs are informational.
- **No hidden data.** Snapshots are computed from explicit source series; LLM prompts are instructed to not invent facts.
- **Regime-first.** We prefer stable, low-noise signals over fragile “alpha stories.”

---

## Quickstart

### Install

```bash
pip install -e .
```

### Configure credentials

Set environment variables (or use a local `.env`):
- **Alpaca** (market data / options chain): `ALPACA_API_KEY`, `ALPACA_API_SECRET` (optional: `ALPACA_DATA_KEY`, `ALPACA_DATA_SECRET`)
- **FRED** (macro/fiscal/monetary series): `FRED_API_KEY`
- **OpenAI** (LLM summaries/outlooks): `OPENAI_API_KEY` (optional: `OPENAI_MODEL`)

---

## Core design (what you’re actually getting)

### Regimes as a production primitive
Each regime module aims to provide:
- **A snapshot**: latest readings with context markers (what’s “high/low” and what would be “stress watch”)
- **A regime label**: a stable, explainable classification
- **A feature vector**: scalars suitable for model training / export

### Data provenance & reproducibility
- **FRED** series are cached to `data/cache/fred/`.
- **FiscalData** tables are cached to `data/cache/fiscaldata/`.
- Local artifacts (caches, SQLite tracker) are excluded from git by `.gitignore`.

---

## Commands

Show all commands:

```bash
lox --help
```

### Regime dashboard + LLM summary

Print macro + liquidity + tariff regimes:

```bash
lox regimes
```

LLM summary (regimes + risks + follow-ups):

```bash
lox regimes --llm
```

### ML-friendly merged feature vector

```bash
lox regime-features
```

### Fiscal (MVP) + LLM outlook

```bash
lox fiscal snapshot
lox fiscal snapshot --refresh
lox fiscal outlook
```

Fiscal snapshot is designed to make market-absorption and liquidity plumbing visible:
- deficit level + impulse (% GDP)
- issuance mix (MSPD Δ outstanding)
- TGA behavior (level + change z-scores)
- auction absorption (tail proxy + dealer take)

### Monetary (MVP)

```bash
lox monetary snapshot
lox monetary snapshot --refresh
```

Monetary MVP focuses on “plumbing” signals:
- EFFR (DFF)
- total reserves (TOTRESNS) + **reserves/GDP**
- Fed balance sheet size (WALCL) + Δ
- ON RRP usage (RRPONTSYD)

### Macro

```bash
lox macro snapshot
lox macro snapshot --asof 2022-06-30
```

### Liquidity (credit + rates)

```bash
lox liquidity snapshot
lox liquidity snapshot --features
```

### USD + LLM outlook

```bash
lox usd snapshot
lox usd snapshot --features
lox usd outlook --year 2026
```

### Tariff / cost-push

```bash
lox tariff baskets
lox tariff snapshot --basket import_retail_apparel
```

### Ticker snapshot + LLM outlook (scenario-based)

```bash
lox ticker snapshot --ticker AAPL
lox ticker outlook --ticker AAPL --year 2026
```

### Option leg selection (building block)

```bash
lox select --ticker NVDA --sentiment positive --debug
```

### Ideas + tracking loop

```bash
lox ideas ai-bubble
lox ideas ai-bubble --with-legs

lox track recent
lox track report
lox track sync
```

---

## Refreshing data

Most snapshots support `--refresh` to force re-download. Example:

```bash
lox regimes --refresh
```

---

## Testing

```bash
pytest -q
```

---

## Roadmap (near-term)

- **Funding regime (next)**: repo/secured funding dislocations + SOFR spreads (explicitly *not* part of monetary MVP)
- Improve regime “handoffs”: fiscal ↔ monetary ↔ funding (plumbing-first)
- Expand ML workflows: dataset export, backtests, and disciplined evaluation
