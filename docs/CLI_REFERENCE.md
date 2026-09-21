# CLI Reference

Full command reference for the Lox CLI.

---

## CLI Structure

```
lox                           # Help with examples
|
+-- PM Morning Report
|   +-- pm                    # Daily CIO briefing (macro + portfolio + LLM)
|
+-- Research
|   +-- research regimes      # Unified regime overview (12 pillars)
|   +-- research ticker NVDA  # Deep ticker research
|   +-- research portfolio    # Open position outlook
|   +-- research scenario SPY # Monte Carlo macro shock sim
|   +-- research chat         # Interactive research chat
|
+-- Regime Drill-Down
|   +-- regime composite      # Composite regime ID (5 macro regimes + playbook)
|   +-- regime growth         # Growth (payrolls, ISM, claims)
|   +-- regime inflation      # Inflation (CPI, PCE, breakevens)
|   +-- regime vol            # Volatility (VIX)
|   +-- regime credit         # Credit spreads (HY OAS, BBB, AAA)
|   +-- regime rates          # Yield curve
|   +-- regime funding        # Funding markets (SOFR)
|   +-- regime consumer       # Consumer (sentiment, spending)
|   +-- regime fiscal         # Fiscal (deficits, TGA)
|   +-- regime earnings       # Earnings (beat rate, revisions)
|   +-- regime oil            # Commodities (oil, gold, copper)
|   +-- regime usd            # USD strength (trade-weighted, FX vol)
|
+-- Idea Generation
|   +-- movers                # Names that move a lot, often + trade structure
|   +-- suggest               # Today's opportunity scan (momentum/flow/regime)
|
+-- Portfolio & Risk
|   +-- status                # Portfolio health
|   +-- risk                  # Greeks dashboard
|   +-- scan -t TICKER        # Options chain scanner
|
+-- Accounting
|   +-- account               # Alpaca account details
|   +-- nav                   # NAV management
|   +-- weekly                # Investor reports
|
+-- Crypto
|   +-- crypto data           # Market data + technicals
|   +-- crypto research       # LLM analysis
|   +-- crypto trade          # Order execution
```

---

## Core Commands

| Command | Purpose |
|---------|---------|
| `lox pm` | PM Morning Report — macro + portfolio + LLM briefing |
| `lox status` | Portfolio health at a glance |
| `lox risk` | Greeks dashboard + theta breakeven |
| `lox scan -t TICKER` | Options chain scanner |
| `lox movers` | Rank the universe by how much it moves, and how often |
| `lox suggest` | Opportunity scan across S&P 500 + Dow + macro ETFs |
| `lox research ticker TICKER` | Deep ticker research |
| `lox research regimes` | Unified regime overview |

---

## PM Morning Report

```bash
lox pm                # Full report with streaming LLM CIO brief (default)
lox pm --no-llm       # Data sections only (macro + scenarios + portfolio)
lox pm --json         # Machine-readable JSON output
```

Sections: Macro Environment (12-pillar heatmap + composite regime headline), Active Scenarios, Portfolio (NAV/Greeks/bleeders/winners/risk signals), LLM CIO Briefing.

---

## Research Commands

| Command | Purpose |
|---------|---------|
| `lox research ticker AAPL` | Full research: momentum, HF metrics, fundamentals |
| `lox research ticker NVDA --llm` | With LLM synthesis |
| `lox research regimes` | Unified regime overview |
| `lox research regimes --llm` | With AI commentary |
| `lox research regimes --trend` | Trend dashboard (sparklines, momentum) |
| `lox research regimes -d vol` | Drill into volatility regime |
| `lox research regimes --book` | Position exposure vs regimes |
| `lox research portfolio` | LLM outlook on open positions |
| `lox research scenario SPY` | Monte Carlo macro shock sim |

---

## Regime Commands

```bash
# Composite regime (hedge-fund-style macro regime ID)
lox regime composite              # Full dashboard: regime, confidence, playbook, swing factors
lox regime composite --json       # Machine-readable JSON output

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
lox regime earnings               # Earnings (S&P 500 beat rate, revisions)
lox regime oil                    # Oil/commodities (energy, metals)
lox regime usd                    # USD strength (trade-weighted dollar, FX vol, EM/commodity impact)

# Add --llm to any command for LLM analysis
lox regime vol --llm

# Add --book for position impact analysis
lox regime credit --book
```

---

## Idea Generation

Two screens, asking different questions.

### `lox movers` — what actually moves

`lox suggest` prefilters on a single session's change and volume, so it surfaces
today's news pops. `lox movers` asks the prior question: **which names reliably
deliver range**, measured over a rolling window, so there is something to trade
in the first place. Then it maps each name onto a trade structure.

```bash
lox movers                       # Top 20 movers across S&P 500 + Dow + macro ETFs
lox movers -n 30                 # Deeper list
lox movers --move 0.03           # Only count 3%+ days as "a move"
lox movers --window 120          # Measure over ~6 months instead of ~3
lox movers --character TRENDER   # Only names whose moves stick (directional trades)
lox movers --character CHOPPY    # Only range traders (long/short vol)
lox movers --etf-only            # Macro ETFs only
lox movers --universe core       # ~30 liquid macro ETFs (fast, few API calls)
lox movers -t NVDA               # Single-name movement profile
lox movers --llm                 # Hand the screen to the analyst chat
lox movers --json                # Machine-readable
```

**What it measures** (over the window, from daily closes):

| Metric | Meaning |
|--------|---------|
| Avg/d | Mean absolute daily move — the everyday size of a move |
| Freq | Share of sessions clearing the move bar (`--move`, default 2%) |
| Persistence | Does it clear the bar in *every* sub-window, or was it one gap? |
| E[1m] | 1-sigma expected move over ~21 sessions — what an option has to beat |
| Vol | 20d vs 60d realized vol — is the name waking up right now? |
| Efficiency | Net move / total travel — trending mover vs chop |

**Character → trade structure:**

| Character | What it means | Structure |
|-----------|---------------|-----------|
| `TRENDER` | Moves stick — high trend efficiency | Directional debit spread, 30-45 DTE |
| `CHOPPY` + vol expanding | Big range, no direction, waking up | Long strangle / straddle, 21-45 DTE |
| `CHOPPY` + vol contracting | Big range, no direction, vol fading | Iron condor outside the expected move |
| `GAPPER` | Range concentrated in a few sessions | Event risk — define risk, don't hold naked premium |
| `QUIET` | Not enough movement to pay for anything | No trade |

Every row ends in a `lox scan` command for the matching side and DTE band, so the
handoff from idea to contract is a copy-paste.

**Filters:** `--min-price` (default $5) and `--min-dollar-volume` (default $20M/day)
gate on tradeability before any history is pulled — a name that moves but can't be
filled is not an opportunity.

**Cost:** one batch quote request for the whole universe, then price history for the
`--pool` survivors only (default 120). Use `--universe core` for a near-free scan.

### `lox suggest` — what's happening today

```bash
lox suggest                      # Full scanner (S&P 500 + Dow + ETFs)
lox suggest --signal flow        # Flow acceleration candidates only
lox suggest --signal tailwind    # Regime tailwind candidates only
lox suggest --etf-only           # Exclude individual stocks
lox suggest --deep               # Monte Carlo + extended flow analysis
lox suggest --track-record       # Suggestion performance dashboard
lox suggest -t AAPL              # Single-ticker deep dive
```

Scores four pillars — momentum, flow, regime alignment, catalyst — with
regime-conditional weights and anti-staleness rotation.

---

## Options Commands

| Command | Purpose |
|---------|---------|
| `lox scan -t NVDA` | Options chain (default: puts, 30-365 DTE) |
| `lox scan -t CRWV --want put --min-days 100` | Custom DTE range |

---

## Interactive Chat

| Command | Purpose |
|---------|---------|
| `lox chat` | Interactive research chat |
| `lox chat -t AAPL` | Ticker-focused chat |
| `lox chat -c fiscal` | Chat with regime context |
| `lox chat -c regimes` | Chat with all regime data |

---

## Daily Workflow

```bash
# Morning: PM Report (2 min)
lox pm                        # Full CIO briefing with LLM
lox regime composite          # Composite regime + playbook

# Drill-down (as needed)
lox regime vol                # Volatility deep dive
lox regime credit --book      # Credit + position exposure
lox regime usd --llm          # USD regime + FX analysis
lox research ticker NVDA      # Full research report

# Idea generation
lox movers                    # What is actually moving, and how to trade it
lox movers --character TRENDER   # Directional candidates only
lox suggest                   # What is happening today

# Options scanning
lox scan -t CRWV --want put
lox scan -t NVDA --min-days 60 --max-days 180

# Scenarios
lox research scenario SPY     # Monte Carlo macro shock sim

# EOD
lox nav snapshot              # Record NAV
```

---

## Dashboard Access

| Environment | URL | Command |
|------------|-----|---------|
| **Production** | [loxfund.com](https://loxfund.com) | Direct browser access |
| **Local** | http://localhost:5001 | `cd dashboard && python app.py` |
