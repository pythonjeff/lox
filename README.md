# Lox Capital — Discretionary Macro with Systematic Research

**A research platform for regime-aware portfolio management and tail-risk hedging**

---

## Philosophy

Lox Capital combines **discretionary portfolio management** with **systematic research tools**. Every trade decision is made by a human analyst after rigorous examination of macro regimes, quantitative metrics, and market context.

The platform provides:
- **Regime classification** across inflation, growth, liquidity, and volatility
- **Quantitative dashboards** with z-scores, percentiles, and historical context
- **PhD-level research briefs** synthesizing news, data, and scenario analysis
- **Trade idea generation** as starting points for further analysis

**The tools inform. The portfolio manager decides.**

---

## Performance

| Metric | Value |
|--------|-------|
| **Inception** | January 2026 |
| **Initial Capital** | $938 |
| **Current NAV** | $968 |
| **TWR (Since Inception)** | **+3.20%** |
| **Benchmark (SPY)** | +2.1% |
| **Strategy** | Discretionary macro with tail-risk hedging |

*Last updated: Jan 19, 2026 • Run `lox nav snapshot` for live NAV*

---

## Investment Process

### 1. Regime Assessment
Every morning begins with understanding the current macro environment:

```bash
lox dashboard                    # Unified view: inflation, growth, liquidity, vol
lox labs rates snapshot --llm    # PhD-level rates/curve analysis
lox labs vol --llm               # Volatility regime with scenario probabilities
```

The dashboard surfaces:
- **Inflation**: CPI trends, breakevens, sticky vs flexible components
- **Growth**: Payrolls momentum, claims, PMI signals
- **Liquidity**: Fed balance sheet, SOFR-IORB spreads, TGA dynamics
- **Volatility**: VIX level/percentile, term structure, mean-reversion signals

### 2. Deep Research
Before any trade, I conduct multi-source analysis:

```bash
lox labs commodities snapshot --llm   # Oil, gold, copper with news synthesis
lox labs growth snapshot --llm        # Employment trends with probability scenarios
lox monetary fedfunds-outlook         # Fed policy outlook with historical analogs
```

Each `--llm` report provides:
- **Probability-weighted scenarios** (Bull/Base/Bear with specific targets)
- **Event-driven catalyst calendar** (impact estimates for upcoming releases)
- **Cross-asset trade expressions** (sector ETFs, not just the obvious plays)
- **Historical analogs** with what's different this time
- **News synthesis** with source citations

### 3. Idea Generation
The platform generates trade ideas as **starting points for analysis**, not execution signals:

```bash
lox suggest --style defensive    # Ideas aligned to risk posture
lox labs hedge                   # Portfolio-aware defensive ideas
lox labs grow                    # Regime-aligned offensive ideas
```

Every idea is evaluated against:
- Current portfolio Greeks and exposures
- Regime alignment and conviction level
- Risk/reward with explicit targets and stops

### 4. Risk Analysis
Before execution, stress-test the portfolio:

```bash
lox labs mc-v01 --regime RISK_OFF --real    # Crash scenario (-25% drift)
lox labs mc-v01 --regime VOL_CRUSH --real   # Hedge drag scenario
lox labs mc-v01 --regime SLOW_BLEED --real  # Death by 1000 cuts
lox analyze --depth deep                     # Full position analysis
```

### 5. Execution Decision
After completing research, I make the final call:
- Size based on conviction and portfolio impact
- Entry triggers based on technical/catalyst timing
- Stops and targets informed by scenario analysis

**No trade is executed without completing this workflow.**

---

## Research Capabilities

### Regime Dashboard
```bash
lox dashboard                     # All pillars at a glance
lox dashboard --focus inflation   # Deep-dive with component breakdown
lox dashboard --features          # Export metrics as ML features
```

### LLM Research Briefs
```bash
lox labs vol --llm                # Volatility regime analysis
lox labs rates snapshot --llm     # Rates/curve analysis  
lox labs commodities snapshot --llm
lox labs inflation snapshot --llm
lox labs growth snapshot --llm
lox labs liquidity snapshot --llm
```

Each brief includes:
- Regime status with confidence level
- Key metrics with percentiles and z-scores
- News synthesis with citations [1], [2], etc.
- Scenario analysis with probability weights
- Cross-asset trade expressions with sector implications
- Historical context and analogs

### Monte Carlo Scenarios
```bash
lox labs mc-v01 --regime RISK_OFF --real     # -25% equity shock
lox labs mc-v01 --regime VOL_CRUSH --real    # VIX collapse to 10
lox labs mc-v01 --regime SLOW_BLEED --real   # Persistent -5% drift
lox labs mc-v01 --regime STAGFLATION --real  # Inflation + growth shock
lox labs mc-v01 --regime ALL --real          # 6-month baseline
```

---

## Quick Reference

### Daily Workflow
```bash
# Morning: Regime context (15 min)
lox status
lox dashboard
lox labs vol --llm

# Research: Deep analysis (as needed)
lox labs rates snapshot --llm
lox labs commodities snapshot --llm

# Ideas: Starting points for analysis
lox suggest --style defensive

# Risk: Pre-trade stress test
lox labs mc-v01 --regime RISK_OFF --real

# EOD: Record NAV
lox nav snapshot
```

### Portfolio Commands
| Command | Purpose |
|---------|---------|
| `lox status` | Portfolio health at a glance |
| `lox status -v` | With position details |
| `lox analyze --depth deep` | Full LLM analysis |
| `lox nav snapshot` | Record current NAV |
| `lox account summary` | Account + positions + P&L |

### Research Commands
| Command | Purpose |
|---------|---------|
| `lox dashboard` | All regime pillars |
| `lox labs vol --llm` | Volatility research brief |
| `lox labs rates snapshot --llm` | Rates/curve analysis |
| `lox labs mc-v01 --real` | Monte Carlo scenarios |
| `lox monetary fedfunds-outlook` | Fed policy outlook |

---

## Strategy

### Mandate
Discretionary macro portfolio management with systematic research support. Primary focus on tail-risk hedging with positive carry during calm periods.

### Thesis
- Persistent inflation risk above 2010s baseline
- Rising macro volatility from fiscal/rates dynamics
- Structural shift in Treasury issuance → higher vol premium
- Regime awareness improves timing and sizing

### Implementation
- **Long convexity**: OTM puts on SPY/QQQ, VIX calls
- **Delta management**: Adjust exposures based on regime signals
- **Carry optimization**: Time spreads and premium selling in favorable regimes
- **Cross-asset**: Sector rotation based on rate/vol/growth implications

### Risk Limits
| Metric | Limit |
|--------|-------|
| Max position size | 15% NAV |
| Max portfolio delta | ±30% |
| Max theta decay | 2% NAV/month |
| VaR 95% (3M) | <10% |

---

## Data Sources

| Source | Data |
|--------|------|
| **FRED** | Macro time series (rates, employment, inflation) |
| **Alpaca** | Real-time quotes, positions, news with full content |
| **FMP** | Stock news, economic calendar, company data |
| **CBOE** | VIX, term structure, options data |

The LLM analyst aggregates these sources into research briefs with proper citations.

---

## Installation

```bash
git clone <repo>
cd ai-options-trader-starter
pip install -e .

# Configure API keys
cat > .env << EOF
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_PAPER=true
FRED_API_KEY=your_fred_key
FMP_API_KEY=your_fmp_key
OPENAI_API_KEY=your_openai_key
EOF

# Verify
lox status
lox dashboard
```

---

## Architecture

```
lox/
├── Research Layer
│   ├── dashboard      → Unified regime classification
│   ├── labs *         → PhD-level research briefs (--llm)
│   └── mc-v01         → Monte Carlo scenario analysis
│
├── Portfolio Layer
│   ├── status         → Fast portfolio health
│   ├── analyze        → Risk analysis
│   ├── suggest        → Trade idea generation
│   └── nav            → NAV tracking & investor reporting
│
├── Data Layer
│   ├── FRED           → Macro time series
│   ├── Alpaca         → Market data + news
│   ├── FMP            → News + calendar + quotes
│   └── Unified News   → Aggregated + deduplicated
│
└── Analysis Layer
    ├── Regime pillars → Inflation, growth, liquidity, vol
    ├── Sector maps    → Cross-asset implications
    └── LLM analyst    → Research synthesis + scenarios
```

---

**Lox Capital** | Discretionary Macro | Research-Driven | Since Jan 2026

*The platform provides research and analysis tools. All investment decisions are made by the portfolio manager after independent evaluation. Not investment advice.*
