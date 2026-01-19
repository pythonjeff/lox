# Lox Capital — Systematic Options & Tail-Risk Hedging

**A quantitative trading system for regime-aware options portfolio management**

---

## Performance

| Metric | Value |
|--------|-------|
| **Inception** | January 2026 |
| **Initial Capital** | $938 |
| **Current NAV** | $968 |
| **TWR (Since Inception)** | **+3.20%** |
| **Benchmark (SPY)** | +2.1% |
| **Strategy** | Tail-risk hedging with convex payoffs |

*Last updated: Jan 19, 2026 • Run `lox nav snapshot` for live NAV*

---

## Quick Start

```bash
# Install
pip install -e .

# Configure .env with API keys (Alpaca, FRED, FMP, OpenAI)

# Check portfolio health
lox status

# Get trade ideas
lox suggest --style defensive

# Full macro dashboard
lox dashboard
```

---

## Core Commands

### Portfolio Commands
```bash
lox status              # Portfolio health at a glance (2s)
lox status -v           # With position details
lox report              # Investor-ready summary
lox analyze             # Quick risk analysis
lox analyze --depth deep # Full LLM analysis
lox suggest             # Trade suggestions
lox run                 # Full workflow (status → analyze → suggest)
```

### Regime Dashboard
```bash
lox dashboard                     # All pillars: inflation, growth, liquidity, vol
lox dashboard --focus inflation   # Deep-dive: CPI, breakevens, stickiness
lox dashboard --focus liquidity   # Deep-dive: Fed balance sheet, SOFR, TGA
lox dashboard --focus growth      # Deep-dive: payrolls, claims, unemployment
lox dashboard --focus volatility  # Deep-dive: VIX, term structure, skew
lox dashboard --features          # Export ML features as JSON
```

### Monte Carlo Risk Analysis
```bash
lox labs mc-v01 --regime RISK_OFF --real     # Crash scenario (-25% drift)
lox labs mc-v01 --regime VOL_CRUSH --real    # Worst for hedges (VIX→10)
lox labs mc-v01 --regime SLOW_BLEED --real   # Death by 1000 cuts
lox labs mc-v01 --regime ALL --real          # Baseline 6-month outlook
```

### Trade Ideas
```bash
lox labs hedge          # Defensive ideas (portfolio-aware)
lox labs grow           # Offensive ideas (regime-aligned)
lox options scan SPY    # Options strike ladder with delta
```

---

## Architecture

```
lox/
├── Core Commands
│   ├── status      → Fast portfolio health
│   ├── report      → Investor summary
│   ├── analyze     → Risk analysis
│   ├── suggest     → Trade ideas
│   └── run         → Full workflow
│
├── Dashboard
│   └── dashboard   → Unified regime view
│       ├── Inflation (CPI, breakevens, stickiness)
│       ├── Growth (payrolls, claims, unemployment)
│       ├── Liquidity (Fed balance sheet, SOFR, TGA)
│       └── Volatility (VIX, term structure)
│
├── Labs (Power User)
│   ├── mc-v01      → Monte Carlo with 8 regime scenarios
│   ├── hedge       → Portfolio-aware defensive ideas
│   ├── grow        → Regime-aligned offensive ideas
│   └── fedfunds-outlook → PhD-level macro analysis
│
└── Options
    ├── scan        → Strike ladder with Greeks
    ├── moonshot    → High-variance scanner
    └── pick        → Budget-constrained selection
```

---

## Strategy

### Mandate
Systematic tail-risk hedging with positive carry during calm periods.

### Thesis
- Persistent inflation risk above 2010s baseline
- Rising macro volatility from fiscal/rates dynamics
- Structural shift in Treasury issuance → higher vol premium

### Implementation
- **Long convexity**: OTM puts on SPY/QQQ, VIX calls
- **Delta hedge**: Short equity/credit to offset directional exposure
- **Carry optimization**: Time spreads and selling near-the-money premium
- **Regime-aware**: Adjust Greeks based on macro/funding/vol regimes

### Risk Limits
| Metric | Limit |
|--------|-------|
| Max position size | 15% NAV |
| Max portfolio delta | ±30% |
| Max theta decay | 2% NAV/month |
| VaR 95% (3M) | <10% |

---

## Installation

```bash
# Clone and install
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

## Daily Workflow

```bash
# Morning: Context
lox status
lox dashboard

# Midday: Ideas  
lox suggest --style defensive
lox labs hedge

# EOD: Risk
lox labs mc-v01 --regime RISK_OFF --real
lox nav snapshot
```

---

## ML Features

The dashboard exports all pillar metrics as ML features:

```bash
lox dashboard --features > features.json
```

Output includes:
- `inflation_cpi_yoy_level`, `inflation_cpi_yoy_zscore`, `inflation_cpi_yoy_mom_3m`
- `growth_payrolls_3m_ann_level`, `growth_unemployment_rate_level`
- `liquidity_net_liquidity_level`, `liquidity_sofr_iorb_spread_level`
- `volatility_vix_level`, `volatility_vix_percentile`, `volatility_term_structure_level`

Use for regime-conditional Monte Carlo, backtesting, or custom models.

---

## Command Reference

### Quick Access
| Command | What it does |
|---------|-------------|
| `lox status` | Portfolio health (fast, no LLM) |
| `lox dashboard` | All regimes at a glance |
| `lox suggest` | Quick trade ideas |
| `lox labs hedge` | Defensive ideas |
| `lox labs mc-v01 --real` | Monte Carlo risk analysis |

### Full Analysis
| Command | What it does |
|---------|-------------|
| `lox analyze --depth deep` | LLM-powered analysis |
| `lox run` | Full workflow |
| `lox autopilot run-once --engine ml` | ML trade generation |
| `lox monetary fedfunds-outlook` | PhD-level macro dashboard |

### NAV & Reporting
| Command | What it does |
|---------|-------------|
| `lox nav snapshot` | Record current NAV |
| `lox nav investor contribute` | Record investor flow |
| `lox weekly report` | Weekly performance summary |
| `lox account summary` | Account + positions + P&L |

---

## Testing

```bash
pytest -q                           # Full suite
pytest tests/test_macro_playbook.py # Regime tests
```

---

## Documentation

- `docs/OBJECTIVES.md` — Strategy objectives
- `docs/PROJECT_CONSTITUTION.md` — Design principles
- `docs/MONTE_CARLO_V01_SUMMARY.md` — MC methodology
- `docs/COMMANDS_GUIDE.md` — Full command reference

---

**Lox Capital** | Systematic Options | Tail-Risk Hedging | Since Jan 2026

*Not investment advice. Research software for educational purposes.*
