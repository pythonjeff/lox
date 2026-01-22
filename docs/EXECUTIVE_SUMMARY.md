# Lox Capital — Technical Executive Summary

**One-Page Technical Overview for Reviewers**

---

## What Is This?

A **discretionary macro portfolio** with systematic research infrastructure. The platform provides quantitative regime analysis, risk simulation, and AI-powered insights while the portfolio manager makes all trade decisions.

---

## Key Quantitative Systems

### 1. Palmer (Macro Intelligence)

**What it does:** Real-time regime classification using macro indicators.

**How it works:**
- Inputs: VIX, HY OAS (credit spreads), 10Y Treasury yield
- Classification: Rules-based decision tree (no ML)
- Output: RISK-ON / CAUTIOUS / RISK-OFF

**Thresholds:**
| Regime | VIX | HY OAS |
|--------|-----|--------|
| RISK-ON | <18 | <325bp |
| CAUTIOUS | 18-25 | 325-400bp |
| RISK-OFF | >25 | >400bp |

**Update frequency:** 30 minutes (background refresh)

---

### 2. Monte Carlo Simulation

**What it does:** 6-month P&L distribution forecast with risk metrics.

**Model:**
- Equity: GBM with jump diffusion
- IV: Mean-reverting Ornstein-Uhlenbeck
- Correlation: -0.60 to -0.80 (return-vol negative skew)
- Position P&L: Taylor expansion (δ, ν, θ, γ)

**Risk outputs:**
- VaR 95%, VaR 99%, CVaR 95%
- Win probability
- Position-level CVaR attribution

**Regime conditioning:** Parameters (drift, vol, jump probability) vary by regime.

---

### 3. Stress Testing

**What it does:** Deterministic P&L under extreme scenarios.

**Predefined scenarios:**
| Scenario | SPX | VIX | Analog |
|----------|-----|-----|--------|
| Equity Crash | -20% | +25pts | COVID 2020 |
| Rates Shock | -5% | +5pts | Oct 2022 |
| Credit Event | -10% | +15pts | GFC-lite |
| Flash Crash | -10% | +30pts | Aug 2024 |

**Output:** Greek-decomposed P&L (delta, vega, theta, gamma attribution)

---

### 4. P&L Attribution

**Position-level decomposition:**
```
P&L = Δ_delta + Δ_vega + Δ_theta + Δ_gamma
```

**Greeks computed via Black-Scholes:** Standard closed-form for vanilla options.

---

## Data Sources

| Source | Data | Freshness |
|--------|------|-----------|
| FRED | Macro (HY OAS, yields) | T+1 |
| FMP | Quotes, news, calendar | Real-time |
| Alpaca | Positions, orders | Real-time |
| Trading Economics | Economic calendar | Real-time |
| OpenAI | LLM synthesis | 30-min cache |

---

## Architecture

```
CLI (Typer) ──┬── Regime classifiers ──┬── Dashboard (Flask)
              │                        │
              ├── Monte Carlo engine ──┤
              │                        │
              └── Greek calculators ───┘
                        │
                  Data layer (FRED, FMP, Alpaca)
```

**Deployment:** Heroku (dashboard) + local CLI

---

## Validation Status

| System | Status | Notes |
|--------|--------|-------|
| Regime classification | ✓ Rules-based | Deterministic, auditable |
| Monte Carlo | ✓ Implemented | Backtest validation planned |
| Greeks | ✓ Black-Scholes | Standard closed-form |
| Stress tests | ✓ Implemented | Historical analog calibration |
| LLM insights | ✓ GPT-4o-mini | Prompt deterministic, temp=0.3 |

---

## What's Missing (Roadmap)

1. **Backtest validation** — MC distributional coverage tests
2. **Historical regime accuracy** — Transition matrix analysis
3. **Factor model** — Multi-factor attribution beyond Greeks
4. **Real-time alerting** — Push notifications for regime changes

---

## Repository Structure

```
ai-options-trader-starter/
├── src/ai_options_trader/    # Core library
│   ├── cli_commands/         # CLI implementations
│   ├── portfolio/            # Positions, Greeks, stress tests
│   ├── llm/                  # Monte Carlo, LLM prompts
│   └── {macro,vol,rates}/    # Regime pillars
├── dashboard/                # Flask web app
├── docs/                     # Technical documentation
│   ├── METHODOLOGY.md        # Full formulas and algorithms
│   └── TECHNICAL_SPEC.md     # Architecture and ops
└── data/cache/               # Cached data files
```

---

**For full technical details:** See [METHODOLOGY.md](METHODOLOGY.md) and [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md)

**Live dashboard:** https://loxfund-284aa251b4f3.herokuapp.com

---

*Document version 1.0 | January 2026*
