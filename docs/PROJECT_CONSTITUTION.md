# Project Constitution — Lox (systematic macro trader)

## North Star
Build a **systematic macro trading research system** that:
- turns “world state” into **regime features (scalars)**
- uses **ML** to forecast forward returns/risk with uncertainty
- uses an **LLM interface** to summarize, scenario-plan, and draft trade structures
- produces **execution-aware, risk-defined options ideas** (initially paper)

This project is explicitly meant to build “real desk” skills: clean data, no leakage, repeatable evaluation, and risk-aware execution.

## Core loop (always)
**Data → Features → Forecast → Portfolio/Trades → Execution → Logging → Evaluation → Iteration**

- **Data**: FRED + broker market data + (optional) macro/news feeds.
- **Features**: flat `dict[str, float]` regime vectors (ML-friendly).
- **Forecast (ML)**: probabilistic forward-looking outputs (direction, magnitude, uncertainty).
- **Trades**: map forecasts → option structures + sizing under constraints.
- **Execution**: preview first; paper-trade optional; real trading only after robust evaluation.
- **Logging**: store features, forecasts, decisions, and outcomes.
- **Evaluation**: walk-forward testing, leakage checks, stability checks, and drawdown control.

## Non-negotiables (systematic standards)
- **No leakage**: every computation must be valid “as of” the timestamp used.
- **Reproducibility**: same inputs must produce the same outputs (cache + deterministic transforms).
- **Modularity**: signals/features, models, and strategies are pluggable.
- **Risk-first**: position sizing, exposure caps, and kill-switch logic are part of every strategy.
- **Liquidity-aware**: trade selection must respect spreads/volume/OI and avoid untradeable ideas.
- **Auditability**: every recommendation should be explainable back to inputs + model output.

## What an LLM is allowed to do (and not do)
### LLM responsibilities
- Summarize current regimes and highlight key drivers.
- Propose 2–3 scenarios (“if X then Y”) and what would invalidate them.
- Draft trade structures consistent with forecast + risk constraints.
- Produce copy/pasteable reports.

### LLM is NOT the oracle
- It should not be the sole source of “direction.” Predictions must be tied to measurable signals/ML forecasts and validated historically.

## Strategy scope (phased)
### Phase 1 — Macro “pod” instruments (liquid, interpretable)
Prefer a small macro set first (ETFs / highly liquid underlyings):
- Broad risk: SPY / QQQ / IWM
- Duration: TLT / IEF
- Credit proxies: HYG / LQD
- Dollar: UUP
- Commodities: GLD (and others later)

### Phase 2 — Expand to sectors / then single names
Only expand universe after:
- the forecast target is defined
- the backtest harness is in place
- the liquidity/transaction-cost model is credible

## Current primitives (today)
- **Regime feature vectors**: `lox regime-features` prints a single flat feature map for ML.
- **Macro regime**: uses CPI target and payroll growth to label `inflation` vs `stagflation` (and falls back to a quantitative 2×2 with guardrails).
- **Liquidity regime**: credit spreads + rates behavior (tightness score).
- **Tariff regime**: cost-pressure momentum + equity denial (per basket).

## Definition of “progress”
We prioritize improvements that are:
- measurable (improve a metric or reduce risk)
- repeatable (not one-off prompts)
- incremental (small changes, frequent evaluation)

## Default guardrails (until proven otherwise)
- Paper trading by default.
- Defined-risk option structures.
- Conservative sizing.
- Avoid thin markets.


