# Demo Output: PM Morning Report

## `lox pm` Output

```
╭──────────────── LOX CAPITAL — PM MORNING REPORT  Mar 6 2026 ─────────────────╮
│  Risk: 56/100 (CAUTIOUS)  Quadrant: — MIXED                                  │
│  NAV: $13,306  P&L: $+2,906 (+27.9%)  [LIVE]                                 │
╰──────────────────────────────────────────────────────────────────────────────╯

[1] MACRO ENVIRONMENT
  Pillar                      Score            Δ7d    Regime
  Growth        █████░░░░░     54      ▼▼      +11    Stable Growth
  Inflation     █████░░░░░     56       —       -0    Elevated
  Volatility    ██████░░░░     67      ▼▼      +16    Elevated volatility
  Credit        ███░░░░░░░     40       —       -2    Credit Calm
  Rates         █████░░░░░     57       ▲       -6    Bear flattener
  Liquidity     ███████░░░     79      ▲▲       -9    Funding stress
  Consumer      ████████░░     80       —        0    Consumer Stress
  Fiscal        █████░░░░░     52       —       +1    Fiscal Drag
  Earnings      ███░░░░░░░     31       —        0    Earnings Expansion
  Oil/Cmdty     ███████░░░     71      ▼▼       +8    Commodity reflation

[2] SCENARIOS
    HIGH  TRADE WAR ESCALATION (4/4) → LONG GLD

[3] PORTFOLIO
  NAV: $13,306 Cash: 1.3% Delta: -201 Theta: $-52/day Vega: +506

  Bleeding: SLV260821P00070000  $-284 (-23.5%) → Cut or roll
  Winner:   HYG260717P00074000  $+128 (+39.0%) → Trail stop
  Winner:   SPY260618P00630000  $+903 (+38.1%) → Trail stop

  ⚠ Net short delta (-201) — exposed to market rally
  ✓ Long gamma (+53.93) — convexity in your favor
  ⚠ Significant theta decay ($-52/day) — time working against you
  ⚠ Theta burn $52/day vs $13,306 NAV = 39bp/day

[4] PM BRIEFING
  [Streaming LLM CIO brief — dense, opinionated, 250 words max]
```

---

## Key Design Principles

### Dense & Actionable
- **No verbose explanations** — every line earns its place
- **Position callouts** — bleeders (>15% down) and winners (>20% up) with action verbs
- **Risk signals** — auto-generated from Greeks analysis

### Regime-Aware
- **10-pillar heatmap** — scores, trend arrows, 7d deltas, key metrics
- **Scenario integration** — conviction-ranked macro scenarios with top trade

### LLM CIO Brief
- **On by default** — streaming output, no waiting
- **CIO persona** — opinionated, cites exact numbers, takes a side
- **250 word max** — every sentence has a purpose
- **Actionable** — ends with what to watch today
