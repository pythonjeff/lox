# Trade Ideas Commands

Two simple commands for regime-aware trade recommendations.

---

## `lox labs hedge` - Defensive Ideas

**Protection, tail risk, hedges**

```bash
lox labs hedge
```

**Example Output:**

```
═══ Lox Hedge Recommendations ═══

Current Regime:
  Macro        STAGFLATION
  Liquidity    FRAGILE (RRP depleted)
  Volatility   LOW (VIX 14.2 - hedges cheap)

Your Portfolio:
  Long equity  2 positions
  Put hedges   3 contracts
  Vol exposure 1 positions

Portfolio Gaps:
  • No credit hedge (HYG puts)
  • Vol is cheap (add VIX exposure)

═══ Recommended Hedges ═══

#  Trade                      Why                                    Payoff If                     Cost
1  SPY 3M puts (10% OTM)      Tail protection in stagflation         SPY drops 15%+ → 5-10x       ~$150
2  HYG 3M puts (5% OTM)       Credit spreads widen                   Credit stress → 3-5x         ~$100
3  VIX 1M calls (25 strike)   Vol is cheap, expect spikes           VIX >30 → 10x+               ~$75

To execute:
  • Run: lox options recommend --ticker SPY --direction bearish
  • Or manually enter trades in your broker
```

---

## `lox labs grow` - Offensive Ideas

**Growth, momentum, opportunities**

```bash
lox labs grow
```

**Example Output:**

```
═══ Lox Growth Opportunities ═══

Current Regime:
  Macro        GOLDILOCKS
  Liquidity    ORDERLY
  Volatility   LOW (VIX 13.8 - calm markets)

═══ Recommended Opportunities ═══

#  Trade                           Why                                   Target                          Allocation
1  Tech / growth equities          Low inflation + strong growth         Look at: QQQ, XLK, ARKK, SMH    ~$200 (40%)
2  Sell short-dated premium        Low vol = high theta                  Sell covered calls on longs     ~$100 (20%)
3  Long-dated calls on quality     Vol is cheap, extend duration         SPY, QQQ 6M calls (ATM)        ~$125 (25%)

To execute:
  • Run: lox options recommend --ticker QQQ --direction bullish
  • Or manually enter trades in your broker
```

---

## Key Features

### ✅ Regime-Aware
Automatically adjusts based on:
- **Macro regime**: STAGFLATION, GOLDILOCKS, INFLATIONARY, DISINFLATIONARY
- **Liquidity**: FRAGILE vs ORDERLY
- **Volatility**: VIX cheap vs expensive

### ✅ Portfolio-Aware (hedge only)
`lox labs hedge` analyzes your positions to find gaps:
- Missing downside protection
- Over-hedged (theta drag)
- No vol exposure when VIX is cheap
- No credit hedge when spreads are tight

### ✅ Flexible Output
Can show:
- **Specific picks**: "SPY 3M puts (10% OTM)"
- **Directional guidance**: "Look at: XLE, USO, DBC"
- **Sector rotation**: "Tech / growth equities → QQQ, XLK"

### ✅ Budget-Aware
```bash
lox labs hedge --budget 1000     # Allocate $1000 to hedges
lox labs grow --budget 2000      # Allocate $2000 to growth
```

---

## Usage Examples

### Quick check for hedge ideas
```bash
lox labs hedge
```

### Large hedge budget
```bash
lox labs hedge --budget 5000 --max 5
```

### Daily growth scan
```bash
lox labs grow
```

### Aggressive growth allocation
```bash
lox labs grow --budget 10000 --max 6
```

---

## Comparison to Old Autopilot

| Feature | Old `autopilot run-once --llm --llm-news` | New `hedge` / `grow` |
|---------|-------------------------------------------|----------------------|
| Output length | 300+ lines | 25 lines |
| Time to run | 5+ minutes | Instant |
| Interactive prompts | Yes (every position) | No |
| LLM reasoning | Verbose, per-position | Regime-level only |
| Actionable | Hard to parse | Clear 3 trades |
| Use case | Full portfolio review | Quick daily ideas |

---

## Integration with Existing Commands

### 1. Get ideas
```bash
lox labs hedge          # Defensive ideas
lox labs grow           # Offensive ideas
```

### 2. Get specific contracts
```bash
lox options recommend --ticker SPY --direction bearish
lox options recommend --ticker QQQ --direction bullish
```

### 3. Full portfolio review (if needed)
```bash
lox account summary
lox labs mc-v01 --regime RISK_OFF --real
```

---

## Regime → Trade Mapping

### STAGFLATION
- **Hedge**: SPY puts, HYG puts, VIX calls
- **Grow**: Commodities (GLD, DBC), defensives (XLU, XLP)

### GOLDILOCKS
- **Hedge**: Trim hedges, sell premium, buy long-dated protection
- **Grow**: Tech (QQQ, XLK), long-dated calls, momentum

### INFLATIONARY
- **Hedge**: TLT puts, credit protection
- **Grow**: Energy (XLE), gold (GLDM), short bonds (TBT)

### DISINFLATIONARY
- **Hedge**: SPY puts (recession risk), tail protection
- **Grow**: Long duration bonds (TLT), quality growth at discount

---

## Next Steps

1. ✅ Commands implemented and working
2. ⏳ Test in production with real FRED/FMP keys
3. ⏳ Add LLM narrative (optional) for "why this regime"
4. ⏳ Integrate with ML scoring engine for ranking ideas
