# Simplified Hedge Recommendation System

## Problem with Current `autopilot`

The `lox autopilot run-once --engine ml --basket extended --llm --llm-news` command is:
- ❌ Too verbose (300+ lines of output)
- ❌ Reviews every position with full LLM reasoning
- ❌ Asks if you want to close each one
- ❌ Then suggests new trades with more verbose reasoning
- ❌ **Takes 5+ minutes and is hard to parse**

## New Solution: `lox labs hedge-ideas`

### What It Does

**One simple command:**
```bash
lox labs hedge-ideas
```

**Clean output (25 lines):**
```
═══ Lox Hedge Recommendations ═══

Current Regime:
  Macro        STAGFLATION
  Liquidity    FRAGILE (RRP depleted)
  Volatility   LOW (VIX 14.2 - hedges cheap)

Your Portfolio:
  Long equity  2 positions
  Short equity 0 positions
  Put hedges   3 contracts
  Call hedges  1 contracts
  Vol exposure 1 positions

Portfolio Gaps:
  • No credit hedge (HYG puts)
  • Vol is cheap (add VIX exposure)

═══ Recommended Hedges ═══

#  Trade                   Why                                  Payoff If                  Cost
1  SPY 3M puts (10% OTM)   Tail protection in weak growth +     SPY drops 15%+ → hedge     ~$150 (30% of budget)
                           high inflation                        pays 5-10x                 
2  HYG 3M puts (5% OTM)    Credit spreads widen in stagflation  Credit stress → hedge      ~$100 (20% of budget)
                                                                pays 3-5x                  
3  VIX 1M calls (25        Vol is cheap, stagflation → vol      VIX >30 → hedge pays 10x+  ~$75 (15% of budget)
   strike)                 spikes                                                          

To execute:
  1. Review ideas above
  2. Run: lox options recommend --ticker <TICKER> --direction <bearish/bullish>
  3. Or: lox autopilot run-once --execute (for ML-driven execution)
```

### Key Features

1. **Regime-Aware**: Automatically adjusts recommendations based on:
   - Macro regime (STAGFLATION, GOLDILOCKS, INFLATIONARY, DISINFLATIONARY)
   - Liquidity state (FRAGILE vs ORDERLY)
   - VIX level (cheap vs expensive hedges)

2. **Portfolio-Aware**: Analyzes your current positions to find gaps:
   - Missing downside protection
   - Over-hedged (theta drag)
   - No vol exposure when VIX is cheap
   - No credit hedge when spreads are tight

3. **Actionable**: Each idea includes:
   - Specific trade (SPY 3M puts 10% OTM)
   - Why it fits the regime
   - Expected payoff scenario
   - Estimated cost

### Usage Examples

```bash
# Basic usage
lox labs hedge-ideas

# With custom budget
lox labs hedge-ideas --budget 1000

# Show more ideas
lox labs hedge-ideas --max 5 --budget 2000
```

## Recommended Changes to Autopilot

To make `lox autopilot run-once` actually usable, we should:

### Option 1: Make it non-interactive by default
```bash
# Remove the "Close BTCUSD now? [y/N]" prompts
# Just show a summary:
lox autopilot run-once --engine ml --basket extended
```

**Output:**
```
Current Positions (7):
  ✓ GLDM      +1.8%   (HOLD - bullish news)
  ✓ NVDA put  +5.5%   (HOLD - hedge working)
  ⚠ TAN put   -37.5%  (TRIM - deep OTM, low recovery)
  ⚠ BTC       -3.1%   (REVIEW - bearish sentiment)

Recommended Trades (2):
  1. BUY QID shares (bearish macro)
  2. TRIM TAN260618P00032000 (cut losses)

Execute these? [y/N]
```

### Option 2: Separate position review from new trade generation
```bash
# Review existing positions
lox autopilot review

# Get new trade ideas
lox autopilot ideas --budget 500

# Execute trades
lox autopilot execute
```

### Option 3: Use hedge-ideas for new trade generation
```bash
# Review + suggest hedges
lox autopilot review --hedge-ideas
```

---

## Implementation Status

- ✅ `lox labs hedge-ideas` is implemented and working
- ⏳ Need to simplify `lox autopilot run-once` output
- ⏳ Need to add `--quiet` or `--summary` flag to autopilot
- ⏳ Need to remove interactive prompts (or add `--no-interactive` flag)

## Next Steps

1. Test `lox labs hedge-ideas` in production
2. Decide on autopilot simplification approach (Option 1, 2, or 3)
3. Implement the changes
4. Update README with new workflow
