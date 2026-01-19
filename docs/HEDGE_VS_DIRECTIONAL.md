# Understanding `lox labs hedge` vs `lox labs grow`

## The Key Difference

### `lox labs hedge` - INVERSE to your portfolio
**Purpose:** Protect your existing long equity exposure

These are **true hedges** - they go UP when your portfolio goes DOWN:
- ✅ **SPY puts** → profits when SPY falls (protects long equity)
- ✅ **VIX calls** → profits when volatility spikes (typically with equity selloffs)
- ✅ **HYG puts** → profits when credit spreads widen (risk-off)
- ✅ **TLT puts** (in inflation regime) → profits when bonds fall

**Label in output:** `HEDGE: ...`

---

### `lox labs grow` - SAME direction or new exposure
**Purpose:** Add new positions that benefit from current regime

These are **directional plays** - they go UP when the thesis plays out:
- ✅ **QQQ calls** → profits when tech rallies (same direction as growth)
- ✅ **GLD shares** → profits when gold rises (inflation hedge, but NOT inverse to equity)
- ✅ **TLT calls** (in disinflation regime) → profits when bonds rally

**Label in output:** `SAME DIRECTION: ...` or just the thesis

---

## Examples by Regime

### Stagflation (weak growth + high inflation)

**`lox labs hedge`** - Protect your long equity:
```
1. SPY 3M puts (10% OTM)
   HEDGE: Protects your long equity from stagflation selloff
   
2. HYG 3M puts (5% OTM)
   HEDGE: Credit spreads widen in stagflation
   
3. VIX 1M calls (25 strike)
   HEDGE: Vol is cheap, stagflation → vol spikes
```

**`lox labs grow`** - Add new exposure:
```
1. Sector rotation → commodities
   SAME DIRECTION: Inflation regime favors hard assets
   Look at: GLD, GLDM, DBC, USO
   
2. Defensive sectors
   SAME DIRECTION: Weak growth → utilities, staples hold up
   Look at: XLU, XLP, VDC
```

---

### Goldilocks (low inflation + strong growth)

**`lox labs hedge`** - Light hedging:
```
1. SPY 6M puts (15% OTM)
   HEDGE: VIX <15 = cheap tail protection (extend duration)
   
2. Sell near-term premium
   INCOME: Low vol = high theta, collect premium
```

**`lox labs grow`** - Aggressive growth:
```
1. Tech / growth equities
   SAME DIRECTION: Low inflation + strong growth = multiple expansion
   Look at: QQQ, XLK, ARKK, SMH
   
2. Long-dated calls on quality names
   SAME DIRECTION: Vol is cheap, extend duration for leverage
   SPY, QQQ 6M calls (ATM or slightly OTM)
```

---

## Why the Confusion?

### Common mistake: "Gold is a hedge"
- ❌ **NOT a hedge for long equity** - gold can go down when stocks go down
- ✅ **IS a hedge for inflation** - gold goes up when CPI rises
- 💡 **For `lox labs hedge`:** If you're long equity, gold is NOT a hedge (use SPY puts)
- 💡 **For `lox labs grow`:** If regime is inflationary, gold is a SAME DIRECTION play

### Another example: "TLT is a hedge"
- ❌ **NOT always a hedge for equity** - depends on WHY stocks are falling
- ✅ **IS a hedge for growth scare** - if stocks fall due to recession, TLT rallies
- ❌ **NOT a hedge for inflation scare** - if stocks fall due to inflation, TLT also falls
- 💡 In stagflation: TLT is NOT a hedge (both equity and bonds fall)
- 💡 In disinflation: TLT calls could be in `lox labs grow` (rates down → bonds up)

---

## Quick Reference

| Your Portfolio | Hedge (inverse) | Same Direction (grow) |
|----------------|-----------------|----------------------|
| Long equity (SPY, QQQ, stocks) | SPY puts, VIX calls | More equity (leverage), call spreads |
| Long bonds (TLT) | TLT puts, rising rate bets | More duration (EDV), TLT calls |
| Long gold (GLD) | Gold puts, deflation plays | More gold, mining stocks |
| Short volatility (selling premium) | VIX calls, put spreads | Sell more premium, iron condors |

---

## Updated Output Labels

### Before (confusing):
```
Trade: SPY 3M puts (10% OTM)
Why: Balanced tail protection
```
**Problem:** Doesn't say it's a HEDGE for your existing positions

### After (clear):
```
Trade: SPY 3M puts (10% OTM)
Why: HEDGE: Protects your long equity from stagflation selloff
```
**Better:** Explicitly says "HEDGE" and explains what it protects

---

## Summary

- **`lox labs hedge`** = protection AGAINST your existing exposure (inverse correlation)
- **`lox labs grow`** = new opportunities IN LINE WITH current regime (directional)

All ideas from `lox labs hedge` should be labeled:
- `HEDGE: ...` - true inverse protection
- `INCOME: ...` - premium collection strategies
- `TRIM: ...` - reduce existing exposure

All ideas from `lox labs grow` should be about adding exposure, not protecting it.
