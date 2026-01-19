# Hedge Fund Portfolio Balancing

## The Correct Approach

`lox labs hedge` now works like a **real hedge fund risk manager**:

### ✅ Portfolio-Based (not regime-based)
- Analyzes YOUR ACTUAL POSITIONS
- Finds INVERSE trades to balance your book
- If you have puts → suggests calls
- If you have calls → suggests puts
- If you're net short → adds long exposure
- If you're net long → adds protection

---

## Example: Your Current Portfolio

```
Your Portfolio:
  Long equity      2 positions  (BTCUSD, GLDM)
  Short equity     0 positions
  Put hedges       3 contracts  (HYG, NVDA, TAN)
  Call hedges      1 contracts  (TLT)
  Vol exposure     1 positions  (VIXM)

Net Exposure:
  Long exposure    3 positions (2 equity + 1 call)
  Short exposure   3 positions (3 puts)
  Vol exposure     1 positions
  Net bias         BALANCED ✓
```

### Offsetting Recommendations

Since you're **BALANCED** (3 long vs 3 short), the hedge command suggests:

```
1. Portfolio is balanced ✓
   Why: Equal long/short exposure → no major offsetting needed
   
2. Add small VIX hedge (5-10% of portfolio)
   Why: INSURANCE: Vol is very cheap (VIX 14.9)
   Payoff: Vol spike → insurance pays off
```

---

## Example: Over-Hedged Portfolio

```
Your Portfolio:
  Long equity      1 positions  (SPY)
  Put hedges       5 contracts  (SPY, QQQ, IWM, HYG, TLT)

Net Exposure:
  Long exposure    1 positions
  Short exposure   5 positions
  Net bias         NET SHORT (over-hedged) ⚠️
```

### Offsetting Recommendations

Since you're **NET SHORT** (1 long vs 5 short), the hedge command suggests:

```
1. SPY 3M calls (ATM or slight OTM)
   OFFSET: You have SPY puts → add calls to balance
   Payoff: SPY rallies → calls offset put losses
   Cost: ~$175
   
2. SPY shares or SPY 3M calls (ATM)
   OFFSET: Portfolio is net short → add long exposure
   Payoff: Market rallies → offsets put losses
   Cost: ~$150
   
3. Trim QQQ puts by 30-50%
   REDUCE: Over-hedged → reduce short exposure
   Payoff: Preserve capital, reduce theta drag
   Cost: $0 (reduces exposure)
```

---

## Example: Under-Hedged Portfolio

```
Your Portfolio:
  Long equity      5 positions  (AAPL, MSFT, NVDA, GOOGL, AMZN)
  Put hedges       0 contracts

Net Exposure:
  Long exposure    5 positions
  Short exposure   0 positions
  Net bias         NET LONG (under-hedged) ⚠️
```

### Offsetting Recommendations

Since you're **NET LONG** (5 long vs 0 short), the hedge command suggests:

```
1. AAPL 3M puts (10% OTM)
   OFFSET: You're long AAPL → add puts to protect
   Payoff: AAPL drops → puts offset equity losses
   Cost: ~$150
   
2. NVDA put spreads (buy ATM, sell 10% OTM)
   OFFSET: You're long NVDA → add put spreads for balance
   Payoff: NVDA drops → spreads offset losses
   Cost: ~$125
   
3. SPY 3M puts (10% OTM)
   OFFSET: Portfolio is net long → add broad protection
   Payoff: Market correction → puts offset equity losses
   Cost: ~$150
```

---

## Key Principles (Hedge Fund Style)

### 1. **Net Exposure Management**
- Calculate: `Net Long = (equity longs + calls) - puts`
- If net long > +1 → add protection (puts)
- If net short < -1 → add exposure (calls/shares)
- If balanced → tactical adjustments only

### 2. **Specific Offsetting**
- Don't just add "SPY puts" generically
- If you have NVDA puts → suggest NVDA calls
- If you have TLT calls → suggest TLT put spreads
- Match the underlying for precise balancing

### 3. **Cost-Conscious**
- If over-hedged → TRIM positions (costs $0)
- If balanced → small tactical hedges only (5-10%)
- If under-hedged → add protection with budget

### 4. **Theta Awareness**
- Over-hedged portfolio → bleeding theta on puts
- Suggestion: reduce put size or offset with long delta
- Under-hedged portfolio → unprotected tail risk
- Suggestion: add cheap protection (VIX <15)

---

## Comparison

### ❌ Old Approach (regime-based)
```
You have: 3 SPY puts
Regime: STAGFLATION
Suggestion: Add more SPY puts + HYG puts + VIX calls
```
**Problem:** Doubling down on same direction!

### ✅ New Approach (portfolio-based)
```
You have: 3 SPY puts
Net exposure: NET SHORT (over-hedged)
Suggestion: Add SPY calls OR trim puts to balance
```
**Correct:** Offsetting your existing positions!

---

## When to Use Each Command

| Command | Purpose | When to Use |
|---------|---------|------------|
| `lox labs hedge` | Balance portfolio | Daily risk check, after new trades |
| `lox labs grow` | Add new exposure | Looking for opportunities, regime change |
| `lox labs mc-v01 --real` | Stress test | Before major events, regime uncertainty |

---

## Trade Execution

After running `lox labs hedge`, you'll see specific recommendations like:

```
1. NVDA 3M calls (ATM or slight OTM)
   OFFSET: You have NVDA puts → add calls to balance
```

**To execute:**
```bash
lox options recommend --ticker NVDA --direction bullish
```

Or manually enter the trade in your broker.

---

## Summary

**`lox labs hedge` = Find the OPPOSITE of what you have**

- Not about regime
- Not about market thesis
- Pure risk management
- Balance your book
- Reduce net exposure
- Hedge fund best practice

This is how professional risk managers think about portfolio construction.
