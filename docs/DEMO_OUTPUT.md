# Demo Output: Trade Ideas Commands

## Scenario: Stagflation Regime

Current macro environment:
- CPI YoY: 3.2% (elevated)
- Payrolls 3M annualized: -0.17% (weak)
- VIX: 14.2 (cheap hedges)
- ON RRP: ~$50B (buffer depleted)

---

### `lox labs hedge` Output

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

#  Trade                      Why                                          Payoff If                              Cost
1  SPY 3M puts (10% OTM)      Tail protection in weak growth + high        SPY drops 15%+ → hedge pays 5-10x     ~$150 (30% of budget)
                               inflation                                                                          
2  HYG 3M puts (5% OTM)       Credit spreads widen in stagflation          Credit stress → hedge pays 3-5x       ~$100 (20% of budget)
3  VIX 1M calls (25 strike)   Vol is cheap, stagflation → vol spikes      VIX >30 → hedge pays 10x+             ~$75 (15% of budget)

To execute:
  • Run: lox options recommend --ticker SPY --direction bearish
  • Or manually enter trades in your broker
```

---

### `lox labs grow` Output

```
═══ Lox Growth Opportunities ═══

Current Regime:
  Macro        STAGFLATION
  Liquidity    FRAGILE (RRP depleted)
  Volatility   LOW (VIX 14.2 - calm markets)

═══ Recommended Opportunities ═══

#  Trade                        Why                                          Target                               Allocation
1  Sector rotation →            Inflation regime favors hard assets          Look at: GLD, GLDM, DBC, USO         ~$150 (30%)
   commodities                                                                                                    
2  Defensive sectors            Weak growth → utilities, consumer staples    Look at: XLU, XLP, VDC               ~$125 (25%)
                                hold up                                                                           
3  Avoid growth / stay small    High inflation + weak growth = bad for       No new tech/growth exposure          ~$0 (trim existing)
                                growth                                                                            

To execute:
  • Run: lox options recommend --ticker GLD --direction bullish
  • Or manually enter trades in your broker
```

---

## Scenario: Goldilocks Regime

Current macro environment:
- CPI YoY: 2.1% (on target)
- Payrolls 3M annualized: +2.5% (strong)
- VIX: 13.8 (very cheap)
- ON RRP: ~$180B (ample buffer)

---

### `lox labs hedge` Output

```
═══ Lox Hedge Recommendations ═══

Current Regime:
  Macro        GOLDILOCKS
  Liquidity    ORDERLY
  Volatility   LOW (VIX 13.8 - hedges cheap)

Your Portfolio:
  Long equity  5 positions
  Put hedges   3 contracts
  Vol exposure 2 positions

Portfolio Gaps:
  None - portfolio looks well-hedged for this regime

═══ Recommended Hedges ═══

#  Trade                         Why                                          Payoff If                             Cost
1  SPY 6M puts (15% OTM)         VIX <15 = cheap insurance, extend duration   Black swan → hedge pays 20x+         ~$125 (25% of budget)
2  Sell near-term premium        Low vol = high theta, collect premium        Market stable → pocket premium        ~$0 (credit received)
3  Reduce hedge size 30%         Calm markets = theta drag hurts              Preserve capital for next vol spike   ~$0 (trim existing)

To execute:
  • Run: lox options recommend --ticker SPY --direction bearish
  • Or manually enter trades in your broker
```

---

### `lox labs grow` Output

```
═══ Lox Growth Opportunities ═══

Current Regime:
  Macro        GOLDILOCKS
  Liquidity    ORDERLY
  Volatility   LOW (VIX 13.8 - calm markets)

═══ Recommended Opportunities ═══

#  Trade                           Why                                          Target                                Allocation
1  Tech / growth equities          Low inflation + strong growth = multiple     Look at: QQQ, XLK, ARKK, SMH          ~$200 (40%)
                                    expansion                                    (semiconductors)                      
2  Sell short-dated premium        Low vol = high theta, collect premium        Sell covered calls on existing        ~$100 (20%)
                                                                                 longs                                 
3  Long-dated calls on quality     Vol is cheap, extend duration for            SPY, QQQ 6M calls (ATM or slightly    ~$125 (25%)
   names                            leverage                                     OTM)                                  

To execute:
  • Run: lox options recommend --ticker QQQ --direction bullish
  • Or manually enter trades in your broker
```

---

## Key Takeaways

### ✅ Simple and Fast
- **No verbose LLM reasoning** per position
- **No interactive prompts** ("Close X now?")
- **Output in 2 seconds** vs 5 minutes for old autopilot

### ✅ Regime-Aware
- **Stagflation**: Defensive positioning, commodities, credit hedges
- **Goldilocks**: Aggressive growth, sell premium, extend duration

### ✅ Flexible Guidance
- **Specific**: "SPY 3M puts (10% OTM)"
- **Directional**: "Look at: GLD, GLDM, DBC, USO"
- **Strategic**: "Avoid growth / stay small"

### ✅ Budget-Conscious
- Shows allocation % for each idea
- Respects `--budget` flag
- Suggests "trim existing" when overweight
