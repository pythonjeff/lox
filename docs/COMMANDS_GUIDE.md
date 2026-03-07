## Lox Fund CLI — Command Guide

### 0) Quick mental model
- **`lox pm`** → **Regime drill-down** → **Research** → **NAV tracking**

### 1) Daily commands (start here)
- **`lox pm`**: PM Morning Report — macro regime heatmap, active scenarios, portfolio Greeks, LLM CIO brief. LLM is on by default.
- **`lox pm --no-llm`**: Data sections only (fast).
- **`lox pm --json`**: Machine-readable output.
- **`lox status`**: Quick portfolio health (NAV, P&L, cash).
- **`lox risk`**: Greeks dashboard with theta breakeven analysis.

### 2) Regime drill-down
- **`lox regime vol`**: Volatility regime (VIX, term structure).
- **`lox regime credit`**: Credit regime (HY OAS, BBB/AAA spreads).
- **`lox regime rates`**: Rates/yield curve analysis.
- **`lox regime funding`**: Funding/liquidity (SOFR, repo, reserves).
- **`lox regime growth`**: Growth regime (payrolls, ISM, claims).
- **`lox regime inflation`**: Inflation regime (CPI, Core PCE, breakevens).
- **`lox regime consumer`**: Consumer regime (sentiment, spending, mortgage rates).
- **`lox regime fiscal`**: Fiscal regime (deficits, TGA, issuance).
- **`lox regime earnings`**: Earnings regime (S&P 500 beat rate, revisions).
- **`lox regime oil`**: Oil/commodities regime.

Add `--llm` to any regime command for LLM analysis. Add `--book` for position impact.

### 3) Unified regime views
- **`lox research regimes`**: All pillars with scores, trend arrows, 7d deltas.
- **`lox research regimes --trend`**: Full trend dashboard (sparklines, momentum z, velocity).
- **`lox research regimes --detail credit`**: Drill into a specific pillar.
- **`lox research regimes --scenarios`**: Active macro scenarios.
- **`lox research regimes --llm`**: AI commentary on regime state.
- **`lox research regimes --book`**: Position exposure vs regime state.

### 4) Research
- **`lox research ticker NVDA`**: Full hedge-fund-style research (momentum, HF metrics, fundamentals, SEC filings).
- **`lox research ticker NVDA --llm`**: With LLM synthesis.
- **`lox research portfolio`**: LLM outlook on all open positions.
- **`lox research scenario SPY`**: Monte Carlo macro shock simulation.

### 5) Options scanning
- **`lox scan -t NVDA`**: Options chain (default: puts, 30-365 DTE).
- **`lox scan -t CRWV --want put --min-days 100`**: Custom DTE range.

### 6) Interactive chat
- **`lox chat`**: Interactive research chat with portfolio context.
- **`lox chat -t AAPL`**: Ticker-focused chat.
- **`lox chat -c fiscal`**: Chat with fiscal regime data.
- **`lox chat -c funding`**: Chat with funding/liquidity context.
- **`lox chat -c regimes`**: Chat with all regime classifications.

### 7) Fund accounting
- **`lox account`**: Alpaca account details.
- **`lox account summary`**: LLM summary with risk watch + news + calendar.
- **`lox nav snapshot`**: Write NAV snapshot and compute returns.
- **`lox nav investor report`**: Investor ownership/basis/value/P&L via unitized NAV.
- **`lox nav investor contribute JL 50 --note "Feb add"`**: Log contribution to both ledgers.
- **`lox weekly report`**: Weekly summary (NAV, trades, thesis, macro).
- **`lox weekly report --share`**: Investor-facing version.
- **`lox closed-trades`**: Realized P&L from closed positions.

### 8) Crypto perps
- **`lox crypto data`**: BTC, ETH, SOL overview + technicals.
- **`lox crypto research`**: Data + macro regime LLM analysis.
- **`lox crypto analyze`**: Perps-specific trading analysis.
- **`lox crypto balance`**: Account balance & equity.
- **`lox crypto positions`**: Open positions with PnL.
- **`lox crypto trade BTC BUY 0.001 --leverage 5`**: Place a trade.
- **`lox crypto close BTC`**: Close a position.

### Notes
- Run **`lox <command> --help`** for flags.
- Paper vs live is controlled by **`ALPACA_PAPER`** and **`--live`**.
- All `--llm` flags require `OPENAI_API_KEY` in `.env`.
- `lox pm` has LLM on by default; use `--no-llm` for data only.
