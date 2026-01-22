# Lox Capital — Technical Specification

**Version 1.0 | January 2026**

This document provides operational and architectural documentation for technical reviewers and due diligence processes.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Pipeline & Lineage](#2-data-pipeline--lineage)
3. [API Integration Layer](#3-api-integration-layer)
4. [Error Handling & Resilience](#4-error-handling--resilience)
5. [Logging & Monitoring](#5-logging--monitoring)
6. [Reproducible Environment](#6-reproducible-environment)
7. [Deployment Architecture](#7-deployment-architecture)
8. [Security Considerations](#8-security-considerations)

---

## 1. System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                │
├────────────────────────────┬─────────────────────────────────────────┤
│    Web Dashboard           │         CLI Interface                    │
│    (Flask + JS)            │         (Typer + Rich)                   │
│    Port 5001               │         `lox` command                    │
└────────────────────────────┴─────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                             │
├───────────────┬───────────────┬───────────────┬──────────────────────┤
│  Regime       │  Portfolio    │  Monte Carlo  │  LLM Analyst         │
│  Classifiers  │  Management   │  Simulation   │  (GPT-4o-mini)       │
└───────────────┴───────────────┴───────────────┴──────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                  │
├───────────────┬───────────────┬───────────────┬──────────────────────┤
│  FRED         │  FMP          │  Alpaca       │  Trading Economics   │
│  (Macro)      │  (Market)     │  (Portfolio)  │  (Calendar)          │
└───────────────┴───────────────┴───────────────┴──────────────────────┘
```

### Module Structure

```
src/ai_options_trader/
├── cli.py                    # Entry point, command registration
├── config.py                 # Settings and environment loading
├── cli_commands/             # CLI command implementations
│   ├── dashboard_cmd.py      # Unified dashboard
│   ├── monte_carlo_v01_cmd.py# Monte Carlo simulation
│   ├── stress_cmd.py         # Stress testing
│   └── ...                   # ~50 command modules
├── portfolio/                # Portfolio management
│   ├── positions.py          # Position and Portfolio classes
│   ├── greeks.py             # Greek calculations
│   ├── stress_test.py        # Stress test framework
│   └── alpaca_adapter.py     # Alpaca → Portfolio conversion
├── llm/                      # LLM and simulation
│   ├── monte_carlo.py        # MC engine (v0)
│   ├── monte_carlo_v01.py    # MC engine (v0.1, position-level)
│   └── analyst.py            # LLM prompt construction
├── regimes/                  # Regime classification
│   └── core.py               # Base regime framework
├── macro/                    # Macro pillar
│   ├── models.py             # Data models
│   └── regime.py             # Macro regime classifier
├── volatility/               # Vol pillar
├── rates/                    # Rates pillar
├── funding/                  # Funding pillar
├── fiscal/                   # Fiscal pillar
└── data/                     # Data fetching
    ├── fred.py               # FRED API client
    ├── alpaca.py             # Alpaca client factory
    └── market.py             # Market data utilities
```

### Dashboard Architecture

```
dashboard/
├── app.py                    # Flask application (1830 lines)
│   ├── PALMER_CACHE          # Server-side analysis cache
│   ├── get_positions_data()  # Position fetching + conservative marking
│   ├── _generate_palmer_analysis() # LLM + regime analysis
│   └── Background thread     # 30-min auto-refresh
├── templates/
│   └── dashboard.html        # Single-page dashboard template
├── static/
│   ├── style.css             # Styling (animations, responsive)
│   └── dashboard.js          # Client-side rendering + polling
└── requirements.txt          # Flask dependencies
```

---

## 2. Data Pipeline & Lineage

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL APIS                                 │
├─────────┬─────────┬─────────┬─────────┬─────────────────────────────┤
│  FRED   │   FMP   │ Alpaca  │   TE    │   OpenAI                    │
│         │         │         │         │                             │
│ • Macro │ • VIX   │ • Pos   │ • Cal   │ • Insights                  │
│ • Rates │ • 10Y   │ • Orders│ • News  │ • Theories                  │
│ • Credit│ • News  │ • Quotes│         │                             │
└────┬────┴────┬────┴────┬────┴────┬────┴─────────────────────────────┘
     │         │         │         │
     ▼         ▼         ▼         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CACHING LAYER                                   │
│                                                                      │
│  data/cache/                                                         │
│  ├── fred/           # Macro series (CSV, daily refresh)             │
│  ├── fmp_prices/     # Price history (CSV)                           │
│  ├── correlations/   # Trained correlation matrices (NPY)            │
│  ├── playbook/       # Regime feature matrix (CSV)                   │
│  └── econ_calendar/  # Economic calendar (transient)                 │
│                                                                      │
│  Server-side cache (in-memory):                                      │
│  └── PALMER_CACHE    # 30-min TTL                                    │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PROCESSING LAYER                                 │
│                                                                      │
│  1. Normalization: Z-scores against 2-year rolling window            │
│  2. Classification: Regime rules applied                             │
│  3. Greeks: Black-Scholes calculation                                │
│  4. Attribution: Position-level P&L decomposition                    │
│  5. Simulation: Monte Carlo scenario generation                      │
│  6. LLM: Prompt construction and response parsing                    │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                                   │
│                                                                      │
│  • Dashboard HTML/JSON                                               │
│  • CLI Rich console output                                           │
│  • JSON API responses                                                │
│  • SQLite tracking (data/tracker.sqlite3)                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Freshness Requirements

| Data Source | Staleness Threshold | Fallback Behavior |
|-------------|---------------------|-------------------|
| FRED (macro) | 2 days (T+1 typical) | Use cached value, flag stale |
| FMP quotes | 30 minutes | Use last known, show warning |
| Alpaca positions | 5 minutes | Real-time, no cache |
| Trading Economics | 1 hour | Fall back to FMP calendar |
| LLM analysis | 30 minutes | Use cached Palmer output |

### Cache Strategy

**File-based cache (`data/cache/`):**
- CSV format for tabular data (human-readable, version-controllable)
- NPY format for numpy arrays (correlation matrices)
- Keyed by date or parameter hash

**Server-side cache (`PALMER_CACHE`):**
```python
PALMER_CACHE = {
    "analysis": str,           # LLM insight text
    "regime_snapshot": dict,   # Raw indicator values
    "traffic_lights": dict,    # Processed statuses
    "events": dict,            # Economic calendar
    "headlines": list,         # News headlines
    "monte_carlo": dict,       # MC forecast
    "timestamp": str,          # ISO timestamp
    "last_refresh": datetime,  # When cache was populated
}
PALMER_REFRESH_INTERVAL = 30 * 60  # 30 minutes
```

---

## 3. API Integration Layer

### FRED API

**Client:** `src/ai_options_trader/data/fred.py`

```python
class FredClient:
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    def fetch_series(self, series_id: str, start_date: str, refresh: bool):
        """Fetch FRED series with optional cache refresh."""
        cache_path = Path(f"data/cache/fred/{series_id}.csv")
        
        if not refresh and cache_path.exists():
            return pd.read_csv(cache_path)
        
        response = requests.get(self.BASE_URL, params={
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
        })
        # ... parse and cache
```

**Key Series:**
| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| BAMLH0A0HYM2 | HY OAS | Daily |
| DGS10 | 10Y Treasury | Daily |
| T10YIE | 10Y Breakeven | Daily |
| CPIAUCSL | CPI (all urban) | Monthly |
| PAYEMS | Nonfarm payrolls | Monthly |

### FMP API

**Client:** Direct `requests` calls in various modules

**Endpoints Used:**
| Endpoint | Purpose | Rate Limit |
|----------|---------|------------|
| `/api/v3/quote/{symbol}` | Real-time quotes | 300/min |
| `/api/v3/stock_news` | News by ticker | 300/min |
| `/api/v3/economic_calendar` | Economic events | 300/min |
| `/api/v3/historical-price-full/{symbol}` | Price history | 300/min |

### Alpaca API

**Client:** `src/ai_options_trader/data/alpaca.py`

```python
def make_clients(settings: Settings):
    """Create Alpaca trading and data clients."""
    
    is_paper = settings.ALPACA_PAPER.lower() == "true"
    base_url = "https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets"
    
    trading_client = TradingClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_API_SECRET,
        paper=is_paper,
    )
    
    data_client = StockHistoricalDataClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_API_SECRET,
    )
    
    return trading_client, data_client
```

**Key Endpoints:**
- `GET /v2/account` — Account info
- `GET /v2/positions` — Open positions
- `GET /v2/orders` — Order history
- `GET /v2/stocks/{symbol}/quotes/latest` — Latest quote
- `GET /v1beta1/news` — News articles

### Trading Economics API

**Client:** `fetch_trading_economics_calendar()` in `dashboard/app.py`

```python
def fetch_trading_economics_calendar(api_key, today_str):
    url = f"https://api.tradingeconomics.com/calendar/country/united%20states/{today_str}/{today_str}"
    headers = {"Authorization": f"Client {api_key}"}
    response = requests.get(url, headers=headers, timeout=10)
```

### OpenAI API

**Client:** OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(api_key=settings.openai_api_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
    max_tokens=200,
)
```

---

## 4. Error Handling & Resilience

### Error Classification

| Error Type | Handling Strategy | User Impact |
|------------|-------------------|-------------|
| **Network timeout** | Retry 3x with exponential backoff | Brief delay |
| **API rate limit** | Exponential backoff, queue requests | Temporary degradation |
| **Invalid API key** | Fail fast, log error | Feature unavailable |
| **Missing data** | Use fallback source or cached value | Stale data flagged |
| **LLM failure** | Return generic message | Degraded insight |
| **Parse error** | Log and skip item | Partial data |

### Implementation Patterns

**Retry with Backoff:**
```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.Timeout, requests.ConnectionError) as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
        return wrapper
    return decorator
```

**Graceful Degradation:**
```python
def get_hy_oas():
    """Get HY OAS with fallback."""
    try:
        # Primary: FRED API
        return fetch_from_fred("BAMLH0A0HYM2")
    except Exception as e:
        print(f"FRED error: {e}")
        
        # Fallback: cached value
        cache_path = Path("data/cache/fred/BAMLH0A0HYM2.csv")
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            return df.iloc[-1]["value"]
        
        return None  # Feature unavailable
```

### Error Logging

```python
import logging
import traceback

logger = logging.getLogger("lox")

def log_error(context: str, error: Exception, critical: bool = False):
    """Structured error logging."""
    error_info = {
        "context": context,
        "error_type": type(error).__name__,
        "error_msg": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    if critical:
        logger.error(f"CRITICAL: {context}", extra=error_info)
    else:
        logger.warning(f"WARNING: {context}", extra=error_info)
```

---

## 5. Logging & Monitoring

### Log Configuration

**Default log level:** WARNING (production), DEBUG (development)

**Log format:**
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### Key Logging Points

| Component | Log Level | Information |
|-----------|-----------|-------------|
| Palmer refresh | INFO | Timestamp, success/failure |
| API calls | DEBUG | Endpoint, response time |
| Regime change | WARNING | Old state, new state, indicators |
| Error recovery | WARNING | Error type, fallback used |
| LLM calls | DEBUG | Prompt length, response length |

### Monitoring Endpoints

**Dashboard Health:**
```
GET /api/positions       # Position data
GET /api/regime-analysis # Palmer cache status
```

**Cache Status:**
```python
# In dashboard, check PALMER_CACHE metadata
{
    "last_refresh": "2026-01-22T10:00:00Z",
    "cache_age_minutes": 15,
    "error": null,  # or error message if failed
}
```

---

## 6. Reproducible Environment

### Python Environment

**Python version:** 3.11+

**Dependency management:** `pyproject.toml` with pip

```toml
[project]
name = "ai-options-trader"
version = "3.0.0"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "requests>=2.31.0",
    "alpaca-py>=0.15.0",
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    # ...
]
```

### Installation Steps

```bash
# Clone repository
git clone <repo-url>
cd ai-options-trader-starter

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install package in editable mode
pip install -e .

# Verify installation
lox --help
```

### Environment Variables

**Required (`.env` file):**
```
ALPACA_API_KEY=<your-key>
ALPACA_API_SECRET=<your-secret>
ALPACA_PAPER=false              # true for paper, false for live
FRED_API_KEY=<your-key>
FMP_API_KEY=<your-key>
OPENAI_API_KEY=<your-key>
```

**Optional:**
```
TRADING_ECONOMICS_API_KEY=<your-key>
PALMER_ADMIN_SECRET=<refresh-secret>
OPENAI_MODEL=gpt-4o-mini       # Default model
```

### Reproducibility for Monte Carlo

```python
# Set random seed for reproducible simulations
import numpy as np
np.random.seed(42)

# Run simulation
results = engine.generate_scenarios(n_scenarios=10000)
```

---

## 7. Deployment Architecture

### Local Development

```bash
# CLI tools
lox status
lox dashboard
lox labs stress

# Dashboard (local)
cd dashboard
python app.py  # Runs on http://localhost:5001
```

### Production (Heroku)

**Procfile:**
```
web: gunicorn app:app --workers 2 --threads 4 --timeout 120
```

**Runtime:** `python-3.11.x`

**Environment:**
- Dyno type: Basic ($7/mo)
- Workers: 2 (Gunicorn)
- Auto-sleep: Disabled for continuous refresh

**Deployment:**
```bash
# Push to Heroku
git push heroku main

# Set environment variables
heroku config:set ALPACA_API_KEY=<key>
heroku config:set OPENAI_API_KEY=<key>
# ... etc
```

### Dashboard Background Thread

The Palmer refresh runs in a background thread (not a separate process):

```python
# In dashboard/app.py
def _palmer_background_refresh():
    """Background thread - refreshes every 30 minutes."""
    while True:
        time.sleep(PALMER_REFRESH_INTERVAL)
        _refresh_palmer_cache()

# Started on app initialization
thread = threading.Thread(target=_palmer_background_refresh, daemon=True)
thread.start()
```

**Note:** This works with Gunicorn's threaded workers but requires attention to GIL considerations for CPU-bound operations.

---

## 8. Security Considerations

### API Key Management

- **Never commit `.env`** to version control (`.gitignore` includes it)
- Use environment variables in production (Heroku config vars)
- Rotate keys periodically
- Use read-only keys where possible (FMP, FRED)

### Alpaca Account Access

- **Live vs Paper:** Controlled by `ALPACA_PAPER` environment variable
- Default to paper in development
- Explicit flag for live account in Monte Carlo (`--live`)

### Dashboard Admin Access

```python
# Force refresh requires admin secret
@app.route('/api/regime-analysis/force-refresh')
def api_regime_analysis_force():
    secret = request.args.get("secret", "")
    if secret != ADMIN_SECRET:
        return jsonify({"error": "Unauthorized"}), 403
    # ...
```

### Data Privacy

- No PII stored
- Portfolio data is user-specific (single-tenant)
- LLM prompts contain only market data, no personal information

### Rate Limiting

- API clients respect rate limits
- Dashboard polling interval: 5 minutes (client-side)
- Palmer refresh: 30 minutes (server-side)

---

## Appendix A: Command Reference

### Top-Level Commands

```bash
lox status                    # Portfolio health
lox dashboard                 # Unified regime dashboard
lox account summary           # Account details

lox options scan              # Options scanner
lox options pick              # Best option selection
lox ideas catalyst            # Event-driven ideas
```

### Labs Commands (Power User)

```bash
lox labs stress               # Stress testing
lox labs mc-v01               # Monte Carlo simulation
lox labs deep TICKER          # Deep dive analysis
lox labs vol --llm            # Volatility regime + LLM
lox labs rates snapshot --llm # Rates analysis + LLM
```

### Full Command List

Run `lox --help` for complete command listing.

---

## Appendix B: Database Schema

### SQLite Tracking (`data/tracker.sqlite3`)

```sql
-- NAV history
CREATE TABLE nav_history (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Trade tracking
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    filled_at TEXT,
    strategy TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Regime history
CREATE TABLE regime_history (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    regime TEXT NOT NULL,
    vix REAL,
    hy_oas REAL,
    yield_10y REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Maintainer:** Lox Capital Research
