from __future__ import annotations

import numpy as np
import pandas as pd
from ai_options_trader.tariff.sources.base import TariffDataProvider


class PlaceholderTariffProvider(TariffDataProvider):
    def get_cost_proxies(self, start_date: str) -> pd.DataFrame:
        idx = pd.date_range(start=start_date, end=pd.Timestamp.utcnow().date(), freq="D")
        # Random walk placeholder (replace with real proxy data)
        x = np.cumsum(np.random.normal(0, 1, len(idx)))
        return pd.DataFrame({"placeholder_cost_proxy": x}, index=idx)

    def get_equity_prices(self, symbols, start_date: str) -> pd.DataFrame:
        idx = pd.date_range(start=start_date, end=pd.Timestamp.utcnow().date(), freq="D")
        data = {s: 100 + np.cumsum(np.random.normal(0, 1, len(idx))) for s in symbols}
        return pd.DataFrame(data, index=idx)

    def get_earnings_fragility(self, symbols, start_date: str) -> pd.DataFrame:
        idx = pd.date_range(start=start_date, end=pd.Timestamp.utcnow().date(), freq="D")
        return pd.DataFrame({s: 0.0 for s in symbols}, index=idx)
