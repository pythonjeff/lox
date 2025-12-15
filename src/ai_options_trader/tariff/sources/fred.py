from __future__ import annotations

import pandas as pd

from ai_options_trader.data.fred import FredClient
from ai_options_trader.tariff.sources.base import TariffDataProvider


class FredCostProxyProvider(TariffDataProvider):
    """Provides cost proxies from FRED.

    Note: equity prices and earnings fragility are intentionally not sourced from FRED.
    """

    def __init__(self, fred_api_key: str, cache_dir: str = "data/cache/fred"):
        self.fred = FredClient(api_key=fred_api_key, cache_dir=cache_dir)

    def get_cost_proxies(self, start_date: str) -> pd.DataFrame:
        # Placeholder: replace series IDs once the dataset is finalized.
        series_ids = {
            "placeholder_cost_proxy": "CPIAUCSL",
        }

        frames: list[pd.DataFrame] = []
        for col, sid in series_ids.items():
            df = self.fred.fetch_series(sid, start_date=start_date, refresh=False)
            df = df.rename(columns={"value": col}).set_index("date")
            frames.append(df[[col]])

        out = pd.concat(frames, axis=1).sort_index()
        # Align to daily to match equities by forward-fill
        return out.resample("D").ffill()

    def get_equity_prices(self, symbols, start_date: str) -> pd.DataFrame:
        raise NotImplementedError("FredCostProxyProvider does not provide equity prices.")

    def get_earnings_fragility(self, symbols, start_date: str) -> pd.DataFrame:
        idx = pd.date_range(start=start_date, end=pd.Timestamp.utcnow().date(), freq="D")
        return pd.DataFrame(index=idx)


