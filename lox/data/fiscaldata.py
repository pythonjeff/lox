from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class FiscalDataEndpoint:
    """
    FiscalData uses a base URL + endpoint path, e.g.
    base: https://api.fiscaldata.treasury.gov/services/api/fiscal_service
    endpoint: /v1/debt/treasurydirect/auction
    """

    path: str


class FiscalDataClient:
    def __init__(
        self,
        base_url: str = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service",
        cache_dir: str = "data/cache/fiscaldata",
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, cache_key: str) -> Path:
        safe = cache_key.replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_")
        return self.cache_dir / f"{safe}.csv"

    def fetch(
        self,
        *,
        endpoint: FiscalDataEndpoint,
        params: Optional[Dict[str, Any]] = None,
        cache_key: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch an endpoint and return a DataFrame of records.

        Notes:
        - FiscalData responses are typically JSON with a "data" list.
        - We cache the returned table as CSV for reproducibility and speed.
        """
        cache_path = self._cache_path(cache_key)
        if cache_path.exists() and not refresh:
            return pd.read_csv(cache_path)

        # Lazy import so unit tests / restricted environments can import this module
        # without triggering SSL/cert initialization.
        import requests
        from requests.exceptions import RequestException

        url = f"{self.base_url}{endpoint.path}"
        params = dict(params or {})
        # Large page size helps avoid pagination for common use cases.
        params.setdefault("page[size]", 10000)

        last_err: Exception | None = None
        for _attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=30)
                r.raise_for_status()
                js = r.json()
                break
            except RequestException as e:
                last_err = e
        else:
            if cache_path.exists():
                return pd.read_csv(cache_path)
            raise last_err  # type: ignore[misc]

        data = js.get("data", [])
        df = pd.DataFrame(data)
        df.to_csv(cache_path, index=False)
        return df


