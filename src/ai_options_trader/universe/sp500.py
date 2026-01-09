from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Any


@dataclass(frozen=True)
class Sp500Universe:
    tickers: list[str]
    skipped: list[str]
    source: str


DEFAULT_SP500_CSV_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
DEFAULT_SP500_FMP_URL = "https://financialmodelingprep.com/api/v3/sp500_constituent"


def _parse_symbols_from_fmp(js: Any) -> list[str]:
    # Expect list[{"symbol": "...", ...}]
    out: list[str] = []
    if not isinstance(js, list):
        return out
    for row in js:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or row.get("Symbol") or "").strip().upper()
        if not sym:
            continue
        out.append(sym)
    # de-dupe
    seen = set()
    uniq: list[str] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


def load_sp500_universe(
    *,
    refresh: bool = False,
    cache_path: str | Path = "data/cache/universe/sp500_constituents.csv",
    url: str = DEFAULT_SP500_CSV_URL,
    fmp_api_key: str | None = None,
    fmp_url: str = DEFAULT_SP500_FMP_URL,
    skip_dotted: bool = True,
) -> Sp500Universe:
    """
    Load S&P 500 constituents from a simple CSV source (cached locally).

    Why this exists:
    - We want a deterministic "SP500" universe without maintaining a hard-coded 500-ticker list in-repo.
    - Parsing Wikipedia HTML would add fragile dependencies (lxml/bs4); a CSV source is simpler.

    Notes:
    - Some tickers contain '.' (e.g., BRK.B, BF.B). OCC option symbols often use different roots for these,
      and many APIs disagree on the canonical representation. By default we skip dotted tickers.
    """
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if refresh or not p.exists():
        import requests

        # 1) Try CSV URL
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            p.write_bytes(r.content)
        except Exception:
            # 2) Fallback: FMP constituents endpoint
            if fmp_api_key:
                r2 = requests.get(fmp_url, params={"apikey": fmp_api_key}, timeout=30)
                r2.raise_for_status()
                js = r2.json()
                syms = _parse_symbols_from_fmp(js)
                if not syms:
                    raise RuntimeError("FMP returned no symbols for S&P 500 constituents.")
                # Write a minimal CSV cache compatible with DictReader below.
                with p.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["Symbol"])
                    for s in syms:
                        w.writerow([s])
            else:
                raise RuntimeError(
                    "Failed to download S&P 500 constituents CSV, and no FMP_API_KEY was provided for fallback. "
                    "Set `FMP_API_KEY=...` in your `.env`, or pass a working `url=` to `load_sp500_universe`."
                )

    rows: list[dict[str, str]] = []
    with p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})

    tickers: list[str] = []
    skipped: list[str] = []
    for row in rows:
        sym = (row.get("Symbol") or "").strip().upper()
        if not sym:
            continue
        if skip_dotted and "." in sym:
            skipped.append(sym)
            continue
        tickers.append(sym)

    # De-dupe while preserving order
    seen = set()
    uniq = []
    for t in tickers:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)

    return Sp500Universe(tickers=uniq, skipped=skipped, source=str(p))


