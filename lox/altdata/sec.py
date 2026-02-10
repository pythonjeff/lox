"""
SEC EDGAR Filings Fetcher

Fetches company filings from SEC EDGAR:
- 8-K (material events)
- 10-K (annual reports)
- 10-Q (quarterly reports)
- 4 (insider transactions)
- 13F (institutional holdings)

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from lox.altdata.cache import cache_path, read_cache, write_cache
from lox.config import Settings


SEC_EDGAR_BASE = "https://data.sec.gov"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Required user agent per SEC guidelines
SEC_USER_AGENT = "LoxCapital/1.0 (research@loxcapital.com)"


@dataclass(frozen=True)
class SECFiling:
    """Represents a single SEC filing."""
    ticker: str
    cik: str
    form_type: str  # 8-K, 10-K, 10-Q, 4, etc.
    filed_date: str  # YYYY-MM-DD
    accepted_datetime: str
    accession_number: str
    primary_document: str
    description: str = ""
    filing_url: str = ""
    items: list[str] = field(default_factory=list)  # For 8-K: item numbers


@dataclass
class InsiderTransaction:
    """Represents an insider transaction from Form 4."""
    ticker: str
    insider_name: str
    insider_title: str
    transaction_date: str
    transaction_type: str  # P=Purchase, S=Sale, A=Award, etc.
    shares: float
    price_per_share: float | None
    total_value: float | None
    shares_owned_after: float | None


def _get_cik_for_ticker(ticker: str) -> str | None:
    """
    Get CIK (Central Index Key) for a ticker symbol.
    
    Uses SEC's company_tickers.json mapping file.
    """
    cache_key = "sec_ticker_cik_map"
    p = cache_path(cache_key)
    cached = read_cache(p, max_age=timedelta(days=7))
    
    if isinstance(cached, dict):
        mapping = cached
    else:
        try:
            resp = requests.get(
                SEC_COMPANY_TICKERS_URL,
                headers={"User-Agent": SEC_USER_AGENT},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Build ticker -> CIK mapping
            mapping = {}
            for item in data.values():
                if isinstance(item, dict):
                    t = str(item.get("ticker", "")).upper()
                    cik = str(item.get("cik_str", "")).zfill(10)
                    if t and cik:
                        mapping[t] = cik
            
            write_cache(p, mapping)
        except Exception:
            return None
    
    return mapping.get(ticker.upper())


def fetch_sec_filings(
    *,
    settings: Settings,
    ticker: str,
    form_types: list[str] | None = None,
    limit: int = 20,
    cache_max_age: timedelta = timedelta(hours=6),
) -> list[SECFiling]:
    """
    Fetch recent SEC filings for a ticker.
    
    Args:
        settings: Application settings
        ticker: Stock ticker symbol
        form_types: List of form types to filter (e.g., ["8-K", "10-K", "10-Q"])
                   If None, returns all major forms
        limit: Maximum number of filings to return
        cache_max_age: How long to cache results
    
    Returns:
        List of SECFiling objects, most recent first
    """
    t = ticker.strip().upper()
    if not t:
        return []
    
    # Get CIK
    cik = _get_cik_for_ticker(t)
    if not cik:
        return []
    
    # Default form types
    if form_types is None:
        form_types = ["8-K", "10-K", "10-Q", "4"]
    
    form_str = ",".join(sorted(form_types))
    cache_key = f"sec_filings_{t}_{form_str}_limit{limit}"
    p = cache_path(cache_key)
    cached = read_cache(p, max_age=cache_max_age)
    
    if isinstance(cached, list):
        return [
            SECFiling(
                ticker=f["ticker"],
                cik=f["cik"],
                form_type=f["form_type"],
                filed_date=f["filed_date"],
                accepted_datetime=f["accepted_datetime"],
                accession_number=f["accession_number"],
                primary_document=f["primary_document"],
                description=f.get("description", ""),
                filing_url=f.get("filing_url", ""),
                items=f.get("items", []),
            )
            for f in cached
        ]
    
    # Fetch from SEC EDGAR
    try:
        url = f"{SEC_EDGAR_BASE}/submissions/CIK{cik}.json"
        resp = requests.get(
            url,
            headers={"User-Agent": SEC_USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Parse filings
        filings_data = data.get("filings", {}).get("recent", {})
        if not filings_data:
            return []
        
        forms = filings_data.get("form", [])
        filing_dates = filings_data.get("filingDate", [])
        accessions = filings_data.get("accessionNumber", [])
        primary_docs = filings_data.get("primaryDocument", [])
        descriptions = filings_data.get("primaryDocDescription", [])
        accepted_datetimes = filings_data.get("acceptanceDateTime", [])
        items_list = filings_data.get("items", [])
        
        filings: list[SECFiling] = []
        for i in range(min(len(forms), 100)):  # Cap at 100 to avoid huge responses
            form = str(forms[i] if i < len(forms) else "")
            
            # Filter by form type
            if form_types and form not in form_types:
                continue
            
            accession = str(accessions[i] if i < len(accessions) else "")
            accession_formatted = accession.replace("-", "")
            primary_doc = str(primary_docs[i] if i < len(primary_docs) else "")
            
            filing_url = ""
            if accession and primary_doc:
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_formatted}/{primary_doc}"
            
            items = []
            if i < len(items_list) and items_list[i]:
                items = [it.strip() for it in str(items_list[i]).split(",") if it.strip()]
            
            filings.append(SECFiling(
                ticker=t,
                cik=cik,
                form_type=form,
                filed_date=str(filing_dates[i] if i < len(filing_dates) else ""),
                accepted_datetime=str(accepted_datetimes[i] if i < len(accepted_datetimes) else ""),
                accession_number=accession,
                primary_document=primary_doc,
                description=str(descriptions[i] if i < len(descriptions) else ""),
                filing_url=filing_url,
                items=items,
            ))
            
            if len(filings) >= limit:
                break
        
        # Cache results
        cache_data = [
            {
                "ticker": f.ticker,
                "cik": f.cik,
                "form_type": f.form_type,
                "filed_date": f.filed_date,
                "accepted_datetime": f.accepted_datetime,
                "accession_number": f.accession_number,
                "primary_document": f.primary_document,
                "description": f.description,
                "filing_url": f.filing_url,
                "items": f.items,
            }
            for f in filings
        ]
        write_cache(p, cache_data)
        
        return filings
        
    except Exception:
        return []


def fetch_8k_filings(
    *,
    settings: Settings,
    ticker: str,
    limit: int = 10,
    cache_max_age: timedelta = timedelta(hours=6),
) -> list[SECFiling]:
    """
    Fetch recent 8-K filings (material events) for a ticker.
    
    8-K items of interest:
    - 1.01: Entry into Material Agreement
    - 1.02: Termination of Material Agreement
    - 2.01: Completion of Acquisition/Disposition
    - 2.02: Results of Operations (earnings)
    - 2.05: Costs Associated with Exit/Disposal
    - 2.06: Material Impairments
    - 3.01: Notice of Delisting
    - 4.01: Changes in Certifying Accountant
    - 4.02: Non-Reliance on Financial Statements
    - 5.01: Changes in Control
    - 5.02: Executive Changes (departure/appointment)
    - 5.03: Amendments to Articles/Bylaws
    - 7.01: Regulation FD Disclosure
    - 8.01: Other Events
    """
    return fetch_sec_filings(
        settings=settings,
        ticker=ticker,
        form_types=["8-K", "8-K/A"],  # Include amendments
        limit=limit,
        cache_max_age=cache_max_age,
    )


def fetch_annual_quarterly_reports(
    *,
    settings: Settings,
    ticker: str,
    limit: int = 8,  # ~2 years of 10-Qs + 10-Ks
    cache_max_age: timedelta = timedelta(hours=12),
) -> list[SECFiling]:
    """
    Fetch recent 10-K and 10-Q filings for a ticker.
    """
    return fetch_sec_filings(
        settings=settings,
        ticker=ticker,
        form_types=["10-K", "10-K/A", "10-Q", "10-Q/A"],
        limit=limit,
        cache_max_age=cache_max_age,
    )


def fetch_insider_filings(
    *,
    settings: Settings,
    ticker: str,
    limit: int = 20,
    cache_max_age: timedelta = timedelta(hours=6),
) -> list[SECFiling]:
    """
    Fetch recent Form 4 (insider transaction) filings.
    """
    return fetch_sec_filings(
        settings=settings,
        ticker=ticker,
        form_types=["4", "4/A"],
        limit=limit,
        cache_max_age=cache_max_age,
    )


def categorize_8k_items(items: list[str]) -> dict[str, bool]:
    """
    Categorize 8-K items into meaningful buckets.
    
    Returns dict with flags for each category.
    """
    items_str = " ".join(items)
    
    return {
        "has_earnings": "2.02" in items_str,
        "has_executive_changes": "5.02" in items_str,
        "has_material_agreement": "1.01" in items_str or "1.02" in items_str,
        "has_acquisition": "2.01" in items_str,
        "has_impairment": "2.06" in items_str,
        "has_accounting_change": "4.01" in items_str or "4.02" in items_str,
        "has_control_change": "5.01" in items_str,
        "has_other_disclosure": "7.01" in items_str or "8.01" in items_str,
    }


def summarize_filings(filings: list[SECFiling]) -> dict[str, Any]:
    """
    Create a summary of recent filings for analysis.
    """
    if not filings:
        return {"total": 0, "by_type": {}, "recent_8k_items": []}
    
    by_type: dict[str, int] = {}
    recent_8k_items: list[str] = []
    
    for f in filings:
        form = f.form_type.replace("/A", "")  # Normalize amendments
        by_type[form] = by_type.get(form, 0) + 1
        
        if f.form_type.startswith("8-K") and f.items:
            recent_8k_items.extend(f.items)
    
    # Most recent filing
    most_recent = filings[0] if filings else None
    
    return {
        "total": len(filings),
        "by_type": by_type,
        "recent_8k_items": list(set(recent_8k_items)),
        "8k_categories": categorize_8k_items(recent_8k_items),
        "most_recent_date": most_recent.filed_date if most_recent else None,
        "most_recent_form": most_recent.form_type if most_recent else None,
    }
