from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertLink:
    """
    A curated "expert watch" link.

    We intentionally treat some sources as link-only (no scraping) due to paywalls/ToS/API constraints.
    """

    name: str
    url: str
    notes: str | None = None


def default_expert_links() -> list[ExpertLink]:
    """
    Curated sources requested by the operator.
    """
    return [
        ExpertLink(name="WSJ Markets", url="https://www.wsj.com/news/markets", notes="Paywalled; link-only"),
        ExpertLink(name="Bloomberg Markets", url="https://www.bloomberg.com/markets", notes="Paywalled; link-only"),
        ExpertLink(name="CME FedWatch Tool", url="https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html"),
        ExpertLink(name="CoinDesk", url="https://www.coindesk.com/", notes="Crypto news"),
        ExpertLink(name="POTUS (X/Twitter)", url="https://x.com/POTUS", notes="Official account; requires external access for full content"),
    ]


def expert_links_for_ticker(ticker: str) -> list[ExpertLink]:
    """
    Return a small relevant subset of expert links for a given ticker.
    """
    t = (ticker or "").strip().upper()
    links = default_expert_links()
    if t in {"IBIT", "BTC", "ETH"}:
        # Crypto-focused: CoinDesk + POTUS (policy/regulatory headline risk).
        return [l for l in links if l.name in {"CoinDesk", "POTUS (X/Twitter)"}]
    if t in {"TBF", "TLT", "IEF", "SHY", "TMV", "TBT"}:
        # Rates: FedWatch is most relevant.
        return [l for l in links if l.name in {"CME FedWatch Tool", "Bloomberg Markets"}]
    # Default: broad markets.
    return [l for l in links if l.name in {"WSJ Markets", "Bloomberg Markets"}]

