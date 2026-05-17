"""Bubble regime — signal compute layer.

Pulls the five pillars (valuation, concentration, margin debt, speculation,
sentiment) and packages them into a single dataclass for the classifier and
CLI command.
"""
from __future__ import annotations

from dataclasses import dataclass

from lox.bubble.concentration import ConcentrationSnapshot, fetch_concentration_snapshot
from lox.bubble.margin import MarginSnapshot, fetch_margin_snapshot
from lox.bubble.sentiment import SentimentSnapshot, fetch_sentiment_snapshot
from lox.bubble.speculation import SpeculationSnapshot, fetch_speculation_snapshot
from lox.bubble.valuation import BuffettSnapshot, fetch_buffett_snapshot
from lox.config import Settings


@dataclass
class BubbleSignals:
    asof: str
    valuation: BuffettSnapshot
    concentration: ConcentrationSnapshot
    margin: MarginSnapshot
    speculation: SpeculationSnapshot
    sentiment: SentimentSnapshot


def compute_bubble_signals(*, settings: Settings, refresh: bool = False) -> BubbleSignals:
    val = fetch_buffett_snapshot(settings=settings, refresh=refresh)
    con = fetch_concentration_snapshot(settings=settings, refresh=refresh)
    mar = fetch_margin_snapshot(settings=settings, refresh=refresh)
    spec = fetch_speculation_snapshot(settings=settings, refresh=refresh)
    sent = fetch_sentiment_snapshot(settings=settings, refresh=refresh)

    # asof = freshest pillar so the panel header isn't blank
    candidates = [d for d in (con.asof, sent.asof, spec.asof, val.asof, mar.asof) if d]
    asof = max(candidates) if candidates else ""

    return BubbleSignals(asof=asof, valuation=val, concentration=con,
                         margin=mar, speculation=spec, sentiment=sent)
