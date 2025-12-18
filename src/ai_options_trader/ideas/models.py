from __future__ import annotations

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field  # type: ignore


class Idea(BaseModel):
    ticker: str
    direction: str  # "bullish"|"bearish"|"neutral"
    score: float
    tags: List[str] = Field(default_factory=list)
    thesis: str = ""
    rationale: str = ""

    # Optional: filled when we fetch option legs
    option_leg: Optional[Dict[str, Any]] = None

    # Explainability payload used by CLI to print "why"
    why: Dict[str, Any] = Field(default_factory=dict)


