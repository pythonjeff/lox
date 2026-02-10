"""
Unified Regime Framework - Core abstractions.

Design:
1. Each regime pillar is a self-contained module with:
   - Raw metrics (from FRED/market data)
   - Z-scores for ML training
   - A composite score
   - A classification (enum)
   - Display helpers

2. All pillars share a common interface for:
   - ML feature extraction
   - Dashboard display
   - Deep-dive analysis

Author: Lox Capital Research
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Type
import pandas as pd


class RegimeLevel(str, Enum):
    """Standard regime classification levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    NEUTRAL = "neutral"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class Metric:
    """
    A single economic/market metric with context.
    
    This is the atomic unit of the regime framework.
    """
    name: str                          # e.g., "CPI YoY"
    value: Optional[float]             # Current value
    unit: str = ""                     # e.g., "%", "bps", "$B"
    
    # Context for interpretation
    z_score: Optional[float] = None    # Standardized value
    percentile: Optional[float] = None # Historical percentile (0-100)
    
    # Changes
    delta_1d: Optional[float] = None
    delta_1w: Optional[float] = None
    delta_1m: Optional[float] = None
    delta_3m: Optional[float] = None
    
    # Thresholds for classification
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None
    
    # Metadata
    source: str = ""                   # e.g., "FRED:CPIAUCSL"
    asof: Optional[date] = None
    
    @property
    def level(self) -> RegimeLevel:
        """Classify based on z-score."""
        if self.z_score is None:
            return RegimeLevel.NEUTRAL
        z = self.z_score
        if z <= -2.0:
            return RegimeLevel.VERY_LOW
        elif z <= -1.0:
            return RegimeLevel.LOW
        elif z <= 1.0:
            return RegimeLevel.NEUTRAL
        elif z <= 2.0:
            return RegimeLevel.ELEVATED
        elif z <= 3.0:
            return RegimeLevel.HIGH
        else:
            return RegimeLevel.EXTREME
    
    @property
    def trend(self) -> str:
        """Determine trend direction from deltas."""
        if self.delta_1m is not None:
            if self.delta_1m > 0.5:
                return "rising"
            elif self.delta_1m < -0.5:
                return "falling"
        return "stable"
    
    def format_value(self) -> str:
        """Format value with unit."""
        if self.value is None:
            return "N/A"
        if self.unit == "%":
            return f"{self.value:.2f}%"
        elif self.unit == "bps":
            return f"{self.value:.0f}bp"
        elif self.unit == "$B":
            return f"${self.value:.0f}B"
        elif self.unit == "$T":
            return f"${self.value:.2f}T"
        else:
            return f"{self.value:.2f}{self.unit}"
    
    def format_delta(self, period: str = "1m") -> str:
        """Format change for a period."""
        delta = getattr(self, f"delta_{period}", None)
        if delta is None:
            return "N/A"
        sign = "+" if delta >= 0 else ""
        if self.unit == "%":
            return f"{sign}{delta:.2f}pp"
        elif self.unit == "bps":
            return f"{sign}{delta:.0f}bp"
        else:
            return f"{sign}{delta:.2f}"
    
    def to_feature(self) -> Dict[str, float]:
        """Extract ML features from this metric."""
        features = {}
        key = self.name.lower().replace(" ", "_").replace("/", "_")
        
        if self.value is not None:
            features[f"{key}_level"] = self.value
        if self.z_score is not None:
            features[f"{key}_zscore"] = self.z_score
        if self.delta_1m is not None:
            features[f"{key}_mom_1m"] = self.delta_1m
        if self.delta_3m is not None:
            features[f"{key}_mom_3m"] = self.delta_3m
        
        return features


@dataclass
class RegimePillar(ABC):
    """
    Base class for a regime pillar (e.g., Inflation, Growth, Liquidity).
    
    Each pillar contains multiple metrics and produces:
    - A composite score
    - A regime classification
    - ML features
    - Dashboard display
    """
    name: str
    description: str
    metrics: List[Metric] = field(default_factory=list)
    
    # Computed
    composite_score: Optional[float] = None
    regime: Optional[str] = None
    
    @abstractmethod
    def compute(self, settings: Any) -> None:
        """Fetch data and compute metrics."""
        pass
    
    @abstractmethod
    def classify(self) -> str:
        """Classify into a regime label."""
        pass
    
    def to_features(self) -> Dict[str, float]:
        """Extract all ML features from this pillar."""
        features = {}
        prefix = self.name.lower().replace(" ", "_")
        
        for metric in self.metrics:
            metric_features = metric.to_feature()
            for k, v in metric_features.items():
                features[f"{prefix}_{k}"] = v
        
        if self.composite_score is not None:
            features[f"{prefix}_score"] = self.composite_score
        
        return features
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a specific metric by name."""
        for m in self.metrics:
            if m.name == name:
                return m
        return None
    
    def summary(self) -> str:
        """One-line summary for dashboard."""
        level = self.regime or "UNKNOWN"
        score = f"({self.composite_score:.1f})" if self.composite_score else ""
        return f"{level} {score}"


# Registry of all pillar types
_PILLAR_REGISTRY: Dict[str, Type[RegimePillar]] = {}


def register_pillar(name: str):
    """Decorator to register a pillar type."""
    def decorator(cls: Type[RegimePillar]):
        _PILLAR_REGISTRY[name] = cls
        return cls
    return decorator


def get_pillar(name: str) -> Optional[Type[RegimePillar]]:
    """Get a pillar class by name."""
    return _PILLAR_REGISTRY.get(name)


def list_pillars() -> List[str]:
    """List all registered pillar names."""
    return list(_PILLAR_REGISTRY.keys())
