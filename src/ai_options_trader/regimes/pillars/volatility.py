"""
Volatility Pillar - VIX and vol surface analysis.

Metrics:
- VIX level and percentile
- VIX term structure (contango/backwardation)
- Realized vs implied vol
- VIX momentum

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Any

from ai_options_trader.regimes.core import RegimePillar, Metric, register_pillar


@register_pillar("volatility")
@dataclass
class VolatilityPillar(RegimePillar):
    """
    Volatility regime pillar.
    
    Tracks implied and realized volatility:
    - VIX level, percentile, momentum
    - Term structure (contango/backwardation)
    - Realized vs implied gap
    """
    name: str = "Volatility"
    description: str = "VIX and volatility surface"
    metrics: List[Metric] = field(default_factory=list)
    
    def compute(self, settings: Any) -> None:
        """Fetch volatility data and compute metrics."""
        from ai_options_trader.data.fred import FredClient
        
        if not settings.FRED_API_KEY:
            return
        
        fred = FredClient(api_key=settings.FRED_API_KEY)
        
        # VIX
        try:
            vix_df = fred.fetch_series("VIXCLS", start_date="2020-01-01")
            if vix_df is not None and not vix_df.empty:
                vix_df = vix_df.sort_values("date")
                vix = vix_df["value"].iloc[-1]
                
                # 5-day and 20-day changes
                if len(vix_df) > 5:
                    delta_1w = vix - vix_df["value"].iloc[-6]
                else:
                    delta_1w = None
                
                if len(vix_df) > 22:
                    delta_1m = vix - vix_df["value"].iloc[-23]
                else:
                    delta_1m = None
                
                # Percentile (1-year lookback)
                lookback = min(252, len(vix_df))
                if lookback > 20:
                    recent = vix_df["value"].iloc[-lookback:]
                    percentile = (recent < vix).sum() / lookback * 100
                else:
                    percentile = None
                
                self.metrics.append(Metric(
                    name="VIX",
                    value=vix,
                    unit="",
                    delta_1w=delta_1w,
                    delta_1m=delta_1m,
                    percentile=percentile,
                    threshold_low=15,
                    threshold_high=25,
                    source="FRED:VIXCLS",
                ))
                
                if percentile is not None:
                    self.metrics.append(Metric(
                        name="VIX Percentile",
                        value=percentile,
                        unit="%",
                        threshold_low=25,
                        threshold_high=75,
                        source="Computed",
                    ))
                
                # 20-day realized vol (approximation from VIX history)
                if len(vix_df) > 20:
                    vol_20d = vix_df["value"].iloc[-20:].std()
                    self.metrics.append(Metric(
                        name="VIX 20d StdDev",
                        value=vol_20d,
                        unit="",
                        source="Computed",
                    ))
        except Exception:
            pass
        
        # VIX 3-month (VIX3M) for term structure
        try:
            vix3m_df = fred.fetch_series("VXVCLS", start_date="2020-01-01")  # VIX 3-month
            if vix3m_df is not None and not vix3m_df.empty:
                vix3m_df = vix3m_df.sort_values("date")
                vix3m = vix3m_df["value"].iloc[-1]
                
                self.metrics.append(Metric(
                    name="VIX 3M",
                    value=vix3m,
                    unit="",
                    source="FRED:VXVCLS",
                ))
                
                # Term structure
                vix_metric = self.get_metric("VIX")
                if vix_metric and vix_metric.value is not None:
                    term_ratio = vix3m / vix_metric.value
                    self.metrics.append(Metric(
                        name="VIX Term Structure",
                        value=term_ratio,
                        unit="ratio",
                        threshold_low=0.95,  # Backwardation
                        threshold_high=1.10,  # Normal contango
                        source="Computed",
                    ))
        except Exception:
            pass
        
        self._compute_score()
    
    def _compute_score(self) -> None:
        """Compute composite volatility score (0-100, higher = more vol)."""
        scores = []
        
        vix = self.get_metric("VIX")
        if vix and vix.value is not None:
            if vix.value < 15:
                scores.append(20)
            elif vix.value < 20:
                scores.append(40)
            elif vix.value < 25:
                scores.append(60)
            elif vix.value < 35:
                scores.append(80)
            else:
                scores.append(95)
        
        vix_pct = self.get_metric("VIX Percentile")
        if vix_pct and vix_pct.value is not None:
            scores.append(vix_pct.value)
        
        # Term structure adjustment
        term = self.get_metric("VIX Term Structure")
        if term and term.value is not None:
            if term.value < 0.95:  # Backwardation = near-term stress
                scores.append(80)
            elif term.value > 1.10:  # Steep contango = complacency
                scores.append(30)
            else:
                scores.append(50)
        
        if scores:
            self.composite_score = sum(scores) / len(scores)
        
        self.regime = self.classify()
    
    def classify(self) -> str:
        """Classify volatility regime."""
        if self.composite_score is None:
            return "UNKNOWN"
        
        vix = self.get_metric("VIX")
        term = self.get_metric("VIX Term Structure")
        
        score = self.composite_score
        
        # Check for specific conditions
        if vix and vix.value is not None and vix.value > 35:
            return "CRISIS"
        
        if term and term.value is not None and term.value < 0.90:
            return "STRESSED"  # Severe backwardation
        
        if score >= 70:
            return "ELEVATED"
        elif score >= 50:
            return "MODERATE"
        elif score >= 30:
            return "LOW"
        else:
            return "COMPLACENT"
    
    def term_structure_status(self) -> str:
        """Describe term structure state."""
        term = self.get_metric("VIX Term Structure")
        if term is None or term.value is None:
            return "Unknown"
        
        ratio = term.value
        if ratio < 0.90:
            return "Severe backwardation (near-term stress)"
        elif ratio < 0.98:
            return "Backwardation (caution)"
        elif ratio < 1.05:
            return "Flat (transition)"
        elif ratio < 1.15:
            return "Normal contango"
        else:
            return "Steep contango (complacency)"
    
    def deep_dive(self) -> dict:
        """Hedge-fund style deep dive."""
        return {
            "regime": self.regime,
            "score": self.composite_score,
            "term_structure": self.term_structure_status(),
            "metrics": {m.name: m.value for m in self.metrics},
            "momentum": self._momentum_analysis(),
            "interpretation": self._interpret(),
            "trade_implications": self._trade_implications(),
        }
    
    def _momentum_analysis(self) -> dict:
        """Analyze VIX momentum."""
        vix = self.get_metric("VIX")
        
        analysis = {
            "short_term": "stable",
            "medium_term": "stable",
            "spike_risk": "low",
        }
        
        if vix:
            if vix.delta_1w is not None:
                if vix.delta_1w > 3:
                    analysis["short_term"] = "spiking"
                elif vix.delta_1w > 1:
                    analysis["short_term"] = "rising"
                elif vix.delta_1w < -2:
                    analysis["short_term"] = "collapsing"
                elif vix.delta_1w < -1:
                    analysis["short_term"] = "falling"
            
            if vix.delta_1m is not None:
                if vix.delta_1m > 5:
                    analysis["medium_term"] = "elevated"
                elif vix.delta_1m < -5:
                    analysis["medium_term"] = "suppressed"
            
            # Spike risk based on level and term structure
            term = self.get_metric("VIX Term Structure")
            if vix.value is not None and vix.value < 15:
                analysis["spike_risk"] = "elevated"  # Low VIX = vulnerable to spikes
            elif term and term.value is not None and term.value < 0.95:
                analysis["spike_risk"] = "high"  # Backwardation = near-term stress
        
        return analysis
    
    def _trade_implications(self) -> List[str]:
        """Trading implications for vol regime."""
        implications = []
        
        vix = self.get_metric("VIX")
        term = self.get_metric("VIX Term Structure")
        
        if vix and vix.value is not None:
            if vix.value < 15:
                implications.append("Vol cheap — consider buying protection")
            elif vix.value > 30:
                implications.append("Vol expensive — consider selling or waiting")
        
        if term and term.value is not None:
            if term.value < 0.95:
                implications.append("Backwardation — avoid long VIX futures")
            elif term.value > 1.15:
                implications.append("Steep contango — VIX futures decay accelerated")
        
        return implications
    
    def _interpret(self) -> str:
        """Generate interpretation."""
        parts = []
        
        vix = self.get_metric("VIX")
        if vix and vix.value is not None:
            if vix.value < 15:
                parts.append(f"VIX at {vix.value:.1f} indicates complacency.")
            elif vix.value < 20:
                parts.append(f"VIX at {vix.value:.1f} — low but normal.")
            elif vix.value < 25:
                parts.append(f"VIX at {vix.value:.1f} — moderately elevated.")
            elif vix.value < 35:
                parts.append(f"VIX at {vix.value:.1f} — elevated uncertainty.")
            else:
                parts.append(f"VIX at {vix.value:.1f} — crisis-level vol.")
            
            if vix.percentile is not None:
                parts.append(f"({vix.percentile:.0f}th percentile over 1Y)")
        
        term_status = self.term_structure_status()
        if "Unknown" not in term_status:
            parts.append(f"Term structure: {term_status}.")
        
        return " ".join(parts) if parts else "Insufficient data."
