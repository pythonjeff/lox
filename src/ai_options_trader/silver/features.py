"""
Silver feature vector generation for ML integration.
"""

from __future__ import annotations

from typing import Optional

from ai_options_trader.regimes.schema import RegimeVector
from ai_options_trader.silver.models import SilverInputs, SilverState
from ai_options_trader.silver.regime import SilverRegime


def silver_feature_vector(
    state: SilverState,
    regime: SilverRegime,
) -> RegimeVector:
    """
    Convert silver state + regime to normalized feature vector.
    
    Features are normalized to [-1, 1] or [0, 1] range for ML consumption.
    """
    inp = state.inputs

    # Normalize scores from [-100, 100] to [-1, 1]
    def norm_score(v: Optional[float]) -> float:
        return (v or 0) / 100.0

    # Normalize returns (clip extreme values)
    def norm_ret(v: Optional[float], scale: float = 30.0) -> float:
        return max(-1, min(1, (v or 0) / scale))

    # Normalize z-scores (clip at +/- 3)
    def norm_z(v: Optional[float]) -> float:
        return max(-1, min(1, (v or 0) / 3.0))

    # Normalize volatility (scale by typical range)
    def norm_vol(v: Optional[float], scale: float = 50.0) -> float:
        return min(1, (v or 0) / scale)

    # Normalize GSR (center around 70, scale by 30)
    def norm_gsr(v: Optional[float]) -> float:
        return max(-1, min(1, ((v or 70) - 70) / 30.0))

    # Boolean to float
    def bool_to_float(v: Optional[bool]) -> float:
        if v is True:
            return 1.0
        elif v is False:
            return -1.0
        return 0.0

    # Regime encoding
    regime_scores = {
        "silver_rally": 1.0,
        "silver_squeeze": 0.8,
        "silver_recovery": 0.3,
        "silver_neutral": 0.0,
        "silver_consolidation": 0.0,
        "silver_breakdown": -0.7,
        "silver_capitulation": -1.0,
    }

    features = {
        # Regime
        "regime_score": regime_scores.get(regime.name, 0.0),
        "regime_bullish": 1.0 if "bullish" in regime.tags else (-1.0 if "bearish" in regime.tags else 0.0),
        
        # Trend
        "trend_score": norm_score(inp.trend_score),
        "momentum_score": norm_score(inp.momentum_score),
        "relative_value_score": norm_score(inp.relative_value_score),
        
        # Returns
        "ret_5d": norm_ret(inp.slv_ret_5d_pct, 15),
        "ret_20d": norm_ret(inp.slv_ret_20d_pct, 30),
        "ret_60d": norm_ret(inp.slv_ret_60d_pct, 50),
        
        # Z-scores
        "zscore_20d": norm_z(inp.slv_zscore_20d),
        "zscore_60d": norm_z(inp.slv_zscore_60d),
        
        # MA positioning
        "above_50ma": bool_to_float(inp.slv_above_50ma),
        "above_200ma": bool_to_float(inp.slv_above_200ma),
        "golden_cross": bool_to_float(inp.slv_50ma_above_200ma),
        
        # Distance from MAs (normalized)
        "pct_from_50ma": norm_ret(inp.slv_pct_from_50ma, 20),
        "pct_from_200ma": norm_ret(inp.slv_pct_from_200ma, 30),
        
        # GSR
        "gsr_level": norm_gsr(inp.gsr),
        "gsr_zscore": norm_z(inp.gsr_zscore),
        "gsr_expanding": bool_to_float(inp.gsr_expanding),
        
        # Volatility
        "vol_20d": norm_vol(inp.slv_vol_20d_ann_pct),
        "vol_zscore": norm_z(inp.slv_vol_zscore),
        
        # Correlation
        "spy_corr": (inp.slv_spy_corr_60d or 0.0),
    }

    return RegimeVector(
        pillar="silver",
        asof=state.asof,
        features=features,
    )


def get_put_bias_features(state: SilverState, regime: SilverRegime) -> dict:
    """
    Get features specifically relevant to put position analysis.
    
    Returns dict with:
    - bearish_score: -1 to 1, higher = more bearish (favorable for puts)
    - key_levels: important price levels to watch
    - catalysts: potential regime change triggers
    """
    inp = state.inputs

    # Compute bearish score
    bearish_score = 0.0

    # Trend contribution
    if inp.trend_score is not None:
        bearish_score -= inp.trend_score / 100  # Negative trend = positive bearish

    # MA positioning
    if inp.slv_above_200ma is False:
        bearish_score += 0.2
    if inp.slv_above_50ma is False:
        bearish_score += 0.15
    if inp.slv_50ma_above_200ma is False:
        bearish_score += 0.15

    # GSR expanding (silver weakening)
    if inp.gsr_expanding is True:
        bearish_score += 0.1

    # Regime contribution
    if regime.name == "silver_breakdown":
        bearish_score += 0.3
    elif regime.name == "silver_rally":
        bearish_score -= 0.3
    elif regime.name == "silver_squeeze":
        bearish_score -= 0.4

    bearish_score = max(-1, min(1, bearish_score))

    # Key levels
    key_levels = {}
    if inp.slv_ma_50:
        key_levels["50-day MA"] = inp.slv_ma_50
    if inp.slv_ma_200:
        key_levels["200-day MA"] = inp.slv_ma_200
    if inp.slv_price and inp.slv_ma_200:
        # Death cross target (approx)
        key_levels["Bear target"] = inp.slv_ma_200 * 0.9

    # Catalysts
    catalysts = []
    if inp.slv_above_50ma is True and inp.slv_pct_from_50ma and inp.slv_pct_from_50ma < 2:
        catalysts.append("Near 50-day MA - potential break lower")
    if inp.slv_50ma_above_200ma is True:
        # Check for potential death cross
        if inp.slv_ma_50 and inp.slv_ma_200:
            ma_spread = (inp.slv_ma_50 / inp.slv_ma_200 - 1) * 100
            if ma_spread < 2:
                catalysts.append("50/200 MA converging - death cross risk")
    if inp.gsr and inp.gsr > 85:
        catalysts.append("Extreme GSR - silver may be due for relative bounce")
    if inp.slv_vol_zscore and inp.slv_vol_zscore > 1.5:
        catalysts.append("Elevated volatility - increased move potential")

    return {
        "bearish_score": bearish_score,
        "key_levels": key_levels,
        "catalysts": catalysts,
    }
