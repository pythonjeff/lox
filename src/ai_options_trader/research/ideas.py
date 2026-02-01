"""
Trade Ideas Generator - Find overbought/oversold opportunities.

Identifies stocks that may have moved irrationally and could be candidates
for mean reversion trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd
import numpy as np

from ai_options_trader.config import Settings
from ai_options_trader.research.momentum import MomentumMetrics, calculate_momentum_metrics


@dataclass
class TradeIdea:
    """A potential trade idea based on technical extremes."""
    ticker: str
    direction: Literal["long", "short"]  # long = buy oversold, short = sell overbought
    signal_type: str  # "oversold_bounce", "overbought_fade", "momentum_continuation"
    
    # Current state
    current_price: float
    
    # Why this idea
    momentum_score: int
    rsi: float
    return_1m: float
    return_3m: float
    pct_from_52w_high: float
    
    # Scoring
    conviction_score: int  # 0-100
    reasoning: list[str]
    
    # Suggested trade
    suggested_entry: str  # "at market", "on pullback to $X", etc.
    suggested_target_pct: float
    suggested_stop_pct: float
    risk_reward: float


@dataclass
class IdeasReport:
    """Collection of trade ideas."""
    generated_at: str
    universe_size: int
    
    # Ideas by type
    oversold_ideas: list[TradeIdea]
    overbought_ideas: list[TradeIdea]
    momentum_ideas: list[TradeIdea]
    
    # Summary
    market_breadth: str  # "bullish", "bearish", "mixed"
    avg_rsi: float
    pct_oversold: float
    pct_overbought: float


def generate_trade_ideas(
    settings: Settings,
    universe: list[str],
    min_conviction: int = 50,
    max_ideas: int = 10,
) -> IdeasReport:
    """
    Generate trade ideas by scanning a universe for extremes.
    
    Args:
        settings: Application settings
        universe: List of ticker symbols to scan
        min_conviction: Minimum conviction score (0-100)
        max_ideas: Maximum ideas per category
    
    Returns:
        IdeasReport with categorized trade ideas
    """
    from ai_options_trader.data.market import fetch_equity_daily_closes
    
    # Fetch prices for universe
    try:
        px_df = fetch_equity_daily_closes(
            settings=settings,
            symbols=universe,
            start="2023-01-01",
            refresh=False,
        )
    except Exception:
        px_df = pd.DataFrame()
    
    oversold_ideas = []
    overbought_ideas = []
    momentum_ideas = []
    
    rsi_values = []
    
    for ticker in universe:
        if ticker not in px_df.columns:
            continue
        
        prices = px_df[ticker].dropna()
        if len(prices) < 50:
            continue
        
        # Calculate momentum
        momentum = calculate_momentum_metrics(prices, ticker)
        current_price = float(prices.iloc[-1])
        
        rsi_values.append(momentum.rsi_14)
        
        # Check for oversold conditions
        oversold_signals = []
        oversold_score = 0
        
        if momentum.rsi_14 < 30:
            oversold_signals.append(f"RSI deeply oversold ({momentum.rsi_14:.0f})")
            oversold_score += 30
        elif momentum.rsi_14 < 40:
            oversold_signals.append(f"RSI oversold ({momentum.rsi_14:.0f})")
            oversold_score += 15
        
        if momentum.pct_from_52w_high < -30:
            oversold_signals.append(f"Down {abs(momentum.pct_from_52w_high):.0f}% from 52W high")
            oversold_score += 25
        elif momentum.pct_from_52w_high < -20:
            oversold_signals.append(f"Down {abs(momentum.pct_from_52w_high):.0f}% from 52W high")
            oversold_score += 15
        
        if momentum.return_1m < -15:
            oversold_signals.append(f"Sharp 1M decline ({momentum.return_1m:.0f}%)")
            oversold_score += 20
        elif momentum.return_1m < -10:
            oversold_signals.append(f"1M decline ({momentum.return_1m:.0f}%)")
            oversold_score += 10
        
        if momentum.momentum_score < -40:
            oversold_signals.append(f"Strong bearish momentum ({momentum.momentum_score})")
            oversold_score += 15
        
        # Bounce signal: oversold with recent stabilization
        if len(prices) >= 5:
            recent_returns = prices.pct_change().iloc[-5:]
            if recent_returns.iloc[-1] > 0 and recent_returns.iloc[-2] > 0:
                oversold_signals.append("Showing signs of stabilization")
                oversold_score += 10
        
        if oversold_score >= min_conviction and oversold_signals:
            # Calculate trade parameters
            target_pct = min(abs(momentum.pct_from_52w_high) * 0.3, 20)  # Aim for 30% recovery, max 20%
            stop_pct = 8  # 8% stop loss
            rr = target_pct / stop_pct
            
            oversold_ideas.append(TradeIdea(
                ticker=ticker,
                direction="long",
                signal_type="oversold_bounce",
                current_price=current_price,
                momentum_score=momentum.momentum_score,
                rsi=momentum.rsi_14,
                return_1m=momentum.return_1m,
                return_3m=momentum.return_3m,
                pct_from_52w_high=momentum.pct_from_52w_high,
                conviction_score=min(oversold_score, 100),
                reasoning=oversold_signals,
                suggested_entry="at market" if momentum.rsi_14 < 25 else f"on pullback to ${current_price * 0.98:.2f}",
                suggested_target_pct=round(target_pct, 1),
                suggested_stop_pct=stop_pct,
                risk_reward=round(rr, 1),
            ))
        
        # Check for overbought conditions
        overbought_signals = []
        overbought_score = 0
        
        if momentum.rsi_14 > 70:
            overbought_signals.append(f"RSI deeply overbought ({momentum.rsi_14:.0f})")
            overbought_score += 30
        elif momentum.rsi_14 > 60:
            overbought_signals.append(f"RSI overbought ({momentum.rsi_14:.0f})")
            overbought_score += 15
        
        if momentum.pct_from_52w_high > -5:
            overbought_signals.append("Near 52W high")
            overbought_score += 20
        
        if momentum.return_1m > 15:
            overbought_signals.append(f"Sharp 1M rally ({momentum.return_1m:+.0f}%)")
            overbought_score += 25
        elif momentum.return_1m > 10:
            overbought_signals.append(f"1M rally ({momentum.return_1m:+.0f}%)")
            overbought_score += 15
        
        if momentum.return_3m > 30:
            overbought_signals.append(f"Extended 3M rally ({momentum.return_3m:+.0f}%)")
            overbought_score += 20
        
        if momentum.momentum_score > 60:
            overbought_signals.append(f"Extreme bullish momentum ({momentum.momentum_score})")
            overbought_score += 10
        
        if overbought_score >= min_conviction and overbought_signals:
            target_pct = min(momentum.return_1m * 0.3, 10)  # Aim for 30% giveback, max 10%
            stop_pct = 5
            rr = target_pct / stop_pct
            
            overbought_ideas.append(TradeIdea(
                ticker=ticker,
                direction="short",
                signal_type="overbought_fade",
                current_price=current_price,
                momentum_score=momentum.momentum_score,
                rsi=momentum.rsi_14,
                return_1m=momentum.return_1m,
                return_3m=momentum.return_3m,
                pct_from_52w_high=momentum.pct_from_52w_high,
                conviction_score=min(overbought_score, 100),
                reasoning=overbought_signals,
                suggested_entry=f"on rally to ${current_price * 1.02:.2f}",
                suggested_target_pct=round(target_pct, 1),
                suggested_stop_pct=stop_pct,
                risk_reward=round(rr, 1),
            ))
        
        # Check for momentum continuation
        mom_signals = []
        mom_score = 0
        
        if 40 < momentum.momentum_score < 60 and momentum.trend_direction == "bullish":
            if momentum.above_sma_20 and momentum.above_sma_50:
                mom_signals.append("Uptrend with room to run")
                mom_score += 20
            if 45 < momentum.rsi_14 < 65:
                mom_signals.append(f"RSI neutral ({momentum.rsi_14:.0f})")
                mom_score += 15
            if 0 < momentum.return_1m < 8:
                mom_signals.append("Healthy 1M gain")
                mom_score += 15
            if momentum.golden_cross:
                mom_signals.append("Golden cross in place")
                mom_score += 20
            
            if mom_score >= min_conviction and mom_signals:
                target_pct = 10
                stop_pct = 5
                
                momentum_ideas.append(TradeIdea(
                    ticker=ticker,
                    direction="long",
                    signal_type="momentum_continuation",
                    current_price=current_price,
                    momentum_score=momentum.momentum_score,
                    rsi=momentum.rsi_14,
                    return_1m=momentum.return_1m,
                    return_3m=momentum.return_3m,
                    pct_from_52w_high=momentum.pct_from_52w_high,
                    conviction_score=min(mom_score, 100),
                    reasoning=mom_signals,
                    suggested_entry="at market",
                    suggested_target_pct=target_pct,
                    suggested_stop_pct=stop_pct,
                    risk_reward=round(target_pct / stop_pct, 1),
                ))
    
    # Sort by conviction and limit
    oversold_ideas.sort(key=lambda x: x.conviction_score, reverse=True)
    overbought_ideas.sort(key=lambda x: x.conviction_score, reverse=True)
    momentum_ideas.sort(key=lambda x: x.conviction_score, reverse=True)
    
    # Calculate market breadth
    avg_rsi = np.mean(rsi_values) if rsi_values else 50
    pct_oversold = sum(1 for r in rsi_values if r < 30) / len(rsi_values) * 100 if rsi_values else 0
    pct_overbought = sum(1 for r in rsi_values if r > 70) / len(rsi_values) * 100 if rsi_values else 0
    
    if avg_rsi > 55:
        market_breadth = "bullish"
    elif avg_rsi < 45:
        market_breadth = "bearish"
    else:
        market_breadth = "mixed"
    
    return IdeasReport(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        universe_size=len(universe),
        oversold_ideas=oversold_ideas[:max_ideas],
        overbought_ideas=overbought_ideas[:max_ideas],
        momentum_ideas=momentum_ideas[:max_ideas],
        market_breadth=market_breadth,
        avg_rsi=round(avg_rsi, 1),
        pct_oversold=round(pct_oversold, 1),
        pct_overbought=round(pct_overbought, 1),
    )


# Default universe for scanning
DEFAULT_SCAN_UNIVERSE = [
    # Mega caps
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
    # Tech
    "AMD", "AVGO", "CRM", "ADBE", "ORCL", "CSCO", "INTC", "IBM", "QCOM", "TXN",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "CME",
    # Healthcare
    "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD",
    # Consumer
    "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "BKNG",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "PSX", "HES",
    # Industrials
    "CAT", "DE", "BA", "HON", "UPS", "LMT", "RTX", "GE", "MMM", "UNP",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "GLD",
    "TLT", "HYG", "EEM", "VWO", "ARKK", "SMH", "XBI", "XHB", "KWEB", "FXI",
]
