"""
Hedge Fund Grade Metrics - Institutional-quality risk/return metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class HedgeFundMetrics:
    """Institutional-grade risk and return metrics."""
    ticker: str
    
    # Risk metrics
    volatility_30d: float  # 30-day annualized volatility
    volatility_90d: float  # 90-day annualized volatility
    volatility_1y: float   # 1-year annualized volatility
    
    # Beta (vs SPY)
    beta_90d: float
    beta_1y: float
    
    # Drawdown analysis
    max_drawdown_1y: float  # Maximum drawdown in past year
    current_drawdown: float  # Current drawdown from peak
    drawdown_duration_days: int  # Days in current drawdown
    
    # Risk-adjusted returns
    sharpe_ratio_ytd: float  # YTD Sharpe (assuming 5% risk-free)
    sharpe_ratio_1y: float   # 1-year Sharpe
    sortino_ratio_1y: float  # 1-year Sortino (downside risk)
    calmar_ratio: float      # Return / Max Drawdown
    
    # Correlation analysis
    correlation_spy: float   # Correlation with SPY
    correlation_qqq: float   # Correlation with QQQ
    
    # Tail risk
    var_95_1d: float         # 95% Value at Risk (1-day)
    cvar_95_1d: float        # Conditional VaR (Expected Shortfall)
    worst_day_1y: float      # Worst single day return (1 year)
    best_day_1y: float       # Best single day return (1 year)
    
    # Skewness and Kurtosis
    skewness: float          # Return distribution skewness
    kurtosis: float          # Return distribution kurtosis (excess)
    
    # Trading metrics
    avg_daily_volume: float   # Average daily dollar volume
    avg_daily_range_pct: float  # Average daily range as %
    
    # Quality scores
    risk_score: int          # 0-100 (lower = less risky)
    quality_score: int       # 0-100 (higher = better risk-adjusted)
    risk_label: str          # Low/Moderate/High/Very High


def calculate_hf_metrics(
    prices: pd.Series,
    ticker: str,
    spy_prices: pd.Series | None = None,
    qqq_prices: pd.Series | None = None,
    volume: pd.Series | None = None,
    high_prices: pd.Series | None = None,
    low_prices: pd.Series | None = None,
    risk_free_rate: float = 0.05,
) -> HedgeFundMetrics:
    """
    Calculate hedge fund grade metrics.
    
    Args:
        prices: Daily close prices
        ticker: Ticker symbol
        spy_prices: SPY prices for beta/correlation (optional)
        qqq_prices: QQQ prices for correlation (optional)
        volume: Daily volume (optional)
        high_prices: Daily highs for range calculation (optional)
        low_prices: Daily lows for range calculation (optional)
        risk_free_rate: Annual risk-free rate (default 5%)
    """
    if prices.empty or len(prices) < 30:
        return _empty_hf_metrics(ticker)
    
    prices = prices.sort_index().dropna()
    returns = prices.pct_change().dropna()
    
    # Annualization factor
    ann_factor = np.sqrt(252)
    
    # Volatility (annualized)
    vol_30d = returns.iloc[-30:].std() * ann_factor if len(returns) >= 30 else 0
    vol_90d = returns.iloc[-90:].std() * ann_factor if len(returns) >= 90 else 0
    vol_1y = returns.iloc[-252:].std() * ann_factor if len(returns) >= 252 else vol_90d
    
    # Beta calculation
    beta_90d = 1.0
    beta_1y = 1.0
    
    # Skip beta calculation for SPY itself
    if ticker.upper() != "SPY" and spy_prices is not None and len(spy_prices) >= 90:
        try:
            spy_returns = spy_prices.pct_change().dropna()
            
            # Align returns
            aligned = pd.concat([returns, spy_returns], axis=1).dropna()
            if len(aligned) >= 60:
                aligned.columns = [ticker, 'SPY']
                
                # 90-day beta
                recent = aligned.iloc[-90:]
                cov = recent.cov().iloc[0, 1]
                var = recent['SPY'].var()
                beta_90d = cov / var if var > 0 else 1.0
                
                # 1-year beta
                yearly = aligned.iloc[-252:] if len(aligned) >= 252 else aligned
                cov = yearly.cov().iloc[0, 1]
                var = yearly['SPY'].var()
                beta_1y = cov / var if var > 0 else beta_90d
        except Exception:
            beta_90d = 1.0
            beta_1y = 1.0
    
    # Drawdown analysis
    rolling_max = prices.expanding().max()
    drawdowns = (prices / rolling_max - 1) * 100
    
    max_dd_1y = abs(drawdowns.iloc[-252:].min()) if len(drawdowns) >= 252 else abs(drawdowns.min())
    current_dd = abs(drawdowns.iloc[-1])
    
    # Drawdown duration
    dd_duration = 0
    for i in range(len(drawdowns) - 1, -1, -1):
        if drawdowns.iloc[i] < -0.1:  # Still in drawdown (>0.1%)
            dd_duration += 1
        else:
            break
    
    # Sharpe ratios
    daily_rf = risk_free_rate / 252
    
    # YTD
    ytd_returns = returns[returns.index >= f"{returns.index[-1].year}-01-01"]
    if len(ytd_returns) > 20:
        excess_ytd = ytd_returns - daily_rf
        sharpe_ytd = (excess_ytd.mean() / excess_ytd.std()) * ann_factor if excess_ytd.std() > 0 else 0
    else:
        sharpe_ytd = 0
    
    # 1-year
    yearly_returns = returns.iloc[-252:] if len(returns) >= 252 else returns
    excess_1y = yearly_returns - daily_rf
    sharpe_1y = (excess_1y.mean() / excess_1y.std()) * ann_factor if excess_1y.std() > 0 else 0
    
    # Sortino ratio (downside deviation only)
    downside_returns = yearly_returns[yearly_returns < 0]
    downside_dev = downside_returns.std() * ann_factor if len(downside_returns) > 0 else vol_1y
    sortino_1y = (yearly_returns.mean() * 252 - risk_free_rate) / downside_dev if downside_dev > 0 else 0
    
    # Calmar ratio
    annual_return = yearly_returns.mean() * 252
    calmar = annual_return / (max_dd_1y / 100) if max_dd_1y > 0 else 0
    
    # Correlations
    corr_spy = 1.0 if ticker.upper() == "SPY" else 0.5
    corr_qqq = 1.0 if ticker.upper() == "QQQ" else 0.5
    
    if ticker.upper() != "SPY" and spy_prices is not None:
        try:
            spy_returns = spy_prices.pct_change().dropna()
            aligned = pd.concat([returns, spy_returns], axis=1).dropna()
            if len(aligned) >= 60:
                corr_spy = aligned.corr().iloc[0, 1]
        except Exception:
            pass
    
    if ticker.upper() != "QQQ" and qqq_prices is not None:
        try:
            qqq_returns = qqq_prices.pct_change().dropna()
            aligned = pd.concat([returns, qqq_returns], axis=1).dropna()
            if len(aligned) >= 60:
                corr_qqq = aligned.corr().iloc[0, 1]
        except Exception:
            pass
    
    # VaR and CVaR (95%)
    var_95 = -np.percentile(returns, 5) * 100
    cvar_95 = -returns[returns <= np.percentile(returns, 5)].mean() * 100 if len(returns) > 0 else var_95
    
    # Best/worst days
    worst_day = returns.iloc[-252:].min() * 100 if len(returns) >= 252 else returns.min() * 100
    best_day = returns.iloc[-252:].max() * 100 if len(returns) >= 252 else returns.max() * 100
    
    # Skewness and kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()  # Excess kurtosis
    
    # Trading metrics
    avg_volume = 0
    if volume is not None and len(volume) >= 20:
        recent_vol = volume.iloc[-20:]
        recent_px = prices.iloc[-20:]
        avg_volume = (recent_vol * recent_px).mean()
    
    avg_range = 0
    if high_prices is not None and low_prices is not None:
        ranges = (high_prices - low_prices) / prices.shift(1) * 100
        avg_range = ranges.iloc[-20:].mean() if len(ranges) >= 20 else 0
    
    # Risk score (0-100, lower = less risky)
    risk_score_components = []
    
    # Volatility component (0-40 points)
    risk_score_components.append(min(vol_1y * 2, 40))
    
    # Beta component (0-20 points)
    risk_score_components.append(min(abs(beta_1y - 1) * 20, 20))
    
    # Drawdown component (0-20 points)
    risk_score_components.append(min(max_dd_1y, 20))
    
    # VaR component (0-20 points)
    risk_score_components.append(min(var_95 * 4, 20))
    
    risk_score = int(min(sum(risk_score_components), 100))
    
    # Risk label
    if risk_score >= 70:
        risk_label = "Very High"
    elif risk_score >= 50:
        risk_label = "High"
    elif risk_score >= 30:
        risk_label = "Moderate"
    else:
        risk_label = "Low"
    
    # Quality score (risk-adjusted, 0-100)
    quality_components = []
    
    # Sharpe contribution (0-40 points)
    quality_components.append(min(max(sharpe_1y * 20, 0), 40))
    
    # Sortino contribution (0-30 points)
    quality_components.append(min(max(sortino_1y * 10, 0), 30))
    
    # Low drawdown contribution (0-30 points)
    quality_components.append(max(30 - max_dd_1y, 0))
    
    quality_score = int(min(sum(quality_components), 100))
    
    return HedgeFundMetrics(
        ticker=ticker,
        volatility_30d=round(vol_30d * 100, 1),
        volatility_90d=round(vol_90d * 100, 1),
        volatility_1y=round(vol_1y * 100, 1),
        beta_90d=round(beta_90d, 2),
        beta_1y=round(beta_1y, 2),
        max_drawdown_1y=round(max_dd_1y, 1),
        current_drawdown=round(current_dd, 1),
        drawdown_duration_days=dd_duration,
        sharpe_ratio_ytd=round(sharpe_ytd, 2),
        sharpe_ratio_1y=round(sharpe_1y, 2),
        sortino_ratio_1y=round(sortino_1y, 2),
        calmar_ratio=round(calmar, 2),
        correlation_spy=round(corr_spy, 2),
        correlation_qqq=round(corr_qqq, 2),
        var_95_1d=round(var_95, 2),
        cvar_95_1d=round(cvar_95, 2),
        worst_day_1y=round(worst_day, 2),
        best_day_1y=round(best_day, 2),
        skewness=round(skewness, 2),
        kurtosis=round(kurtosis, 2),
        avg_daily_volume=avg_volume,
        avg_daily_range_pct=round(avg_range, 2) if not np.isnan(avg_range) else 0,
        risk_score=risk_score,
        quality_score=quality_score,
        risk_label=risk_label,
    )


def _empty_hf_metrics(ticker: str) -> HedgeFundMetrics:
    """Return empty HF metrics for insufficient data."""
    return HedgeFundMetrics(
        ticker=ticker,
        volatility_30d=0, volatility_90d=0, volatility_1y=0,
        beta_90d=1.0, beta_1y=1.0,
        max_drawdown_1y=0, current_drawdown=0, drawdown_duration_days=0,
        sharpe_ratio_ytd=0, sharpe_ratio_1y=0, sortino_ratio_1y=0, calmar_ratio=0,
        correlation_spy=0.5, correlation_qqq=0.5,
        var_95_1d=0, cvar_95_1d=0, worst_day_1y=0, best_day_1y=0,
        skewness=0, kurtosis=0,
        avg_daily_volume=0, avg_daily_range_pct=0,
        risk_score=50, quality_score=50, risk_label="Unknown",
    )
