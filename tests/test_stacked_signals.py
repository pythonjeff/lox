"""Smoke tests for stacked-signal detection logic."""
from __future__ import annotations

from lox.cli_commands.research.ticker.compute import detect_stacked_signals


def _technicals(current=100, high_52w=110, low_52w=80, rsi=50, trend="—", crossover=None, hv=20, macd=None):
    return {
        "current": current,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "rsi": rsi,
        "trend_label": trend,
        "sma_crossover": crossover,
        "volatility_30d": hv,
        "macd_signal": macd,
    }


def _ratings(sb=0, b=0, h=0, s=0, ss=0, consensus="Hold"):
    return {"strongBuy": sb, "buy": b, "hold": h, "sell": s, "strongSell": ss, "consensus": consensus}


def test_crowded_long_top_fires():
    """All analysts Buy + 98th pct + RSI 75 → bear stack."""
    stacks = detect_stacked_signals(
        symbol="ZZZ", is_etf=False,
        technicals=_technicals(current=109, rsi=75, trend="Above all major MAs"),
        fundamentals={},
        earnings_outlook=None,
        ratings_consensus=_ratings(sb=10, b=40, h=5, s=1, ss=0, consensus="Buy"),
        price_target=None, futures_data=None, flow_context=None,
        iv=None, current_price=109,
    )
    assert any(s["name"] == "Crowded long top" and s["direction"] == "bear" for s in stacks)


def test_mean_reversion_candidate_fires():
    """0th pct + oversold + analyst PT upside → bull stack."""
    stacks = detect_stacked_signals(
        symbol="ZZZ", is_etf=False,
        technicals=_technicals(current=81, high_52w=120, low_52w=80, rsi=30, trend="Below all major MAs"),
        fundamentals={},
        earnings_outlook=None,
        ratings_consensus=_ratings(sb=2, b=8, h=5, s=0, ss=0, consensus="Buy"),
        price_target={"targetConsensus": 110},  # +35% upside
        futures_data=None, flow_context=None,
        iv=None, current_price=81,
    )
    assert any(s["name"] == "Mean reversion candidate" and s["direction"] == "bull" for s in stacks)


def test_mechanical_tailwind_fires():
    """Above all MAs + Golden Cross + RSI 60 → bull stack."""
    stacks = detect_stacked_signals(
        symbol="ZZZ", is_etf=False,
        technicals=_technicals(
            rsi=60, trend="Above all major MAs (20/50/200)",
            crossover="Golden Cross (50 > 200)",
            macd="Bullish (MACD above signal)",
        ),
        fundamentals={}, earnings_outlook=None,
        ratings_consensus=None, price_target=None,
        futures_data=None, flow_context=None,
        iv=None, current_price=100,
    )
    assert any(s["name"] == "Mechanical tailwind" and s["direction"] == "bull" for s in stacks)


def test_mechanical_headwind_fires():
    """Below all MAs + Death Cross → bear stack."""
    stacks = detect_stacked_signals(
        symbol="ZZZ", is_etf=False,
        technicals=_technicals(rsi=40, trend="Below all major MAs (20/50/200)",
                                crossover="Death Cross (50 < 200)"),
        fundamentals={}, earnings_outlook=None,
        ratings_consensus=None, price_target=None,
        futures_data=None, flow_context=None,
        iv=None, current_price=85,
    )
    assert any(s["name"] == "Mechanical headwind" and s["direction"] == "bear" for s in stacks)


def test_quality_deteriorating_fires():
    """Negative net margin + below all MAs → bear stack."""
    stacks = detect_stacked_signals(
        symbol="ZZZ", is_etf=False,
        technicals=_technicals(trend="Below all major MAs (20/50/200)"),
        fundamentals={"ratios": {"netProfitMarginTTM": -0.05}},
        earnings_outlook=None,
        ratings_consensus=None, price_target=None,
        futures_data=None, flow_context=None,
        iv=None, current_price=80,
    )
    assert any(s["name"] == "Quality deteriorating" and s["direction"] == "bear" for s in stacks)


def test_sell_side_disconnect_fires():
    """Analyst Buy with +25% PT upside but price below all MAs → caution."""
    stacks = detect_stacked_signals(
        symbol="ZZZ", is_etf=False,
        technicals=_technicals(current=80, rsi=40, trend="Below all major MAs"),
        fundamentals={}, earnings_outlook=None,
        ratings_consensus=_ratings(sb=2, b=10, h=2, s=0, ss=0, consensus="Buy"),
        price_target={"targetConsensus": 100},  # +25% upside
        futures_data=None, flow_context=None,
        iv=None, current_price=80,
    )
    assert any(s["name"] == "Sell-side disconnect" and s["direction"] == "caution" for s in stacks)


def test_flow_confirmed_etf_momentum_fires():
    """ETF + STRONG INFLOWS + above all MAs + high MFI → bull."""
    stacks = detect_stacked_signals(
        symbol="SPY", is_etf=True,
        technicals=_technicals(rsi=60, trend="Above all major MAs (20/50/200)"),
        fundamentals={}, earnings_outlook=None,
        ratings_consensus=None, price_target=None,
        futures_data=None,
        flow_context={"net_flow_signal_20d": "STRONG INFLOWS", "mfi_14d": 72},
        iv=None, current_price=500,
    )
    assert any(s["name"] == "Flow-confirmed ETF momentum" and s["direction"] == "bull" for s in stacks)


def test_no_stacks_when_signals_neutral():
    """Boring middle-of-range neutral RSI no-news case → nothing fires."""
    stacks = detect_stacked_signals(
        symbol="ZZZ", is_etf=False,
        technicals=_technicals(current=95, high_52w=110, low_52w=80, rsi=52, trend="Above 20/50 SMA"),
        fundamentals={"ratios": {"netProfitMarginTTM": 0.10}},
        earnings_outlook=None,
        ratings_consensus=_ratings(sb=3, b=8, h=8, s=2, ss=0, consensus="Hold"),
        price_target={"targetConsensus": 100},  # +5% upside
        futures_data=None, flow_context=None,
        iv=None, current_price=95,
    )
    assert stacks == []


def test_vol_crush_short_straddle_fires():
    """Earnings imminent + stretched + RSI overbought + IV-HV elevated + mixed beats → bear."""
    earnings_outlook = {
        "next_earnings": {"dte": 12},
        "beat_summary": {"beat_rate": 0.50, "avg_abs_move_1d_pct": 3.0},
        "implied_move": None,
    }
    stacks = detect_stacked_signals(
        symbol="ZZZ", is_etf=False,
        technicals=_technicals(current=109, high_52w=110, low_52w=80,
                                rsi=72, trend="Above all major MAs", hv=20),
        fundamentals={},
        earnings_outlook=earnings_outlook,
        ratings_consensus=None, price_target=None,
        futures_data=None, flow_context=None,
        iv=0.30,  # 30% IV vs 20% HV → 10pp spread
        current_price=109,
    )
    assert any(s["name"] == "Vol-crush short straddle setup" and s["direction"] == "bear" for s in stacks)
