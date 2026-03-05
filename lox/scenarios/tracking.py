"""
Prediction tracking — auto-log every scenario run and score against actuals.

Storage: data/predictions/scenario_log.jsonl (one JSON object per line).
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LOG_DIR = Path("data/predictions")
LOG_FILE = LOG_DIR / "scenario_log.jsonl"


# ── Save / Load ──────────────────────────────────────────────────────────


def save_prediction(
    result: Any,
    positions: list[dict] | None = None,
) -> str:
    """Append a prediction record to the JSONL log. Returns the prediction ID."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    horizon_end = date.today() + timedelta(days=result.horizon_days)

    assumptions_dict = {}
    for a in result.assumptions:
        assumptions_dict[a.variable] = {
            "current": a.current,
            "scenario": a.scenario,
            "unit": a.unit,
        }

    record = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "symbol": result.symbol,
        "current_price": result.current_price,
        "horizon_days": result.horizon_days,
        "horizon_end_date": horizon_end.isoformat(),
        "macro_assumptions": assumptions_dict,
        "mc_adjustments": result.mc_adjustments,
        "scenarios": result.scenarios,
        "full_distribution": result.full_distribution,
        "positions": positions or [],
        "actual_price": None,
        "actual_percentile": None,
        "actual_scenario": None,
        "scored_at": None,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record["id"]


def load_predictions(symbol: str | None = None) -> list[dict]:
    """Load all predictions, optionally filtered by symbol."""
    if not LOG_FILE.exists():
        return []

    records = []
    for line in LOG_FILE.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if symbol and rec.get("symbol") != symbol.upper():
                continue
            records.append(rec)
        except json.JSONDecodeError:
            continue

    records.sort(key=lambda r: r.get("timestamp", ""))
    return records


def get_last_prediction(symbol: str) -> dict | None:
    """Get the most recent prediction for a symbol."""
    preds = load_predictions(symbol)
    return preds[-1] if preds else None


# ── Scoring ──────────────────────────────────────────────────────────────


def _compute_percentile(actual: float, dist: dict) -> float:
    """Estimate which percentile the actual price landed in.

    Uses linear interpolation between the stored percentile anchors.
    Returns a value between 0 and 100.
    """
    anchors = [
        (5, dist.get("p5", 0)),
        (10, dist.get("p10", 0)),
        (25, dist.get("p25", 0)),
        (50, dist.get("p50", 0)),
        (75, dist.get("p75", 0)),
        (90, dist.get("p90", 0)),
        (95, dist.get("p95", 0)),
    ]

    if actual <= anchors[0][1]:
        return 2.0
    if actual >= anchors[-1][1]:
        return 98.0

    for i in range(len(anchors) - 1):
        p_lo, v_lo = anchors[i]
        p_hi, v_hi = anchors[i + 1]
        if v_lo <= actual <= v_hi:
            if v_hi == v_lo:
                return float(p_lo)
            frac = (actual - v_lo) / (v_hi - v_lo)
            return p_lo + frac * (p_hi - p_lo)

    return 50.0


def _classify_scenario(actual: float, scenarios: dict) -> str:
    """Which scenario range did the actual price land in?"""
    for key in ("TAIL_RISK", "BEAR", "BASE", "BULL"):
        s = scenarios.get(key, {})
        rng = s.get("range", [0, 0])
        if len(rng) == 2 and rng[0] <= actual <= rng[1]:
            return key
    if actual > scenarios.get("BULL", {}).get("range", [0, 0])[1]:
        return "ABOVE_BULL"
    return "BELOW_TAIL"


def score_predictions(settings: Any) -> list[dict]:
    """Score any completed predictions by fetching actual prices.

    Modifies the log file in-place to persist scores.
    Returns all predictions (scored and unscored).
    """
    preds = load_predictions()
    if not preds:
        return []

    today = date.today()
    updated = False

    for rec in preds:
        if rec.get("actual_price") is not None:
            continue

        end_date = date.fromisoformat(rec["horizon_end_date"])
        if end_date > today:
            continue

        # Fetch actual price at horizon end
        try:
            actual = _fetch_price_on_date(settings, rec["symbol"], end_date)
        except Exception as exc:
            logger.debug("Could not fetch actual for %s: %s", rec["symbol"], exc)
            continue

        if actual is None:
            continue

        rec["actual_price"] = round(actual, 2)
        rec["actual_percentile"] = round(
            _compute_percentile(actual, rec["full_distribution"]), 1
        )
        rec["actual_scenario"] = _classify_scenario(actual, rec["scenarios"])
        rec["scored_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        updated = True

    if updated:
        _rewrite_log(preds)

    return preds


def _fetch_price_on_date(settings: Any, symbol: str, target_date: date) -> float | None:
    """Fetch the closing price for symbol near target_date."""
    from lox.cli_commands.research.ticker.data import fetch_price_data

    price_data = fetch_price_data(settings, symbol)
    historical = price_data.get("historical", [])
    if not historical:
        return None

    # Find the closest trading day <= target_date
    best_price = None
    best_diff = 999
    for bar in historical:
        bar_date = bar.get("date", "")
        if not bar_date:
            continue
        try:
            bd = date.fromisoformat(bar_date)
        except ValueError:
            continue
        diff = abs((bd - target_date).days)
        if diff < best_diff and bd <= target_date + timedelta(days=5):
            best_diff = diff
            best_price = bar.get("close") or bar.get("adjClose")

    return best_price


def _rewrite_log(predictions: list[dict]) -> None:
    """Rewrite the entire log file with updated records."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")


# ── Scorecard ────────────────────────────────────────────────────────────


def compute_scorecard(predictions: list[dict]) -> dict:
    """Compute aggregate accuracy metrics over scored predictions."""
    scored = [p for p in predictions if p.get("actual_price") is not None]
    if not scored:
        return {"n_scored": 0, "n_pending": len(predictions)}

    n = len(scored)
    in_base = sum(1 for p in scored if p.get("actual_scenario") == "BASE")
    in_iqr = sum(
        1 for p in scored
        if 25 <= (p.get("actual_percentile") or 0) <= 75
    )
    in_p10_p90 = sum(
        1 for p in scored
        if 10 <= (p.get("actual_percentile") or 0) <= 90
    )

    mean_pct = sum(p.get("actual_percentile", 50) for p in scored) / n

    return {
        "n_scored": n,
        "n_pending": len(predictions) - n,
        "pct_in_base": round(in_base / n * 100, 1),
        "pct_in_iqr": round(in_iqr / n * 100, 1),
        "pct_in_p10_p90": round(in_p10_p90 / n * 100, 1),
        "mean_percentile": round(mean_pct, 1),
        "calibration_note": _calibration_note(mean_pct, in_iqr / n * 100),
    }


def _calibration_note(mean_pct: float, iqr_hit: float) -> str:
    """One-sentence calibration assessment."""
    if 40 <= mean_pct <= 60 and 40 <= iqr_hit <= 60:
        return "Well-calibrated"
    if mean_pct > 60:
        return "Model skews pessimistic (actuals beating predictions)"
    if mean_pct < 40:
        return "Model skews optimistic (actuals below predictions)"
    if iqr_hit > 65:
        return "Distributions may be too wide (overconfident uncertainty)"
    if iqr_hit < 35:
        return "Distributions may be too narrow (underestimating uncertainty)"
    return "Moderate calibration"
