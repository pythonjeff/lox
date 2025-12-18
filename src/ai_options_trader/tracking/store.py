from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def _json_loads(s: str | None) -> Any:
    if not s:
        return None
    return json.loads(s)


@dataclass(frozen=True)
class RecommendationRow:
    id: str
    run_id: str
    created_at: str
    ticker: str
    direction: str
    score: float
    tags_json: str
    thesis: str
    rationale: str
    why_json: str
    option_leg_json: str | None


@dataclass(frozen=True)
class ExecutionRow:
    id: str
    recommendation_id: str
    created_at: str
    alpaca_order_id: str
    symbol: str
    qty: int
    side: str
    order_type: str
    limit_price: float | None
    status: str
    filled_qty: int | None
    filled_avg_price: float | None
    filled_at: str | None
    raw_json: str | None
    last_sync_at: str | None


class TrackerStore:
    """
    Minimal SQLite-backed store for:
    - recommendations (ideas generated)
    - executions (orders submitted)

    This intentionally keeps JSON blobs for flexibility; we can normalize later.
    """

    def __init__(self, db_path: str = "data/tracker.sqlite3"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendations (
                  id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  ticker TEXT NOT NULL,
                  direction TEXT NOT NULL,
                  score REAL NOT NULL,
                  tags_json TEXT NOT NULL,
                  thesis TEXT NOT NULL,
                  rationale TEXT NOT NULL,
                  why_json TEXT NOT NULL,
                  option_leg_json TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reco_run_id ON recommendations(run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reco_ticker ON recommendations(ticker);")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS executions (
                  id TEXT PRIMARY KEY,
                  recommendation_id TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  alpaca_order_id TEXT NOT NULL,
                  symbol TEXT NOT NULL,
                  qty INTEGER NOT NULL,
                  side TEXT NOT NULL,
                  order_type TEXT NOT NULL,
                  limit_price REAL,
                  status TEXT NOT NULL,
                  filled_qty INTEGER,
                  filled_avg_price REAL,
                  filled_at TEXT,
                  raw_json TEXT,
                  last_sync_at TEXT,
                  FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_order_id ON executions(alpaca_order_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_reco ON executions(recommendation_id);")

    def new_run_id(self) -> str:
        return str(uuid.uuid4())

    def log_recommendation(
        self,
        *,
        run_id: str,
        ticker: str,
        direction: str,
        score: float,
        tags: list[str],
        thesis: str,
        rationale: str,
        why: dict[str, Any],
        option_leg: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> str:
        self.ensure_schema()
        reco_id = str(uuid.uuid4())
        created_at = created_at or _utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO recommendations (
                  id, run_id, created_at, ticker, direction, score,
                  tags_json, thesis, rationale, why_json, option_leg_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    reco_id,
                    run_id,
                    created_at,
                    ticker,
                    direction,
                    float(score),
                    _json_dumps(tags),
                    thesis,
                    rationale,
                    _json_dumps(why),
                    _json_dumps(option_leg) if option_leg else None,
                ),
            )
        return reco_id

    def log_execution(
        self,
        *,
        recommendation_id: str,
        alpaca_order_id: str,
        symbol: str,
        qty: int,
        side: str,
        order_type: str,
        limit_price: float | None,
        status: str,
        filled_qty: int | None = None,
        filled_avg_price: float | None = None,
        filled_at: str | None = None,
        raw: Any | None = None,
        created_at: str | None = None,
    ) -> str:
        self.ensure_schema()
        exec_id = str(uuid.uuid4())
        created_at = created_at or _utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO executions (
                  id, recommendation_id, created_at, alpaca_order_id, symbol, qty, side, order_type,
                  limit_price, status, filled_qty, filled_avg_price, filled_at, raw_json, last_sync_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    exec_id,
                    recommendation_id,
                    created_at,
                    alpaca_order_id,
                    symbol,
                    int(qty),
                    side,
                    order_type,
                    float(limit_price) if limit_price is not None else None,
                    status,
                    int(filled_qty) if filled_qty is not None else None,
                    float(filled_avg_price) if filled_avg_price is not None else None,
                    filled_at,
                    _json_dumps(raw) if raw is not None else None,
                    _utc_now_iso(),
                ),
            )
        return exec_id

    def list_recent_recommendations(self, limit: int = 20) -> list[RecommendationRow]:
        self.ensure_schema()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, run_id, created_at, ticker, direction, score,
                       tags_json, thesis, rationale, why_json, option_leg_json
                FROM recommendations
                ORDER BY created_at DESC
                LIMIT ?;
                """,
                (int(limit),),
            ).fetchall()
        return [RecommendationRow(**dict(r)) for r in rows]

    def list_recent_executions(self, limit: int = 20) -> list[ExecutionRow]:
        self.ensure_schema()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, recommendation_id, created_at, alpaca_order_id, symbol, qty, side, order_type,
                       limit_price, status, filled_qty, filled_avg_price, filled_at, raw_json, last_sync_at
                FROM executions
                ORDER BY created_at DESC
                LIMIT ?;
                """,
                (int(limit),),
            ).fetchall()
        return [ExecutionRow(**dict(r)) for r in rows]

    def update_execution_from_alpaca(
        self,
        *,
        alpaca_order_id: str,
        status: str | None = None,
        filled_qty: int | None = None,
        filled_avg_price: float | None = None,
        filled_at: str | None = None,
        raw: Any | None = None,
    ) -> None:
        self.ensure_schema()
        sets = []
        params: list[Any] = []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if filled_qty is not None:
            sets.append("filled_qty = ?")
            params.append(int(filled_qty))
        if filled_avg_price is not None:
            sets.append("filled_avg_price = ?")
            params.append(float(filled_avg_price))
        if filled_at is not None:
            sets.append("filled_at = ?")
            params.append(filled_at)
        if raw is not None:
            sets.append("raw_json = ?")
            params.append(_json_dumps(raw))
        sets.append("last_sync_at = ?")
        params.append(_utc_now_iso())

        if not sets:
            return
        params.append(alpaca_order_id)

        with self.connect() as conn:
            conn.execute(
                f"UPDATE executions SET {', '.join(sets)} WHERE alpaca_order_id = ?;",
                tuple(params),
            )


def default_tracker_db_path() -> str:
    # Allow override for local setups/CI.
    return os.environ.get("AOT_TRACKER_DB", "data/tracker.sqlite3")


