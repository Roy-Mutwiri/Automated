"""Durable state and the trace store.

Answers the observability requirement: for any utterance, what market state did
the host see, what comment triggered it, what context went in, which model
produced it, how long generation and TTS took, and was the data fresh.

SQLite by default -- stdlib, zero configuration, and genuinely adequate for
seven sessions. The schema is deliberately Postgres-compatible so moving over
is a connection-string change rather than a migration.

Nothing in the speech path reads this store synchronously. Writes are queued
and flushed on a background task, so a slow or missing database degrades
observability and never the broadcast.

Viewer privacy: author names are hashed before they reach a CommentEvent, and
nothing here stores raw handles or message text beyond what the host actually
responded to.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from shared.contracts import AIResponse, ServiceHealth

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS utterances (
    utterance_id   TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    trace_id       TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    text           TEXT NOT NULL,
    trigger_type   TEXT NOT NULL,
    priority       INTEGER NOT NULL,
    model          TEXT,
    market_state_id TEXT,
    market_confidence TEXT,
    first_token_ms INTEGER,
    generation_ms  INTEGER,
    prompt_tokens  INTEGER,
    cache_read_tokens INTEGER,
    stated_price   INTEGER NOT NULL DEFAULT 0,
    safety_passed  INTEGER NOT NULL DEFAULT 1,
    provenance     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_utt_session_time ON utterances(session_id, created_at);
CREATE INDEX IF NOT EXISTS ix_utt_trace ON utterances(trace_id);

CREATE TABLE IF NOT EXISTS blocked (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    at            TEXT NOT NULL,
    reason        TEXT NOT NULL,
    detail        TEXT,
    text          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_blocked_session ON blocked(session_id, at);

CREATE TABLE IF NOT EXISTS health (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    at            TEXT NOT NULL,
    component     TEXT NOT NULL,
    session_id    TEXT,
    state         TEXT NOT NULL,
    detail        TEXT
);
CREATE INDEX IF NOT EXISTS ix_health_time ON health(at);

CREATE TABLE IF NOT EXISTS session_state (
    session_id    TEXT PRIMARY KEY,
    updated_at    TEXT NOT NULL,
    payload       TEXT NOT NULL
);
"""


class TraceStore:
    def __init__(self, path: str | Path = "data/gold-live.db", flush_every_s: float = 2.0) -> None:
        self.path = Path(path)
        self.flush_every_s = flush_every_s
        self._queue: asyncio.Queue[tuple[str, tuple]] = asyncio.Queue(maxsize=2000)
        self._task: asyncio.Task | None = None
        self._running = False
        self.dropped = 0
        self.written = 0

    # -- lifecycle --------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")  # concurrent readers for the dashboard
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def migrate(self) -> None:
        with closing(self.connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    async def start(self) -> None:
        self.migrate()
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            await self._drain_once()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # -- writes -----------------------------------------------------------

    def _enqueue(self, sql: str, params: tuple) -> None:
        try:
            self._queue.put_nowait((sql, params))
        except asyncio.QueueFull:
            # Observability is not worth stalling the broadcast for.
            self.dropped += 1
            if self.dropped % 100 == 1:
                log.warning("trace queue full; dropped %d records", self.dropped)

    def record_utterance(self, r: AIResponse) -> None:
        p = r.provenance
        self._enqueue(
            """INSERT OR REPLACE INTO utterances
               (utterance_id, session_id, trace_id, created_at, text, trigger_type,
                priority, model, market_state_id, market_confidence, first_token_ms,
                generation_ms, prompt_tokens, cache_read_tokens, stated_price,
                safety_passed, provenance)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(r.utterance_id), r.session_id, r.trace_id,
                r.created_at.isoformat(), r.text, r.trigger.type.value,
                int(r.trigger.priority), p.model,
                str(p.market_state_id) if p.market_state_id else None,
                p.market_confidence.value if p.market_confidence else None,
                p.first_token_ms, p.generation_ms, p.prompt_tokens,
                p.cache_read_tokens, int(r.safety.stated_price),
                int(r.safety.passed), p.model_dump_json(),
            ),
        )

    def record_blocked(
        self, session_id: str, reason: str, text: str, detail: str | None = None
    ) -> None:
        self._enqueue(
            "INSERT INTO blocked (session_id, at, reason, detail, text) VALUES (?,?,?,?,?)",
            (session_id, datetime.now(timezone.utc).isoformat(), reason, detail, text[:1000]),
        )

    def record_health(self, h: ServiceHealth) -> None:
        self._enqueue(
            "INSERT INTO health (at, component, session_id, state, detail) VALUES (?,?,?,?,?)",
            (
                h.last_heartbeat.isoformat(), h.component, h.session_id,
                h.state.value, h.degraded_reason,
            ),
        )

    def save_session_state(self, session_id: str, payload: dict[str, Any]) -> None:
        self._enqueue(
            """INSERT OR REPLACE INTO session_state (session_id, updated_at, payload)
               VALUES (?,?,?)""",
            (session_id, datetime.now(timezone.utc).isoformat(), json.dumps(payload)),
        )

    # -- flushing ---------------------------------------------------------

    async def _drain_once(self) -> int:
        if self._queue.empty():
            return 0
        batch: list[tuple[str, tuple]] = []
        while not self._queue.empty() and len(batch) < 500:
            batch.append(self._queue.get_nowait())
        try:
            with closing(self.connect()) as conn:
                for sql, params in batch:
                    conn.execute(sql, params)
                conn.commit()
            self.written += len(batch)
        except sqlite3.Error as exc:
            log.warning("trace flush failed, discarding %d records: %s", len(batch), exc)
        return len(batch)

    async def _flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.flush_every_s)
            try:
                await self._drain_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("trace flush loop error")

    # -- reads (dashboard) ------------------------------------------------

    def recent_utterances(self, session_id: str | None = None, limit: int = 50) -> list[dict]:
        sql = (
            "SELECT utterance_id, session_id, trace_id, created_at, text, trigger_type,"
            " model, market_confidence, first_token_ms, generation_ms"
            " FROM utterances"
        )
        params: tuple = ()
        if session_id:
            sql += " WHERE session_id = ?"
            params = (session_id,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params += (limit,)

        with closing(self.connect()) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(sql, params)]

    def explain(self, trace_id: str) -> dict | None:
        """Everything known about one utterance: why did it say that?"""
        with closing(self.connect()) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM utterances WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if row is None:
                return None
            out = dict(row)
            out["provenance"] = json.loads(out["provenance"])
            return out

    def stats(self, session_id: str, since_minutes: int = 60) -> dict:
        since = (
            datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
        ).isoformat()
        with closing(self.connect()) as conn:
            utterances = conn.execute(
                "SELECT COUNT(*) FROM utterances WHERE session_id=? AND created_at>=?",
                (session_id, since),
            ).fetchone()[0]
            blocked = conn.execute(
                "SELECT COUNT(*) FROM blocked WHERE session_id=? AND at>=?",
                (session_id, since),
            ).fetchone()[0]
            latency = conn.execute(
                "SELECT AVG(first_token_ms) FROM utterances "
                "WHERE session_id=? AND created_at>=? AND first_token_ms IS NOT NULL",
                (session_id, since),
            ).fetchone()[0]
        return {
            "session_id": session_id,
            "window_minutes": since_minutes,
            "utterances": utterances,
            "blocked": blocked,
            "avg_first_token_ms": round(latency) if latency else None,
        }
