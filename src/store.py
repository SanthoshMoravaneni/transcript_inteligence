# src/store.py
# ─────────────────────────────────────────────────────────────────────────────
# SQLite-backed storage for all processed transcript data.
#
# WHY SQLITE INSTEAD OF A JSON FILE?
# ────────────────────────────────────
# The old approach saved everything to a single JSON file. That breaks the
# moment a new transcript arrives — you'd need to reload and rewrite the
# entire file. SQLite fixes all of this:
#
#   - New meeting = one INSERT, not a full file rewrite
#   - Concurrent reads are safe (MCP server + watcher run simultaneously)
#   - Queryable: filter by call type, date, score without loading everything
#   - The database file persists between restarts — no re-processing needed
#
# WHY NOT POSTGRES?
# ─────────────────
# SQLite is the right choice at this scale. No server to run, no credentials
# to manage, ships with Python. Production upgrade path is simple: swap the
# connection string for Postgres — the query logic stays identical.
#
# ALL WRITES ARE IDEMPOTENT.
# Running the ingestion twice on the same folder produces exactly one row.
# The folder content hash (MD5 of the three JSON files) ensures we only
# reprocess a meeting if its source files actually changed.
# ─────────────────────────────────────────────────────────────────────────────

import json
import sqlite3
import threading
import hashlib
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from .classifier import extract_company


class TranscriptStore:
    """Persistent storage layer for processed transcript data."""

    def __init__(self, db_path: str = "transcripts.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_schema()

    # ── SCHEMA ────────────────────────────────────────────────────────────────

    def _create_schema(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS meetings (
                    meeting_id          TEXT PRIMARY KEY,
                    title               TEXT,
                    call_type           TEXT,
                    topic_cluster       TEXT,
                    sentiment_label     TEXT,
                    sentiment_score     REAL,
                    duration            REAL,
                    word_count          INTEGER,
                    summary_text        TEXT,
                    topics_json         TEXT,
                    emails_json         TEXT,
                    competitors_json    TEXT,
                    speaker_stats_json  TEXT,
                    action_items_json   TEXT,
                    processed_at        TEXT,
                    folder_hash         TEXT
                );

                CREATE TABLE IF NOT EXISTS key_moments (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id   TEXT,
                    moment_type  TEXT,
                    moment_text  TEXT,
                    speaker      TEXT,
                    timestamp    REAL,
                    FOREIGN KEY(meeting_id) REFERENCES meetings(meeting_id)
                );

                CREATE TABLE IF NOT EXISTS action_items (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id  TEXT,
                    owner       TEXT,
                    item_text   TEXT,
                    FOREIGN KEY(meeting_id) REFERENCES meetings(meeting_id)
                );

                CREATE INDEX IF NOT EXISTS idx_call_type  ON meetings(call_type);
                CREATE INDEX IF NOT EXISTS idx_topic      ON meetings(topic_cluster);
                CREATE INDEX IF NOT EXISTS idx_score      ON meetings(sentiment_score);
                CREATE INDEX IF NOT EXISTS idx_km_type    ON key_moments(moment_type);
                CREATE INDEX IF NOT EXISTS idx_km_meeting ON key_moments(meeting_id);
                CREATE INDEX IF NOT EXISTS idx_ai_owner   ON action_items(owner);
            """)
            self._conn.commit()

    # ── WRITE ─────────────────────────────────────────────────────────────────

    @staticmethod
    def folder_hash(folder: Path) -> str:
        """MD5 of the three source JSON files. Changes only if content changes."""
        h = hashlib.md5()
        for fname in ["meeting-info.json", "summary.json", "transcript.json"]:
            p = folder / fname
            if p.exists():
                h.update(p.read_bytes())
        return h.hexdigest()

    def is_processed(self, meeting_id: str, folder_hash: str) -> bool:
        """True if this exact folder content has already been stored."""
        with self._lock:
            row = self._conn.execute(
                "SELECT folder_hash FROM meetings WHERE meeting_id = ?",
                (meeting_id,)
            ).fetchone()
        return row is not None and row[0] == folder_hash

    def upsert(self, meeting: dict):
        """Insert or replace a meeting and its related rows."""
        mid = meeting["id"]
        with self._lock:
            c = self._conn.cursor()
            c.execute("DELETE FROM key_moments  WHERE meeting_id = ?", (mid,))
            c.execute("DELETE FROM action_items WHERE meeting_id = ?", (mid,))
            c.execute(
                "INSERT OR REPLACE INTO meetings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    meeting["title"],
                    meeting["call_type"],
                    meeting["topic_cluster"],
                    meeting["sentiment"],
                    meeting["score"],
                    meeting["duration"],
                    meeting["word_count"],
                    meeting["summary_text"],
                    json.dumps(meeting["topics"]),
                    json.dumps(meeting["emails"]),
                    json.dumps(meeting.get("competitors", [])),
                    json.dumps(meeting["speaker_stats"]),
                    json.dumps(meeting["action_items"]),
                    datetime.utcnow().isoformat(),
                    meeting.get("folder_hash", ""),
                ),
            )
            for km in meeting.get("key_moments", []):
                c.execute(
                    "INSERT INTO key_moments "
                    "(meeting_id, moment_type, moment_text, speaker, timestamp) "
                    "VALUES (?,?,?,?,?)",
                    (mid, km.get("type",""), km.get("text",""),
                     km.get("speaker",""), km.get("time", 0)),
                )
            for ai in meeting.get("action_items", []):
                owner, task = (ai.split(":", 1) + [""])[:2] if ":" in ai else ("Unknown", ai)
                c.execute(
                    "INSERT INTO action_items (meeting_id, owner, item_text) VALUES (?,?,?)",
                    (mid, owner.strip(), task.strip()),
                )
            self._conn.commit()

    # ── READ ──────────────────────────────────────────────────────────────────

    def get_meetings(
        self,
        call_type: str = None,
        topic_cluster: str = None,
        min_score: float = None,
        max_score: float = None,
    ) -> list[dict]:
        """Fetch meetings with optional filters. Always live from DB."""
        q = "SELECT * FROM meetings WHERE 1=1"
        p = []
        if call_type:        q += " AND call_type = ?";        p.append(call_type)
        if topic_cluster:    q += " AND topic_cluster = ?";    p.append(topic_cluster)
        if min_score is not None: q += " AND sentiment_score >= ?"; p.append(min_score)
        if max_score is not None: q += " AND sentiment_score <= ?"; p.append(max_score)
        q += " ORDER BY sentiment_score ASC"

        with self._lock:
            c = self._conn.execute(q, p)
            cols = [d[0] for d in c.description]
            rows = c.fetchall()

        result = []
        for row in rows:
            m = dict(zip(cols, row))
            for col in ["topics_json","emails_json","competitors_json",
                        "speaker_stats_json","action_items_json"]:
                try:    m[col] = json.loads(m[col] or "[]")
                except: m[col] = []
            result.append(m)
        return result

    def get_key_moments(
        self,
        moment_type: str = None,
        meeting_id: str = None,
        limit: int = 500,
    ) -> list[dict]:
        q = """
            SELECT km.*, m.title, m.call_type, m.sentiment_score, m.topic_cluster
            FROM key_moments km
            JOIN meetings m ON km.meeting_id = m.meeting_id
            WHERE 1=1
        """
        p = []
        if moment_type: q += " AND km.moment_type = ?"; p.append(moment_type)
        if meeting_id:  q += " AND km.meeting_id = ?";  p.append(meeting_id)
        q += " ORDER BY m.sentiment_score ASC LIMIT ?"; p.append(limit)

        with self._lock:
            c = self._conn.execute(q, p)
            cols = [d[0] for d in c.description]
            return [dict(zip(cols, r)) for r in c.fetchall()]

    def get_action_owners(self, limit: int = 20) -> list[dict]:
        with self._lock:
            c = self._conn.execute("""
                SELECT owner,
                       COUNT(*) as total_items,
                       COUNT(DISTINCT meeting_id) as across_calls
                FROM action_items
                GROUP BY owner ORDER BY total_items DESC LIMIT ?
            """, (limit,))
            cols = [d[0] for d in c.description]
            return [dict(zip(cols, r)) for r in c.fetchall()]

    def search(self, query: str, limit: int = 10) -> list[dict]:
        q = query.lower()
        with self._lock:
            c = self._conn.execute("""
                SELECT meeting_id, title, call_type, topic_cluster,
                       sentiment_score, summary_text
                FROM meetings
                WHERE LOWER(title) LIKE ? OR LOWER(summary_text) LIKE ?
                   OR LOWER(topics_json) LIKE ?
                ORDER BY sentiment_score ASC LIMIT ?
            """, (f"%{q}%", f"%{q}%", f"%{q}%", limit))
            cols = [d[0] for d in c.description]
            return [dict(zip(cols, r)) for r in c.fetchall()]

    def get_stats(self) -> dict:
        """Live counts — used by every MCP summary tool."""
        with self._lock:
            def one(sql, *p):
                return self._conn.execute(sql, p).fetchone()[0]
            def many(sql, *p):
                return dict(self._conn.execute(sql, p).fetchall())

            return {
                "total_meetings": one("SELECT COUNT(*) FROM meetings"),
                "call_types":     many("SELECT call_type, COUNT(*) FROM meetings GROUP BY call_type"),
                "topic_clusters": many("SELECT topic_cluster, COUNT(*) FROM meetings GROUP BY topic_cluster ORDER BY COUNT(*) DESC"),
                "sentiment": {
                    "support_avg":  round(one("SELECT AVG(sentiment_score) FROM meetings WHERE call_type='support'") or 0, 2),
                    "external_avg": round(one("SELECT AVG(sentiment_score) FROM meetings WHERE call_type='external'") or 0, 2),
                    "internal_avg": round(one("SELECT AVG(sentiment_score) FROM meetings WHERE call_type='internal'") or 0, 2),
                },
                "signals": {
                    "churn_signals": one("SELECT COUNT(*) FROM key_moments WHERE moment_type='churn_signal'"),
                    "feature_gaps":  one("SELECT COUNT(*) FROM key_moments WHERE moment_type='feature_gap'"),
                    "action_items":  one("SELECT COUNT(*) FROM action_items"),
                },
            }

    def compute_churn_risk(self) -> list[dict]:
        """
        Recompute churn risk scores live from DB every time this is called.

        Formula:
          (5 - avg_sentiment)           low sentiment = high risk
          + (support_ratio × 2)         mostly reactive interactions = worse
          + (churn_signals × 0.7)       explicit "I might leave" moments
          + (competitor_mentions × 1.2) active competitive evaluation

        Competitor mentions get the highest per-unit weight because
        when a customer names a competitor, they've already started
        the evaluation process. It's not venting — it's a signal.
        """
        meetings = self.get_meetings()
        churn_kms = self.get_key_moments(moment_type="churn_signal", limit=2000)

        churn_by_company: dict[str, int] = defaultdict(int)
        for km in churn_kms:
            company = extract_company(km["title"])
            if company:
                churn_by_company[company] += 1

        account: dict[str, dict] = defaultdict(lambda: {
            "calls": [], "competitor_mentions": 0, "competitors_named": set()
        })

        for m in meetings:
            company = extract_company(m["title"])
            if not company:
                continue
            account[company]["calls"].append({
                "type": m["call_type"], "score": m["sentiment_score"], "title": m["title"]
            })
            comps = m["competitors_json"] if isinstance(m["competitors_json"], list) else []
            account[company]["competitor_mentions"] += len(comps)
            account[company]["competitors_named"].update(comps)

        results = []
        for company, r in account.items():
            if not r["calls"]:
                continue
            avg_s  = sum(c["score"] for c in r["calls"]) / len(r["calls"])
            sup_r  = sum(1 for c in r["calls"] if c["type"] == "support") / len(r["calls"])
            risk   = (5 - avg_s) + (sup_r * 2) + (churn_by_company.get(company, 0) * 0.7) + (r["competitor_mentions"] * 1.2)
            results.append({
                "company":             company,
                "risk_score":          round(risk, 2),
                "avg_sentiment":       round(avg_s, 2),
                "support_ratio":       round(sup_r, 2),
                "churn_signals":       churn_by_company.get(company, 0),
                "competitor_mentions": r["competitor_mentions"],
                "competitors_named":   list(r["competitors_named"]),
                "total_calls":         len(r["calls"]),
                "support_calls":       sum(1 for c in r["calls"] if c["type"] == "support"),
            })

        return sorted(results, key=lambda x: -x["risk_score"])
