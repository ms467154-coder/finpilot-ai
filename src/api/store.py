"""Persistence layer for the Financial Advice Chatbot (Phase 9).

A lightweight SQLite-backed store for:

* **conversations** (``id``, timestamps, JSON message history) - powers multi-turn
  chat context;
* **advice** items (structured :class:`~src.advice.schemas.Advice` records plus a
  ``saved`` flag for the dashboard's Saved Advice section).

The schema deliberately contains **no financial-profile fields** (no salary /
income / expense / net-worth / risk data anywhere).

Thread-safe (a single module-level lock serializes SQLite access for FastAPI's
threadpool). The database file lives at ``data/chatbot.db`` by default and can be
overridden with the ``CHATBOT_DB_PATH`` environment variable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DB_PATH = os.environ.get("CHATBOT_DB_PATH") or str(_ROOT / "data" / "chatbot.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    messages   TEXT NOT NULL,
    user_id    TEXT NOT NULL DEFAULT 'anonymous',
    title      TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS advice (
    id                TEXT PRIMARY KEY,
    conversation_id   TEXT,
    timestamp         TEXT NOT NULL,
    category          TEXT NOT NULL,
    short_title       TEXT NOT NULL,
    key_recommendation TEXT NOT NULL,
    full_text         TEXT NOT NULL,
    source_question   TEXT NOT NULL,
    saved             INTEGER NOT NULL DEFAULT 0
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def derive_title(messages: Sequence[Dict[str, str]], limit: int = 48) -> str:
    """A human-friendly conversation title: the first user message, truncated."""
    for message in messages:
        if message.get("role") == "user":
            text = str(message.get("content") or "").strip().replace("\n", " ")
            if text:
                return text[:limit] + ("…" if len(text) > limit else "")
            break
    return "New conversation"


class SQLiteStore:
    """Minimal thread-safe SQLite store for conversations and advice."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.init_db()

    def init_db(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._migrate_conversations()
            self._conn.commit()

    def _migrate_conversations(self) -> None:
        """Add columns introduced after the first release to existing tables."""
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(conversations)")
        }
        if "user_id" not in columns:
            self._conn.execute(
                "ALTER TABLE conversations ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous'"
            )
        if "title" not in columns:
            self._conn.execute(
                "ALTER TABLE conversations ADD COLUMN title TEXT NOT NULL DEFAULT ''"
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- conversations -------------------------------------------------------
    def create_conversation(
        self,
        messages: Sequence[Dict[str, str]],
        user_id: str = "anonymous",
        title: Optional[str] = None,
    ) -> str:
        conversation_id = uuid.uuid4().hex
        stamp = now_iso()
        payload = json.dumps(list(messages), ensure_ascii=False)
        if title is None:
            title = derive_title(messages)
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversations (id, created_at, updated_at, messages, user_id, title) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (conversation_id, stamp, stamp, payload, user_id, title),
            )
            self._conn.commit()
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "user_id": row["user_id"],
            "title": row["title"],
            "messages": json.loads(row["messages"]),
        }

    def update_conversation(self, conversation_id: str, messages: Sequence[Dict[str, str]]) -> bool:
        payload = json.dumps(list(messages), ensure_ascii=False)
        with self._lock:
            cur = self._conn.execute(
                "UPDATE conversations SET messages = ?, updated_at = ? WHERE id = ?",
                (payload, now_iso(), conversation_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id)
            )
            self._conn.commit()
        return cur.rowcount > 0

    def delete_conversation(self, conversation_id: str, user_id: str = "anonymous") -> bool:
        """Delete a conversation (scoped to its owner) plus its advice records."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            if cur.rowcount:
                self._conn.execute(
                    "DELETE FROM advice WHERE conversation_id = ?", (conversation_id,)
                )
            self._conn.commit()
        return cur.rowcount > 0

    def list_conversations(self, user_id: str = "anonymous") -> List[Dict[str, Any]]:
        """One user's conversation summaries (newest first) for the sidebar."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            messages = json.loads(row["messages"])
            result.append(
                {
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "last_message": messages[-1]["content"] if messages else "",
                }
            )
        return result

    # -- advice --------------------------------------------------------------
    def add_advice(self, advice, conversation_id: Optional[str]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO advice (id, conversation_id, timestamp, category, short_title, "
                "key_recommendation, full_text, source_question, saved) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    advice.id,
                    conversation_id,
                    advice.timestamp.isoformat(),
                    advice.category.value,
                    advice.short_title,
                    advice.key_recommendation,
                    advice.full_text,
                    advice.source_question,
                ),
            )
            self._conn.commit()

    def list_advice(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM advice"
        params: tuple = ()
        if category:
            sql += " WHERE lower(category) = lower(?)"
            params = (category,)
        sql += " ORDER BY timestamp DESC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_advice(r) for r in rows]

    def get_advice(self, advice_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM advice WHERE id = ?", (advice_id,)
            ).fetchone()
        return self._row_to_advice(row) if row else None

    def set_saved(self, advice_id: str, saved: bool = True) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE advice SET saved = ? WHERE id = ?", (1 if saved else 0, advice_id)
            )
            self._conn.commit()
        return cur.rowcount > 0

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _row_to_advice(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "timestamp": row["timestamp"],
            "category": row["category"],
            "short_title": row["short_title"],
            "key_recommendation": row["key_recommendation"],
            "full_text": row["full_text"],
            "source_question": row["source_question"],
            "saved": bool(row["saved"]),
        }


_store: Optional[SQLiteStore] = None


def get_store(db_path: Optional[str] = None) -> SQLiteStore:
    """Return the module-level store singleton (initialized once)."""
    global _store
    if _store is None:
        _store = SQLiteStore(db_path or DEFAULT_DB_PATH)
    return _store


def reset_store() -> None:
    """Drop the singleton (used by tests)."""
    global _store
    if _store is not None:
        _store.close()
        _store = None


__all__ = ["DEFAULT_DB_PATH", "SQLiteStore", "get_store", "reset_store"]