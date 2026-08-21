"""SQLite persistence for conversational memories (Phase 13).

A lightweight, thread-safe SQLite store mirroring the Phase 9 pattern in
:mod:`src.api.store`. Memories are keyed by ``user_id``; every update/delete is
scoped to the owning user so memories can never leak across users.

The schema deliberately has **no financial-profile fields**; ``content`` is
free-form text and ``category`` is a free-form string.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .schemas import Memory

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = os.environ.get("CHATBOT_DB_PATH") or str(_ROOT / "data" / "chatbot.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    conversation_id  TEXT,
    content          TEXT NOT NULL,
    category         TEXT NOT NULL,
    importance_score REAL NOT NULL DEFAULT 0.6,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SQLiteMemoryStore:
    """Thread-safe SQLite persistence for :class:`Memory` records."""

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
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- create -----------------------------------------------------------
    def create_memory(self, memory: Memory) -> Memory:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (id, user_id, conversation_id, content, category, "
                "importance_score, created_at, updated_at, last_accessed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory.id,
                    memory.user_id,
                    memory.conversation_id,
                    memory.content,
                    memory.category,
                    memory.importance_score,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    memory.last_accessed_at.isoformat(),
                ),
            )
            self._conn.commit()
        return memory

    # -- read -------------------------------------------------------------
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def list_memories(self, user_id: str) -> List[Memory]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    # -- update -----------------------------------------------------------
    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        *,
        content: Optional[str] = None,
        category: Optional[str] = None,
        importance_score: Optional[float] = None,
    ) -> Optional[Memory]:
        """Update a memory only if it belongs to ``user_id``."""
        sets, params = [], []
        if content is not None:
            sets.append("content = ?")
            params.append(content)
        if category is not None:
            sets.append("category = ?")
            params.append(category)
        if importance_score is not None:
            sets.append("importance_score = ?")
            params.append(float(importance_score))
        if not sets:
            memory = self.get_memory(memory_id)
            return memory if memory and memory.user_id == user_id else None
        sets.append("updated_at = ?")
        params.append(now_iso())
        params += [memory_id, user_id]
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ? AND user_id = ?",
                params,
            )
            self._conn.commit()
        return self.get_memory(memory_id) if cur.rowcount > 0 else None

    def touch_memory(self, memory_id: str) -> bool:
        """Update ``last_accessed_at`` (call when a memory is used)."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE memories SET last_accessed_at = ? WHERE id = ?",
                (now_iso(), memory_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    # -- delete -----------------------------------------------------------
    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Delete a memory only if it belongs to ``user_id``. Returns found."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            user_id=row["user_id"],
            conversation_id=row["conversation_id"],
            content=row["content"],
            category=row["category"],
            importance_score=row["importance_score"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]),
        )


_store: Optional[SQLiteMemoryStore] = None


def get_memory_store(db_path: Optional[str] = None) -> SQLiteMemoryStore:
    global _store
    if _store is None:
        _store = SQLiteMemoryStore(db_path or DEFAULT_DB_PATH)
    return _store


def reset_memory_store() -> None:
    global _store
    if _store is not None:
        _store.close()
        _store = None


__all__ = [
    "DEFAULT_DB_PATH",
    "SQLiteMemoryStore",
    "get_memory_store",
    "reset_memory_store",
]
