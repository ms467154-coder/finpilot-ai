"""Conversational memory data model (Phase 13).

A memory is a durable, free-form fact/preference/goal about a user that the
chatbot can recall across conversations. Fields are deliberately
general-purpose:

* ``category`` is a free-form string (e.g. "preference", "goal", "context",
  "fact") inferred from the user's words - it is **not** a structured
  financial-profile field (no salary/income/expense/net-worth/risk).
* Numeric financial figures are never required or auto-generated; if a user
  happens to mention one it lives only as free-text ``content``, and obvious raw
  identifiers (e.g. account/card numbers) are redacted on write.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator

CATEGORY_PREFERENCE = "preference"
CATEGORY_GOAL = "goal"
CATEGORY_CONTEXT = "context"
CATEGORY_FACT = "fact"
CATEGORY_GENERAL = "general"

# Privacy: redact obvious raw identifiers (card/account/SSN-like numbers) on write.
_SENSITIVE_PATTERNS = (
    re.compile(r"\b(?:\d[ -]?){4,}\d\b"),      # 5+ digit runs (card/account/phone-like)
    re.compile(r"\b\d{9}\b"),                  # 9-digit SSN/account-like
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),      # formatted SSN
    re.compile(
        r"\b(?:account|acct|routing)\s*(?:number|no|#)?\s*[:=\-]?\s*[\w-]{4,}\b",
        re.IGNORECASE,
    ),
)

_REDACTED = "[redacted-number]"


def redact_sensitive(text: str) -> str:
    """Replace obvious raw identifiers with a redaction marker."""
    if not text:
        return text
    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Memory(BaseModel):
    """A single persistent conversational memory."""

    id: str = Field(..., min_length=1, description="Unique memory identifier")
    user_id: str = Field(..., min_length=1, description="Owner of the memory")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation the memory came from"
    )
    content: str = Field(..., min_length=1, description="Free-form durable fact/preference/goal")
    category: str = Field(
        default=CATEGORY_GENERAL, min_length=1, description="Free-form inferred category"
    )
    importance_score: float = Field(default=0.6, ge=0.0, le=1.0, description="0-1 salience")
    created_at: datetime = Field(..., description="UTC creation timestamp")
    updated_at: datetime = Field(..., description="UTC last update timestamp")
    last_accessed_at: datetime = Field(..., description="UTC last time the memory was used")

    @field_validator("content", mode="before")
    @classmethod
    def _redact_content(cls, value):
        return redact_sensitive(value if isinstance(value, str) else "")

    @field_validator("importance_score", mode="after")
    @classmethod
    def _clamp_importance(cls, value):
        return max(0.0, min(1.0, float(value)))

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


__all__ = [
    "CATEGORY_CONTEXT",
    "CATEGORY_FACT",
    "CATEGORY_GENERAL",
    "CATEGORY_GOAL",
    "CATEGORY_PREFERENCE",
    "Memory",
    "now_utc",
    "redact_sensitive",
]
