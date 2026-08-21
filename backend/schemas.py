"""Pydantic request/response models for the Financial Advice Chatbot API (Phase 9).

Reuses the Phase 8 :class:`~src.advice.schemas.Advice` model for the core advice
fields and extends it with API-level metadata (``saved`` flag, ``conversation_id``).
Phase 13 adds the optional ``user_id`` / ``used_memories`` fields for
conversational memory (see :mod:`src.memory`).

No financial-profile fields (salary/income/expense/net-worth/risk) appear here.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from src.advice.schemas import Advice as AdviceBase
from src.memory.schemas import Memory


class ChatMessage(BaseModel):
    """A single conversational turn (validates stored history shapes)."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """Body of ``POST /chat``."""

    message: str = Field(..., min_length=1, max_length=4000, description="User's chat message")
    conversation_id: Optional[str] = Field(
        default=None, description="Omit to start a new conversation, or pass to continue one"
    )
    user_id: Optional[str] = Field(
        default=None, description="Owner of conversational memories (defaults to 'anonymous')"
    )


class ChatResponse(BaseModel):
    """Response of ``POST /chat``."""

    conversation_id: str
    reply: str
    advice: Optional[AdviceBase] = Field(
        default=None, description="Structured advice when the reply is actionable, else null"
    )
    used_memories: Optional[List[Memory]] = Field(
        default=None,
        description="Memories injected into this reply (transparency; null when none used)",
    )


class AdviceRecord(AdviceBase):
    """Advice as stored/returned by the API (adds saved + conversation linkage)."""

    saved: bool = False
    conversation_id: Optional[str] = None


class ConversationSummary(BaseModel):
    """One row in a user's conversation list (sidebar)."""

    id: str
    title: str
    created_at: str
    updated_at: str
    last_message: str = ""


class ConversationDetail(BaseModel):
    """A full conversation including its message history."""

    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ChatMessage]


class RenameConversationRequest(BaseModel):
    """Body of ``PATCH /conversations/{id}``."""

    title: str = Field(..., min_length=1, max_length=100, description="New conversation title")

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("title must not be blank")
        return value


class RenameConversationResponse(BaseModel):
    """Response of ``PATCH /conversations/{id}``."""

    id: str
    title: str


class DeleteConversationResponse(BaseModel):
    """Response of ``DELETE /conversations/{id}``."""

    id: str
    deleted: bool


class SaveAdviceResponse(BaseModel):
    """Response of ``POST /advice/{id}/save``."""

    id: str
    saved: bool


class DeleteMemoryResponse(BaseModel):
    """Response of ``DELETE /memory/{id}``."""

    id: str
    deleted: bool


__all__ = [
    "AdviceRecord",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ConversationDetail",
    "ConversationSummary",
    "DeleteConversationResponse",
    "DeleteMemoryResponse",
    "RenameConversationRequest",
    "RenameConversationResponse",
    "SaveAdviceResponse",
]
