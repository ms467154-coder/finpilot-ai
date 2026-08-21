"""HTTP route handlers for the Financial Advice Chatbot API (Phase 9).

Routes:

* ``POST /chat`` -- reply + optional structured advice (multi-turn capable).
* ``GET  /advice`` -- stored advice history (optional ``?category=`` filter).
* ``GET  /advice/{id}`` -- a single advice item.
* ``POST /advice/{id}/save`` -- mark an advice item saved for the dashboard.
* ``GET  /memory`` -- a user's conversational memories (optional ``?user_id=``).
* ``DELETE /memory/{id}`` -- forget a specific memory.

Handlers are thin wrappers over the service layer; no business logic lives
here. No endpoint accepts or stores numeric financial-profile data.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api import chat_service
from src.memory import manager as memory_manager
from src.memory.schemas import Memory

from .schemas import (
    AdviceRecord,
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    DeleteConversationResponse,
    DeleteMemoryResponse,
    RenameConversationRequest,
    RenameConversationResponse,
    SaveAdviceResponse,
)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Send a chat message (multi-turn)")
def chat(req: ChatRequest) -> ChatResponse:
    result = memory_manager.run_chat(
        user_id=req.user_id,
        message=req.message,
        conversation_id=req.conversation_id,
    )
    return ChatResponse(
        conversation_id=result.conversation_id,
        reply=result.reply,
        advice=result.advice,
        used_memories=result.used_memories or None,
    )


@router.get("/advice", response_model=List[AdviceRecord], summary="List stored advice history")
def list_advice(category: Optional[str] = Query(default=None)) -> List[AdviceRecord]:
    return [AdviceRecord(**row) for row in chat_service.list_advice(category=category)]


@router.get("/advice/{advice_id}", response_model=AdviceRecord, summary="Get one advice item")
def get_advice(advice_id: str) -> AdviceRecord:
    row = chat_service.get_advice(advice_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Advice {advice_id!r} not found")
    return AdviceRecord(**row)


@router.post("/advice/{advice_id}/save", response_model=SaveAdviceResponse, summary="Mark advice saved")
def save_advice(advice_id: str) -> SaveAdviceResponse:
    if not chat_service.save_advice(advice_id):
        raise HTTPException(status_code=404, detail=f"Advice {advice_id!r} not found")
    return SaveAdviceResponse(id=advice_id, saved=True)


@router.get(
    "/conversations",
    response_model=List[ConversationSummary],
    summary="List a user's conversations (newest first)",
)
def list_conversations(user_id: Optional[str] = Query(default=None)) -> List[ConversationSummary]:
    owner = user_id or memory_manager.default_user_id()
    return [ConversationSummary(**row) for row in chat_service.list_conversations(owner)]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get one conversation's full message history",
)
def get_conversation(conversation_id: str) -> ConversationDetail:
    row = chat_service.get_conversation(conversation_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id!r} not found")
    return ConversationDetail(**row)


@router.patch(
    "/conversations/{conversation_id}",
    response_model=RenameConversationResponse,
    summary="Rename a conversation",
)
def rename_conversation(
    conversation_id: str, req: RenameConversationRequest
) -> RenameConversationResponse:
    if not chat_service.rename_conversation(conversation_id, req.title):
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id!r} not found")
    return RenameConversationResponse(id=conversation_id, title=req.title)


@router.delete(
    "/conversations/{conversation_id}",
    response_model=DeleteConversationResponse,
    summary="Delete a conversation",
)
def delete_conversation(
    conversation_id: str, user_id: Optional[str] = Query(default=None)
) -> DeleteConversationResponse:
    owner = user_id or memory_manager.default_user_id()
    if not chat_service.delete_conversation(conversation_id, owner):
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id!r} not found")
    return DeleteConversationResponse(id=conversation_id, deleted=True)


@router.get("/memory", response_model=List[Memory], summary="List a user's memories")
def list_memories(user_id: Optional[str] = Query(default=None)) -> List[Memory]:
    return memory_manager.list_memories(user_id)


@router.delete(
    "/memory/{memory_id}",
    response_model=DeleteMemoryResponse,
    summary="Forget a memory",
)
def delete_memory(
    memory_id: str, user_id: Optional[str] = Query(default=None)
) -> DeleteMemoryResponse:
    if not memory_manager.forget_memory(user_id, memory_id):
        raise HTTPException(status_code=404, detail=f"Memory {memory_id!r} not found")
    return DeleteMemoryResponse(id=memory_id, deleted=True)


__all__ = ["router"]
