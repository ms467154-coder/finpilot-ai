"""Conversational memory manager (Phase 13).

Orchestrates the chat + memory loop::

    user message
      -> retrieve relevant memories
      -> conversation history + memories (context assembly)
      -> generate_advice() (existing inference)
      -> response
      -> extract important new memories
      -> deduplicate / update / store

Also exposes memory listing and "forget" (delete) for privacy/user control.
Memories are always scoped by ``user_id``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.api import chat_service, store as store_mod
from src.advice import finance_calc, guard, processor, validator
from src.inference import inference
from src.inference.prompt_templates import ADVISOR_SYSTEM_PROMPT

from . import store as memory_store
from .extractor import extract_candidate_memories
from .retriever import load_memory_config, retrieve_relevant_memories, token_similarity
from .schemas import Memory, now_utc


@dataclass
class MemoryChatResult:
    """Outcome of one memory-aware ``POST /chat`` turn."""

    conversation_id: str
    reply: str
    advice: Optional[Any]
    used_memories: List[Memory] = field(default_factory=list)
    extracted_memories: List[Memory] = field(default_factory=list)


def default_user_id() -> str:
    config = load_memory_config()
    return config.get("default_user_id", "anonymous")


def build_memory_context(memories: List[Memory]) -> Optional[str]:
    """Format retrieved memories as a context block for the system prompt."""
    if not memories:
        return None
    lines = [f"- [{memory.category}] {memory.content}" for memory in memories]
    return (
        "Relevant memories about this user that you should use naturally in your "
        "answer (do not ask for this information again):\n" + "\n".join(lines)
    )


def run_chat(
    user_id: Optional[str],
    message: str,
    conversation_id: Optional[str] = None,
) -> MemoryChatResult:
    """Full memory-aware chat turn, mirroring Phase 9 :func:`chat_service.chat`
    persistence + advice behavior while injecting remembered context."""
    config = load_memory_config()
    user_id = user_id or config.get("default_user_id", "anonymous")

    store = store_mod.get_store()
    full_history: List[Dict[str, str]] = []
    conversation = store.get_conversation(conversation_id) if conversation_id else None
    if conversation is not None:
        full_history = conversation["messages"]

    # Short-term context cap (memories live in the system prompt, not the turns).
    max_turns = config.get("max_turns_in_context", 6)
    ctx_history = full_history[-max_turns:] if max_turns else full_history

    memories = retrieve_relevant_memories(
        user_id,
        message,
        top_k=config.get("top_k"),
        threshold=config.get("relevance_threshold"),
    )
    context = build_memory_context(memories)
    system_prompt = f"{ADVISOR_SYSTEM_PROMPT}\n\n{context}" if context else None

    # Verified budget figures: the model must reason from these exact numbers.
    summary = finance_calc.extract_financials(message)
    finance_ctx = finance_calc.finance_context(summary)
    if finance_ctx:
        system_prompt = (system_prompt or ADVISOR_SYSTEM_PROMPT) + "\n\n" + finance_ctx

    # Conversation-level grounding: financial facts from earlier turns plus the
    # follow-up/topic-continuity brief that keeps the model on the current topic.
    prior_user_texts = [t.get("content", "") for t in full_history if t.get("role") == "user"]
    agg_prior = guard.aggregate_financials(prior_user_texts)
    brief = guard.build_conversation_brief(message, ctx_history, agg_prior)
    if brief:
        system_prompt = (system_prompt or ADVISOR_SYSTEM_PROMPT) + "\n\n" + brief

    # Generate -> deterministic correction -> validate. Invalid responses are
    # regenerated with feedback (bounded); if it still fails the user gets an
    # honest clarification instead of the invalid response.
    agg_all = guard.aggregate_financials(prior_user_texts + [message])
    guarded = validator.validate_and_refine(
        message=message,
        history=ctx_history,
        summary=summary,
        agg_summary=agg_all,
        retrieved_texts=[m.content for m in memories],
        base_system_prompt=system_prompt,
        generate=lambda msg, hist, sp: inference.generate_advice(
            msg, conversation_history=hist, system_prompt=sp
        ),
        correct=lambda reply, msg, sm: finance_calc.correct_financial_response(
            reply, msg, summary=sm
        ),
    )
    reply = guarded.reply

    user_turn = {"role": "user", "content": message}
    assistant_turn = {"role": "assistant", "content": reply}
    if conversation is None:
        conversation_id = store.create_conversation(
            [user_turn, assistant_turn], user_id=user_id
        )
    else:
        store.update_conversation(conversation_id, full_history + [user_turn, assistant_turn])

    advice = None
    if not guarded.is_clarification and chat_service.should_extract_advice(reply):
        advice = processor.process_advice(reply, message)
        store.add_advice(advice, conversation_id)

    extracted = extract_and_store(user_id, message, reply, conversation_id)

    return MemoryChatResult(
        conversation_id=conversation_id,
        reply=reply,
        advice=advice,
        used_memories=memories,
        extracted_memories=extracted,
    )


def extract_and_store(
    user_id: str,
    message: str,
    response: str,
    conversation_id: Optional[str],
) -> List[Memory]:
    """Run extraction + dedup + persistence for one completed exchange.

    Near-duplicates are updated in place (e.g. "was saving for a car" ->
    "now saving for a house") instead of stored twice.
    """
    config = load_memory_config()
    candidates = extract_candidate_memories(
        message,
        response,
        default_importance=config.get("default_importance", 0.6),
    )
    dedup_threshold = config.get("dedup_similarity_threshold", 0.5)
    ms = memory_store.get_memory_store()
    existing = ms.list_memories(user_id)
    saved: List[Memory] = []

    for candidate in candidates:
        best_idx = None
        best_sim = 0.0
        for idx, memory in enumerate(existing):
            similarity = token_similarity(candidate.content, memory.content)
            if similarity > best_sim:
                best_sim = similarity
                best_idx = idx

        if best_idx is not None and best_sim >= dedup_threshold:
            memory = existing[best_idx]
            if memory.content == candidate.content:
                continue  # exact duplicate -> skip
            updated = ms.update_memory(
                memory.id,
                user_id,
                content=candidate.content,
                category=candidate.category,
                importance_score=candidate.importance_score,
            )
            if updated is not None:
                saved.append(updated)
                existing[best_idx] = updated
        else:
            created = ms.create_memory(
                Memory(
                    id=uuid.uuid4().hex,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    content=candidate.content,
                    category=candidate.category,
                    importance_score=candidate.importance_score,
                    created_at=now_utc(),
                    updated_at=now_utc(),
                    last_accessed_at=now_utc(),
                )
            )
            saved.append(created)
            existing.append(created)

    return saved


def list_memories(user_id: Optional[str]) -> List[Memory]:
    """List one user's memories (for transparency/debugging)."""
    return memory_store.get_memory_store().list_memories(user_id or default_user_id())


def forget_memory(user_id: Optional[str], memory_id: str) -> bool:
    """Delete a memory, scoped to its owner. Returns False if not found."""
    return memory_store.get_memory_store().delete_memory(
        user_id or default_user_id(), memory_id
    )


__all__ = [
    "MemoryChatResult",
    "build_memory_context",
    "default_user_id",
    "extract_and_store",
    "forget_memory",
    "list_memories",
    "run_chat",
]
