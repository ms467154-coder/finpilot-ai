"""Service-layer glue between the FastAPI backend and the ML/processing stack.

Wires together (Phase 7) :func:`src.inference.inference.generate_advice`, (Phase
8) :func:`src.advice.processor.process_advice`, and (Phase 9)
:mod:`src.api.store` persistence::

    ChatResult = chat(message, conversation_id=None)   # full chat turn
    list_advice(category=None) / get_advice(id) / save_advice(id)

No endpoint or service function here accepts or stores numeric financial-profile
data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.advice import finance_calc, guard, processor, validator
from src.advice.schemas import Advice
from src.api import store as store_mod
from src.inference import inference
from src.inference.prompt_templates import ADVISOR_SYSTEM_PROMPT

# Advice markers that make even a short reply worth structuring as an Advice item.
_ADVICE_MARKERS = (
    "pay off", "pay down", "save", "invest", "start", "should", "make sure",
    "consider", "reduce", "create", "build", "contribute", "set up", "avoid",
    "track", "cut back", "rebalance", "choose", "keep", "add", "review",
)

DEFAULT_MIN_WORDS = 12
DEFAULT_SHORT_MIN_WORDS = 4


@dataclass
class ChatResult:
    """Outcome of one ``POST /chat`` turn."""

    conversation_id: str
    reply: str
    advice: Optional[Advice]


# ---------------------------------------------------------------------------
# Advice-extraction gate
# ---------------------------------------------------------------------------
def should_extract_advice(
    reply: str,
    min_words: int = DEFAULT_MIN_WORDS,
    short_min_words: int = DEFAULT_SHORT_MIN_WORDS,
) -> bool:
    """Decide whether a raw reply is "actionable advice" worth structuring.

    Long substantive replies always qualify; short replies qualify only when
    they contain an explicit advice marker (e.g. "pay off debt first").
    """
    words = len(reply.split())
    if words < short_min_words:
        return False
    if words >= min_words:
        return True
    lowered = reply.lower()
    return any(marker in lowered for marker in _ADVICE_MARKERS)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def chat(
    message: str,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> ChatResult:
    """Run one chat turn: generate a reply, persist turns, extract advice.

    * Uses the conversation's stored history (when ``conversation_id`` is given)
      so the model behaves like a chat.
    * Always persists the user/assistant turns.
    * Persists a structured :class:`Advice` only when the reply is actionable.
    """
    st = store_mod.get_store()
    owner = user_id or "anonymous"
    history: List[Dict[str, str]] = []
    conversation = st.get_conversation(conversation_id) if conversation_id else None
    if conversation is not None:
        history = conversation["messages"]

    # Verify the user's stated figures so the model reasons from correct numbers
    # and any incorrect calculation in the reply is corrected before returning.
    summary = finance_calc.extract_financials(message)
    finance_ctx = finance_calc.finance_context(summary)
    system_prompt = (
        f"{ADVISOR_SYSTEM_PROMPT}\n\n{finance_ctx}" if finance_ctx else None
    )

    # Conversation-level grounding: prior financial facts + follow-up/topic brief.
    prior_user_texts = [t.get("content", "") for t in history if t.get("role") == "user"]
    agg_prior = guard.aggregate_financials(prior_user_texts)
    brief = guard.build_conversation_brief(message, history, agg_prior)
    if brief:
        system_prompt = (system_prompt or ADVISOR_SYSTEM_PROMPT) + "\n\n" + brief

    agg_all = guard.aggregate_financials(prior_user_texts + [message])
    guarded = validator.validate_and_refine(
        message=message,
        history=history,
        summary=summary,
        agg_summary=agg_all,
        retrieved_texts=[],
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
        conversation_id = st.create_conversation(
            [user_turn, assistant_turn], user_id=owner
        )
    else:
        st.update_conversation(conversation_id, history + [user_turn, assistant_turn])

    advice: Optional[Advice] = None
    if not guarded.is_clarification and should_extract_advice(reply):
        advice = processor.process_advice(reply, message)
        st.add_advice(advice, conversation_id)

    return ChatResult(conversation_id=conversation_id, reply=reply, advice=advice)


# ---------------------------------------------------------------------------
# Conversation repository
# ---------------------------------------------------------------------------
def list_conversations(user_id: str = "anonymous") -> List[Dict[str, Any]]:
    """One user's conversation summaries (newest first)."""
    return store_mod.get_store().list_conversations(user_id=user_id)


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """A single conversation incl. its full message history (or ``None``)."""
    return store_mod.get_store().get_conversation(conversation_id)


def rename_conversation(conversation_id: str, title: str) -> bool:
    """Set a conversation's display title. False if the conversation is unknown."""
    return store_mod.get_store().rename_conversation(conversation_id, title)


def delete_conversation(conversation_id: str, user_id: str = "anonymous") -> bool:
    """Delete a conversation (owned by ``user_id``) and its advice. False if unknown."""
    return store_mod.get_store().delete_conversation(conversation_id, user_id=user_id)


# ---------------------------------------------------------------------------
# Advice repository
# ---------------------------------------------------------------------------
def list_advice(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """All stored advice (newest first), optionally filtered by category."""
    return store_mod.get_store().list_advice(category=category)


def get_advice(advice_id: str) -> Optional[Dict[str, Any]]:
    """A single stored advice record (or ``None`` if unknown)."""
    return store_mod.get_store().get_advice(advice_id)


def save_advice(advice_id: str) -> bool:
    """Mark an advice record as saved for the dashboard. False if unknown id."""
    return store_mod.get_store().set_saved(advice_id, True)


__all__ = [
    "ChatResult",
    "chat",
    "delete_conversation",
    "get_advice",
    "get_conversation",
    "list_advice",
    "list_conversations",
    "rename_conversation",
    "save_advice",
    "should_extract_advice",
]