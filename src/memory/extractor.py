"""Memory extraction (Phase 13).

``extract_candidate_memories`` finds durable, worth-remembering statements in a
completed exchange (goals, preferences, facts) and returns them as candidate
memories. It only captures what the user actually said - it never invents or
infers numeric financial-profile data, and it skips transient/one-off chatter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .schemas import CATEGORY_CONTEXT, CATEGORY_FACT, CATEGORY_GOAL, CATEGORY_PREFERENCE

_DURABLE = re.compile(
    r"\b(i|we)\b[^\n.!?]{0,60}\bsaving\s+(?:up\s+)?for\b"
    r"|\b(i|we)\s+(?:would\s+like|want)\s+to\b"
    r"|\b(i|we)\s+prefer\b"
    r"|\b(i|we)\s+already\s+have\b"
    r"|\b(i|we)\s+(?:don't|do\s+not)\s+have\b"
    r"|\bmy\s+goal\b"
    r"|\b(i|we)\s+plan\s+to\b"
    r"|\b(i|we)\s+work\s+(?:at|as|for)\b"
    r"|\b(i|we)\s+need\s+to\b"
    r"|\b(i|we)\s+own\s+(?:a|an|the)\b"
    r"|\b(i|we)\s+rent\s+(?:a|an|the)\b"
    r"|\b(i|we)\s+have\s+(?:a|an|the|already|one|two|three|no)\b",
    re.IGNORECASE,
)

_GOAL = re.compile(
    r"saving\s+(?:up\s+)?for|goal|plan\s+to|want\s+to|would\s+like\s+to|need\s+to",
    re.IGNORECASE,
)

_FACT = re.compile(
    r"already\s+have|own\s+|rent\s+|work\s+(?:at|as|for)|\bhave\s+(?:a|an|the|already|one|two|three|no)",
    re.IGNORECASE,
)

_CHITCHAT = frozenset({"hi", "hello", "hey", "thanks", "thank", "ok", "okay", "bye", "goodbye"})


@dataclass
class MemoryCandidate:
    """A not-yet-persisted memory extracted from an exchange."""

    content: str
    category: str
    importance_score: float


def _normalize(sentence: str) -> str:
    """Normalize first-person phrasing to third-person "user ..." statements."""
    s = sentence
    s = re.sub(r"\bi'm\b", "user is", s, flags=re.IGNORECASE)
    s = re.sub(r"\bi\s+am\b", "user is", s, flags=re.IGNORECASE)
    s = re.sub(r"\bi\s+would\s+like\s+to\b", "user would like to", s, flags=re.IGNORECASE)
    s = re.sub(r"\bi\s+want\s+to\b", "user wants to", s, flags=re.IGNORECASE)
    s = re.sub(r"\bi\s+prefer\b", "user prefers", s, flags=re.IGNORECASE)
    s = re.sub(r"\bi\s+plan\s+to\b", "user plans to", s, flags=re.IGNORECASE)
    s = re.sub(r"\bi\s+already\s+have\b", "user already has", s, flags=re.IGNORECASE)
    s = re.sub(r"\bi\s+(?:don't|do\s+not)\s+have\b", "user does not have", s, flags=re.IGNORECASE)
    s = re.sub(r"\bi\s+have\b", "user has", s, flags=re.IGNORECASE)
    s = re.sub(r"\bmy\s+goal\b", "the user's goal", s, flags=re.IGNORECASE)
    s = re.sub(r"\bsaving\s+up\s+for\b", "saving for", s, flags=re.IGNORECASE)
    s = re.sub(r"\bmy\b", "the user's", s, flags=re.IGNORECASE)
    s = re.sub(r"\bwe\b", "user's household", s, flags=re.IGNORECASE)
    s = re.sub(r"\bi\b", "user", s, flags=re.IGNORECASE)
    return s.strip()


def _classify(content: str) -> str:
    lowered = content.lower()
    if "prefer" in lowered:
        return CATEGORY_PREFERENCE
    if _GOAL.search(content):
        return CATEGORY_GOAL
    if _FACT.search(content):
        return CATEGORY_FACT
    return CATEGORY_CONTEXT


def _importance(category: str, default_importance: float) -> float:
    return {
        CATEGORY_GOAL: 0.7,
        CATEGORY_PREFERENCE: 0.7,
        CATEGORY_FACT: 0.6,
        CATEGORY_CONTEXT: 0.5,
    }.get(category, default_importance)


def extract_candidate_memories(
    user_message: str,
    assistant_response: str,
    *,
    default_importance: float = 0.6,
) -> List[MemoryCandidate]:
    """Return durable memories found in the user's message.

    ``assistant_response`` is accepted for interface parity / future use; the
    heuristic focuses on the user's own first-person statements so it never
    invents data the user didn't state. Questions, chit-chat, and statements
    without a durable marker are skipped.
    """
    sentences = re.split(r"(?<=[.!?])\s+", (user_message or "").strip())
    candidates: List[MemoryCandidate] = []
    seen = set()
    for raw in sentences:
        sentence = raw.strip()
        if not sentence or len(sentence.split()) < 4:
            continue
        if sentence.rstrip().endswith(("?", "!")):
            continue
        words = {word.strip(".,!?;:") for word in sentence.lower().split()}
        if words & _CHITCHAT:
            continue
        if not _DURABLE.search(sentence):
            continue
        content = _normalize(sentence)
        if not content or content in seen:
            continue
        seen.add(content)
        category = _classify(content)
        candidates.append(
            MemoryCandidate(
                content=content,
                category=category,
                importance_score=_importance(category, default_importance),
            )
        )
    return candidates


__all__ = ["MemoryCandidate", "extract_candidate_memories"]
