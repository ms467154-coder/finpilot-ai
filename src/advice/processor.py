"""Advice processing layer (Phase 8).

Orchestrates turning the **raw text** returned by
:func:`src.inference.inference.generate_advice` into a structured
:class:`~src.advice.schemas.Advice` object suitable for storage/display::

    from src.advice import processor

    advice = processor.process_advice(
        raw_text=reply,
        source_question="How can I save money?",
    )
    advice.to_dict()  # -> {id, timestamp, category, short_title,
                      #     key_recommendation, full_text, source_question}

Everything here is deterministic and qualitative; no API routes or persistence
wiring yet (those arrive in Phase 9).
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from src.advice import categorizer
from src.advice.schemas import Advice, AdviceCategory, now_utc

# Action/advice markers used to pick the single most actionable sentence.
_ADVICE_MARKERS = [
    "pay off", "pay down", "start with", "start by", "make sure", "set up",
    "build", "create", "review", "choose", "consider", "contribute", "save",
    "invest", "reduce", "avoid", "keep", "track", "use", "add", "increase",
    "should", "don't", "dont", "put", "cut back", "rebalance",
]

_ROLE_SUFFIXES = ("<|im_end|>", "<|endoftext|>", "</s>")

_TITLE_MAX = 64
_RECOMMENDATION_MAX = 180


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def clean_text(raw: str) -> str:
    """Trim and normalize the raw advisor response (no role tokens)."""
    text = raw or ""
    for suffix in _ROLE_SUFFIXES:
        text = text.replace(suffix, "")
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _sentences(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _first_sentence(text: str) -> str:
    for sentence in _sentences(text):
        if len(sentence.split()) >= 4:
            return sentence
    return _truncate(text, _TITLE_MAX)


# ---------------------------------------------------------------------------
# Title + key recommendation
# ---------------------------------------------------------------------------
def suggest_title(text: str, category: AdviceCategory, source_question: str) -> str:
    """A short human-readable headline derived from the first real sentence."""
    first = _first_sentence(text)
    if not first:
        return f"{category.value} guidance"
    first = re.sub(
        r"^(yes\.\s*|there (are|is)\s+|there's\s+|here are\s+)", "", first, flags=re.IGNORECASE
    )
    title = _truncate(first.rstrip(".:;"), _TITLE_MAX)
    return title[0].upper() + title[1:] if title else f"{category.value} guidance"


def extract_key_recommendation(text: str) -> str:
    """Pick the single most actionable sentence from the advice text."""
    sentences = _sentences(text)
    if not sentences:
        return _truncate(text, _RECOMMENDATION_MAX)

    def score(sentence: str) -> int:
        lower = sentence.lower()
        return sum(lower.count(marker) for marker in _ADVICE_MARKERS)

    best = max(sentences, key=lambda s: (score(s), len(s.split())))
    if score(best) == 0:
        best = _first_sentence(text) or best
    return _truncate(best.rstrip(".:;"), _RECOMMENDATION_MAX)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def process_advice(
    raw_text: str,
    source_question: str,
    *,
    advice_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    category: Optional[AdviceCategory] = None,
    short_title: Optional[str] = None,
    key_recommendation: Optional[str] = None,
) -> Advice:
    """Convert ``raw_text`` into a structured :class:`Advice` object.

    All fields are optional except ``raw_text`` / ``source_question``; the rest
    are derived deterministically (or overridden if supplied).
    """
    full_text = clean_text(raw_text)
    if not full_text:
        raise ValueError("raw_text produced an empty advice body")

    cat = category or categorizer.classify(full_text, source_question)
    title = short_title or suggest_title(full_text, cat, source_question)
    key = key_recommendation or extract_key_recommendation(full_text)

    return Advice(
        id=advice_id or uuid.uuid4().hex,
        timestamp=timestamp or now_utc(),
        category=cat,
        short_title=title,
        key_recommendation=key,
        full_text=full_text,
        source_question=source_question,
    )


def process_many(
    pairs: Iterable[Tuple[str, str]],
    **overrides,
) -> List[Advice]:
    """Process several ``(source_question, raw_text)`` pairs into ``Advice``."""
    return [process_advice(text, question, **overrides) for question, text in pairs]


def advice_to_dict(advice: Advice) -> Dict[str, Union[str, datetime]]:
    """Convenience: JSON-serializable dict (ISO timestamp, string category)."""
    return advice.to_dict()


__all__ = [
    "clean_text",
    "extract_key_recommendation",
    "process_advice",
    "process_many",
    "suggest_title",
]