"""Topic classification for structured advice (Phase 8).

Assigns an :class:`~src.advice.schemas.AdviceCategory` to a piece of advice
using lightweight keyword/topic matching over the advice text and its source
question. Purely qualitative - no numeric financial inputs are used anywhere.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Sequence, Tuple

from src.advice.schemas import AdviceCategory

# Domain keywords per category (matched as lowercase substrings, longest first
# per category so phrase-level terms beat single-word ones).
CATEGORY_KEYWORDS: Dict[AdviceCategory, Sequence[str]] = {
    AdviceCategory.SAVING: [
        "emergency fund", "savings account", "high-yield", "put money aside",
        "set aside", "save", "savings", "deposit",
    ],
    AdviceCategory.BUDGETING: [
        "50/30/20", "monthly budget", "track", "expense", "spending",
        "non-essential spending", "overspend", "budget",
    ],
    AdviceCategory.RETIREMENT: [
        "social security", "employer match", "retirement", "401", "pension",
        "contribute to an ira", "ira", "roth", "retire",
    ],
    AdviceCategory.INVESTING: [
        "index fund", "mutual fund", "asset allocation", "compound interest",
        "diversif", "portfolio", "dividend", "invest", "stock", "bond", "etf",
    ],
    AdviceCategory.DEBT: [
        "credit counseling", "high-interest debt", "debt snowball", "debt avalanche",
        "pay off", "pay down", "debt", "loan", "mortgage",
    ],
    AdviceCategory.CREDIT: [
        "credit score", "credit report", "credit card", "credit limit",
        "utilization", "credit history", "apr",
    ],
    AdviceCategory.INSURANCE: [
        "life insurance", "term life", "whole life", "insurance", "coverage",
        "premium", "deductible",
    ],
    AdviceCategory.TAXES: [
        "tax-advantaged", "tax-deferred", "tax-free", "after-tax", "pre-tax",
        "tax bracket", "capital gain", "tax credit", "deduction",
        "withholding", "taxable",
    ],
}

# Categories that compete for keyword hits; CONCEPTS / GENERAL are decided last.
_TOPIC_CATEGORIES = tuple(CATEGORY_KEYWORDS)

_DEFINITION_PATTERN = re.compile(
    r"\b(what is|what are|what does|what do|define|definition|explain|"
    r"difference between|is the difference|meaning of)\b",
    re.IGNORECASE,
)


def _keyword_scores(text: str) -> Dict[AdviceCategory, int]:
    corpus = text.lower()
    scores: Dict[AdviceCategory, int] = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(corpus.count(kw) for kw in keywords)
    return scores


def classify(text: str, source_question: Optional[str] = None) -> AdviceCategory:
    """Assign an :class:`AdviceCategory` to an advice text (+ its question)."""
    corpus = " ".join(part for part in (text, source_question or "") if part)
    if not corpus.strip():
        return AdviceCategory.GENERAL

    scores = _keyword_scores(corpus)
    top = max(_TOPIC_CATEGORIES, key=lambda c: scores[c])
    top_score = scores[top]

    is_definition = bool(_DEFINITION_PATTERN.search(corpus))
    if top_score == 0:
        return AdviceCategory.CONCEPTS if is_definition else AdviceCategory.GENERAL
    # Definition-style explainers (e.g. "what is compound interest?") land in
    # CONCEPTS unless a topic clearly dominates (score > 2).
    if is_definition and top_score <= 2:
        return AdviceCategory.CONCEPTS
    return AdviceCategory(top)


__all__ = ["CATEGORY_KEYWORDS", "classify"]