"""Evaluation metrics for the Financial Advice Chatbot (Phase 6).

Metrics are designed for **conversational advice generation**: response
relevance, length/coherence heuristics, keyword coverage of core financial
concepts, and a rubric-based (optionally LLM-as-judge) relevance/helpfulness
score.

Deliberately **no** numeric financial-correctness metrics (salary/income/risk
profiling) — this product does not score against numeric targets.

All heuristics are standard-library only; the optional ``llm_judge_score`` uses
a transformers model passed in by the caller.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Financial concept lexicon + category heuristics
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "savings": [
        "emergency fund", "savings account", "high-yield", "savings",
        "money market", "apy", "deposit", "save for",
    ],
    "budgeting": [
        "budget", "expense", "spending", "overspend", "50/30/20",
        "tracking", "lifestyle", "frugal",
    ],
    "retirement": [
        "retirement", "ira", "roth", "401", "pension", "social security",
        "employer match", "contribution", "retire",
    ],
    "investing": [
        "invest", "portfolio", "stock", "bond", "etf", "index fund",
        "mutual fund", "dividend", "diversif", "asset allocation",
        "compound interest", "return",
    ],
    "debt-credit": [
        "debt", "credit score", "credit card", "loan", "mortgage", "apr",
        "debt snowball", "debt avalanche", "interest rate", "repay",
    ],
    "insurance": [
        "insurance", "premium", "coverage", "life insurance", "term life",
        "whole life", "deductible", "claim",
    ],
    "taxes": [
        "tax deduction", "tax credit", "tax-advantaged", "taxable",
        "tax-deferred", "capital gain", "withholding", "roth",
    ],
    "general": [
        "financial", "money", "salary", "graduate", "financially",
        "good financial habits", "advice",
    ],
}

FINANCIAL_TERMS = sorted(
    {t for kw in CATEGORY_KEYWORDS.values() for t in kw},
    key=len,
    reverse=True,
)

NOWORDS_ISH = re.compile(r"\b[a-zA-Z]{2,}\b")


def categorize(question: str) -> str:
    """Buckets a question into one of :data:`CATEGORY_KEYWORDS` by keyword hits."""
    q = question.lower()
    scores = {cat: sum(1 for kw in kws if kw in q) for cat, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def matched_financial_terms(question: str) -> List[str]:
    """The financial-concept terms from the lexicon that appear in ``question``."""
    q = question.lower()
    return [t for t in FINANCIAL_TERMS if t in q]


def tokenize_words(text: str) -> List[str]:
    return NOWORDS_ISH.findall(text.lower())


# ---------------------------------------------------------------------------
# Lexical overlap
# ---------------------------------------------------------------------------
def rouge1_f1(reference: str, generated: str) -> float:
    """Rouge-1 F1 between two short texts (reference=question, generated=answer)."""
    ref = tokenize_words(reference)
    gen = tokenize_words(generated)
    if not ref or not gen:
        return 0.0
    ref_set, gen_set = set(ref), set(gen)
    overlap = len(ref_set & gen_set)
    precision = overlap / len(gen_set)
    recall = overlap / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(question: str, response: str) -> float:
    """Fraction of the question's financial-concept terms present in the answer.

    Falls back to ``rouge1_f1`` when the question has no financial terms.
    """
    terms = matched_financial_terms(question)
    if not terms:
        return rouge1_f1(question, response)
    resp_l = response.lower()
    return sum(1 for t in terms if t in resp_l) / len(terms)


# ---------------------------------------------------------------------------
# Length & coherence heuristics
# ---------------------------------------------------------------------------
def word_count(text: str) -> int:
    return len(tokenize_words(text))


def char_count(text: str) -> int:
    return len(text.strip())


def length_adequacy(text: str, min_words: int = 15, max_words: int = 250) -> float:
    """1.0 for answers in [min_words, max_words], decaying outside that range."""
    words = word_count(text)
    if words < min_words:
        return words / min_words
    if words <= max_words:
        return 1.0
    return max(0.0, 1.0 - (words - max_words) / 250.0)


def repetition_score(text: str) -> float:
    """1.0 when no word-bigrams repeat; penalizes repetitive phrasing."""
    words = tokenize_words(text)
    if len(words) < 4:
        return 0.8
    bigrams = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
    counts: Dict[tuple, int] = {}
    for bg in bigrams:
        counts[bg] = counts.get(bg, 0) + 1
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    frac = duplicated / len(bigrams)
    return max(0.0, 1.0 - frac * 4.0)


def punctuation_score(text: str) -> float:
    """Ratio of sentence-termination marks to the expected sentence count."""
    words = word_count(text)
    if words == 0:
        return 0.0
    enders = len(re.findall(r"[.!?](?=\s|$)", text.strip()))
    expected = max(1, math.ceil(words / 15))
    return min(1.0, enders / expected)


def coherence_score(text: str) -> float:
    """Blend of repetition and punctuation quality, each in [0, 1]."""
    return 0.55 * repetition_score(text) + 0.45 * punctuation_score(text)


# ---------------------------------------------------------------------------
# Relevance + rubric
# ---------------------------------------------------------------------------
def relevance_score(question: str, response: str) -> float:
    """0-1 blend of the question's financial-term coverage and lexical overlap."""
    cov = keyword_coverage(question, response)
    if matched_financial_terms(question):
        return 0.6 * cov + 0.4 * rouge1_f1(question, response)
    return cov  # rouge1_f1 fallback


def rubric_score(question: str, response: str) -> float:
    """Grades advice quality on a 1-5 scale.

    Weights: relevance 0.5, coherence 0.3, length adequacy 0.2. The rubric is
    deliberately download/API-free: it rewards on-topic, coherent, reasonably
    sized answers rather than numeric financial correctness.
    """
    overall = (
        0.5 * relevance_score(question, response)
        + 0.3 * coherence_score(response)
        + 0.2 * length_adequacy(response)
    )
    return max(1.0, min(5.0, round(overall * 5.0, 2)))


# ---------------------------------------------------------------------------
# Optional LLM-as-judge (opt-in)
# ---------------------------------------------------------------------------
LLM_JUDGE_SYSTEM = (
    "You are a strict but fair judge of financial-advice responses. "
    "Given a user's question and an assistant's response, rate the response on "
    "RELEVANCE (does it directly address the user's question, 1-5) and "
    "HELPFULNESS (how useful and actionable is the advice, 1-5). "
    "Respond with exactly two lines like: RELEVANCE: 4 / HELPFULNESS: 3"
)


def llm_judge_score(
    question: str,
    response: str,
    judge_model,
    tokenizer,
    max_new_tokens: int = 64,
) -> Optional[Dict[str, int]]:
    """Score response with an (untuned) judge model calling ``generate``.

    Returns ``{relevance: int, helpfulness: int}`` or ``None`` if the judge's
    output cannot be parsed.
    """
    from src.model import model_loader

    user_prompt = (
        f"USER QUESTION:\n{question}\n\n"
        f"ASSISTANT RESPONSE:\n{response[:1200]}\n\n"
        "Scoring rubric: relevance 1-5 (addresses the question), "
        "helpfulness 1-5 (actionable, useful advice).\n"
        "Output exactly: RELEVANCE: <1-5> and HELPFULNESS: <1-5>"
    )
    raw = model_loader.generate(
        judge_model,
        tokenizer,
        user_prompt,
        system_prompt=LLM_JUDGE_SYSTEM,
        generation_kwargs={"max_new_tokens": int(max_new_tokens), "do_sample": False},
    )
    scores = {
        key: int(val)
        for key, val in re.findall(r"(RELEVANCE|HELPFULNESS)\s*[:=]\s*([1-5])", raw)
    }
    if {"RELEVANCE", "HELPFULNESS"} <= set(scores):
        return {"relevance": scores["RELEVANCE"], "helpfulness": scores["HELPFULNESS"]}
    return None


def compute_metrics(question: str, response: str) -> Dict[str, float]:
    """All heuristic metrics for one example (single source of truth for scoring)."""
    words = word_count(response)
    return {
        "word_count": float(words),
        "char_count": float(char_count(response)),
        "length_adequacy": round(length_adequacy(response), 3),
        "repetition_score": round(repetition_score(response), 3),
        "punctuation_score": round(punctuation_score(response), 3),
        "coherence": round(coherence_score(response), 3),
        "keyword_coverage": round(keyword_coverage(question, response), 3),
        "relevance": round(relevance_score(question, response), 3),
        "rubric": round(rubric_score(question, response), 3),
    }


__all__ = [
    "CATEGORY_KEYWORDS",
    "FINANCIAL_TERMS",
    "categorize",
    "char_count",
    "coherence_score",
    "compute_metrics",
    "keyword_coverage",
    "length_adequacy",
    "llm_judge_score",
    "matched_financial_terms",
    "punctuation_score",
    "relevance_score",
    "repetition_score",
    "rouge1_f1",
    "rubric_score",
    "tokenize_words",
    "word_count",
]