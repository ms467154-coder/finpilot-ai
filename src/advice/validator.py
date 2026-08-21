"""Response validator: extract financial claims, classify them, validate replies.

The model writes every reply, so a wrong turn can state invented figures as
facts, contradict what the user already told us, drift to an unrelated financial
topic, or present an assumption as a certainty. This module is the response
validation layer for FinAdvise:

* :func:`extract_claims` pulls each financial figure out of a reply and
  classifies the claims that carry it:

    - ``USER_PROVIDED`` - matches a figure the user stated in this conversation;
    - ``RETRIEVED`` - matches a figure found in retrieved context/memories;
    - ``CALCULATED`` - a transparent derivation (total, surplus, plan defaults,
      sums/conversions/month multiples) of verified figures;
    - ``ESTIMATE`` / ``ASSUMPTION`` - explicitly hedged ("about 7%",
      "assuming X") rather than asserted as fact;
    - ``UNKNOWN`` - not grounded anywhere and asserted as fact.

* :func:`validate_response` turns that into a pass/fail verdict with human
  readable issues covering unsupported numbers/claims, contradictions with the
  conversation, and topic drift (claims unrelated to current intent).

* :func:`validate_and_refine` is the orchestration helper: generate -> correct
  -> validate; on failure regenerate with the validation feedback (bounded by
  :data:`MAX_REGENERATION_ATTEMPTS`); if it still fails, return an honest
  clarification instead of silently passing the invalid response.

This layer is intentionally lightweight: it reuses the deterministic grounding
from :mod:`src.advice.guard` and the plan from :mod:`src.advice.finance_calc`,
and it takes the model's next response via a plain ``generate`` callable so it
stays decoupled from the inference stack.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from src.advice import finance_calc, guard
from src.advice.guard import GuardedReply
from src.advice.finance_calc import FinancialSummary
from src.inference.prompt_templates import ADVISOR_SYSTEM_PROMPT

MAX_REGENERATION_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# Claim model
# ---------------------------------------------------------------------------
class ClaimCategory(str, enum.Enum):
    """How the validator classifies every extracted financial claim."""

    USER_PROVIDED = "USER_PROVIDED"
    RETRIEVED = "RETRIEVED"
    CALCULATED = "CALCULATED"
    ESTIMATE = "ESTIMATE"
    ASSUMPTION = "ASSUMPTION"
    UNKNOWN = "UNKNOWN"


@dataclass
class Claim:
    """One extracted financial claim (a figure asserted inside a sentence)."""

    text: str
    category: ClaimCategory
    amount: Optional[float]
    is_percent: bool = False


@dataclass
class ValidationResult:
    """Verdict for one candidate response."""

    passed: bool
    claims: List[Claim]
    issues: List[str]


# ---------------------------------------------------------------------------
# Text helpers (kept local to stay decoupled from guard internals)
# ---------------------------------------------------------------------------
_DOLLAR_RE = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)")
_PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)\s*%")


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n", text or "") if p.strip()]
    return parts


def _clean_amount(raw: str) -> float:
    return float(raw.replace(",", ""))


# (category, amount) helpers --------------------------------------------------


def _figure_sets(
    agg_summary: Optional[FinancialSummary],
    retrieved_texts: Sequence[str],
) -> Tuple[Set[float], Set[float], List[float], Set[float], Set[float], Set[float], Set[float]]:
    """Build the grounding sets used to classify figures.

    Returns ``(user_amounts, user_percents, calc_amounts, retrieved_amounts,
    retrieved_percents, framework_percents, requirement_grounded_amounts)``.
    """
    user_amounts: Set[float] = set()
    user_percents: Set[float] = set()
    calc_amounts: List[float] = []
    framework_percents: Set[float] = set()

    if agg_summary is not None:
        if agg_summary.income is not None:
            user_amounts.add(round(agg_summary.income, 2))
        if agg_summary.savings_balance is not None:
            user_amounts.add(round(agg_summary.savings_balance, 2))
        for item in agg_summary.expenses:
            user_amounts.add(round(item.amount, 2))
        for contribution in agg_summary.savings_contributions:
            user_amounts.add(round(contribution, 2))
        for debt in agg_summary.debts:
            user_amounts.add(round(debt.amount, 2))
            if debt.apr is not None:
                user_percents.add(round(debt.apr, 2))
        for payment in agg_summary.debt_payments:
            user_amounts.add(round(payment, 2))

        if agg_summary.expenses:
            calc_amounts.append(agg_summary.total_monthly_expenses)
        if agg_summary.monthly_surplus is not None:
            calc_amounts.append(agg_summary.monthly_surplus)
        if agg_summary.savings_contributions:
            calc_amounts.append(round(sum(agg_summary.savings_contributions), 2))
        if agg_summary.debt_payments:
            calc_amounts.append(round(sum(agg_summary.debt_payments), 2))

        plan = finance_calc.build_plan(agg_summary)
        for attr in (
            "emergency_fund_target", "emergency_fund_gap", "near_term_buffer",
            "recommended_initial_payment", "retained_buffer",
            "remaining_high_interest_debt", "high_interest_debt_total",
            "other_debt_total", "recommended_monthly_debt_payment",
            "recommended_monthly_savings", "recommended_monthly_investing",
        ):
            value = getattr(plan, attr, None)
            if value:
                calc_amounts.append(round(float(value), 2))

    framework_percents.add(finance_calc.HIGH_INTEREST_APR_THRESHOLD)

    retrieved_amounts: Set[float] = set()
    retrieved_percents: Set[float] = set()
    for text in retrieved_texts or ():
        retrieved_amounts.update(
            round(_clean_amount(a), 2) for a in _DOLLAR_RE.findall(text)
        )
        retrieved_percents.update(
            round(float(p), 2) for p in _PERCENT_RE.findall(text)
        )

    return (
        user_amounts, user_percents, calc_amounts,
        retrieved_amounts, retrieved_percents, framework_percents,
    )


# Hedging ---------------------------------------------------------------
_ESTIMATE_RE = re.compile(
    r"\b(about|around|roughly|approximately|approx|estimate(?:d|s)?|historically|"
    r"typically|usually|generally|on average|average|likely|up to|more than|less "
    r"than|between|maybe|perhaps|may|might|could|can|possibly|expected|"
    r"it depends)\w*\b",
    re.IGNORECASE,
)
_ASSUMPTION_RE = re.compile(
    r"\b(assume|assumption|assuming|hypothetical|hypothetically|won'?t[ -]?if|"
    r"let'?s say|say you|for example|if you|provided that|dependent on)\b",
    re.IGNORECASE,
)


def _hedge_kind(sentence: str) -> Optional[str]:
    lowered = sentence.lower()
    if _ASSUMPTION_RE.search(lowered):
        return "assumption"
    if _ESTIMATE_RE.search(lowered):
        return "estimate"
    return None


def _classify_figure(
    amount: float,
    is_percent: bool,
    sentence: str,
    user_amounts: Set[float],
    user_percents: Set[float],
    calc_amounts: List[float],
    retrieved_amounts: Set[float],
    retrieved_percents: Set[float],
    framework_percents: Set[float],
    requirement_sentences: Set[str],
) -> ClaimCategory:
    rounded = round(amount, 2)
    if rounded in user_amounts or (is_percent and rounded in user_percents):
        return ClaimCategory.USER_PROVIDED
    if rounded in retrieved_amounts or (is_percent and rounded in retrieved_percents):
        return ClaimCategory.RETRIEVED
    if not is_percent and (
        rounded in calc_amounts
        or guard.is_grounded_number(amount, calc_amounts, sentence)
    ):
        return ClaimCategory.CALCULATED
    if is_percent and rounded in framework_percents:
        return ClaimCategory.ASSUMPTION
    if sentence in requirement_sentences:
        return ClaimCategory.UNKNOWN
    hedge = _hedge_kind(sentence)
    if hedge:
        return ClaimCategory.ASSUMPTION if hedge == "assumption" else ClaimCategory.ESTIMATE
    return ClaimCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------
def extract_claims(
    reply: str,
    agg_summary: Optional[FinancialSummary],
    retrieved_texts: Sequence[str] | None = None,
    history: Sequence[dict] | None = None,
) -> List[Claim]:
    """Extract and classify every financial claim in ``reply``.

    Classification is per figure: each dollar amount / percentage asserted in a
    sentence becomes one :class:`Claim`. Figures flagged by the guard as
    unsupported requirement / sufficiency claims are ``UNKNOWN`` even when
    hedged. Only figures are extracted (non-financial prose is left alone).
    """
    (
        user_amounts, user_percents, calc_amounts,
        retrieved_amounts, retrieved_percents, framework_percents,
    ) = _figure_sets(agg_summary, retrieved_texts)

    history_texts = [t.get("content", "") for t in history or ()]
    requirement_sentences = set(
        guard.requirement_claim_sentences(
            reply, guard.grounded_amounts(agg_summary)
        )
    )

    claims: List[Claim] = []
    for sentence in _split_sentences(reply):
        amounts = [_clean_amount(a) for a in _DOLLAR_RE.findall(sentence)]
        percents = [float(p) for p in _PERCENT_RE.findall(sentence)]
        for amount in amounts:
            claims.append(
                Claim(
                    text=sentence,
                    category=_classify_figure(
                        amount, False, sentence, user_amounts, user_percents,
                        calc_amounts, retrieved_amounts, retrieved_percents,
                        framework_percents, requirement_sentences,
                    ),
                    amount=amount,
                    is_percent=False,
                )
            )
        for percent in percents:
            claims.append(
                Claim(
                    text=sentence,
                    category=_classify_figure(
                        percent, True, sentence, user_amounts, user_percents,
                        calc_amounts, retrieved_amounts, retrieved_percents,
                        framework_percents, requirement_sentences,
                    ),
                    amount=percent,
                    is_percent=True,
                )
            )
    return claims


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------
def _find_contradictions(reply: str, agg_summary: Optional[FinancialSummary]) -> List[str]:
    """Sentences that restate a user-provided fact with the wrong figure."""
    if agg_summary is None:
        return []
    # (label, kind, expected) - kind is "dollar" or "percent" so a debt balance is
    # only compared against dollar figures and an APR only against percents.
    dollar_labels: Dict[str, float] = {}
    percent_labels: Dict[str, float] = {}
    if agg_summary.income is not None:
        dollar_labels["income"] = round(agg_summary.income, 2)
    surplus = agg_summary.monthly_surplus
    if surplus is not None:
        dollar_labels["surplus"] = round(surplus, 2)
    if agg_summary.savings_balance is not None:
        dollar_labels["savings"] = round(agg_summary.savings_balance, 2)
    if agg_summary.expenses:
        total = round(agg_summary.total_monthly_expenses, 2)
        dollar_labels["expenses"] = total
        dollar_labels["total monthly expenses"] = total
        for item in agg_summary.expenses:
            dollar_labels[item.category.strip().lower()] = round(item.amount, 2)
    for debt in agg_summary.debts:
        key = debt.description.strip().lower()
        dollar_labels[key] = round(debt.amount, 2)
        if debt.apr is not None:
            percent_labels[key] = round(debt.apr, 2)

    issues: List[str] = []
    for sentence in _split_sentences(reply):
        lowered = sentence.lower()
        amounts = [_clean_amount(a) for a in _DOLLAR_RE.findall(sentence)]
        percents = [float(p) for p in _PERCENT_RE.findall(sentence)]
        if amounts:
            for label, expected in dollar_labels.items():
                if label not in lowered:
                    continue
                if min(abs(f - expected) for f in amounts) > 1.0:
                    issues.append(sentence)
                    break
        if percents:
            for label, expected in percent_labels.items():
                if label not in lowered:
                    continue
                if min(abs(f - expected) for f in percents) > 0.01:
                    issues.append(sentence)
                    break
    return _dedupe(issues)


_DRIFT_RE = re.compile(
    r"\b(returns?|roi|yield|dividend|compound(?:ed)? interest|capital gains|"
    r"stock(?:s)?|bonds?|index fund(?:s)?|mutual fund(?:s)?|\betf\b|forex|"
    r"crypto(?:currency)?|real estate|property market|market returns|interest rate(?:s)?|"
    r"inflation hedge|options trading|day trading)\b",
    re.IGNORECASE,
)


def _find_drift(
    reply: str,
    message: str,
    history: Sequence[dict],
    retrieved_texts: Sequence[str],
) -> List[str]:
    """Sentences making numeric claims in a topic never present in the conversation.

    The canonical case: the user asks about starting a business and the reply
    starts quoting investment returns. Leniency: only *numeric* claims in a
    drifted domain are flagged, so advisory prose that merely mentions another
    topic is not penalized.
    """
    context = (message or "") + "\n" + "\n".join(
        t.get("content", "") for t in history or ()
    ) + "\n" + "\n".join(retrieved_texts or ())
    context_lower = context.lower()

    issues: List[str] = []
    for sentence in _split_sentences(reply):
        if not _DOLLAR_RE.search(sentence) and not _PERCENT_RE.search(sentence):
            continue
        matches = _DRIFT_RE.findall(sentence.lower())
        if matches and not any(match in context_lower for match in matches):
            issues.append(sentence)
    return _dedupe(issues)


def _dedupe(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _format_amount(value: Optional[float]) -> str:
    if value is None:
        return "?"
    return finance_calc.format_amount(value)


def _as_value(claim: Claim) -> str:
    if claim.is_percent:
        return f"{claim.amount:g}%"
    return _format_amount(claim.amount)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_response(
    reply: str,
    message: str,
    history: Sequence[dict],
    agg_summary: Optional[FinancialSummary] = None,
    retrieved_texts: Sequence[str] | None = None,
) -> ValidationResult:
    """Classify every claim in ``reply`` and return a pass/fail verdict.

    A reply fails when any of these hold:

    * a claim is ``UNKNOWN`` but presented as fact (unsupported number or claim,
      or advice that cannot be justified from the available context);
    * a figure contradicts a fact the user already provided;
    * a numeric claim is made in a financial topic never present in the
      conversation (topic drift / unrelated to current intent).
    """
    retrieved_texts = retrieved_texts or ()
    claims = extract_claims(reply, agg_summary, retrieved_texts, history)
    issues: List[str] = []

    for claim in claims:
        if claim.category is ClaimCategory.UNKNOWN:
            issues.append(
                f"Presents {_as_value(claim)} as a fact without support: "
                f"{claim.text!r}"
            )

    for sentence in guard.insufficiency_claim_sentences(reply):
        issues.append(f"Unsupported financial claim: {sentence!r}")

    for sentence in _find_contradictions(reply, agg_summary):
        issues.append(
            f"Contradicts a figure the user already provided: {sentence!r}"
        )

    for sentence in _find_drift(reply, message, history, retrieved_texts):
        issues.append(
            f"Topic drift / numeric claim unrelated to current intent: {sentence!r}"
        )

    return ValidationResult(passed=not issues, claims=claims, issues=_dedupe(issues))


# ---------------------------------------------------------------------------
# Regeneration feedback + loop
# ---------------------------------------------------------------------------
def feedback_text(result: ValidationResult) -> str:
    """A system-prompt block telling the model what to fix and to try again."""
    lines = [
        "The previous answer was rejected by response validation. Produce a new, "
        "corrected answer that fixes every issue below; do not repeat the rejected "
        "content. Give only what can be supported by the user's stated figures, "
        "figures computed from them, or retrieved context. If you cannot support a "
        "number, say you do not have enough information instead of inventing it."
    ]
    lines.extend(f"- {issue}" for issue in result.issues)
    return "\n".join(lines)


def validate_and_refine(
    message: str,
    history: Sequence[dict],
    summary: Optional[FinancialSummary],
    agg_summary: Optional[FinancialSummary],
    retrieved_texts: Sequence[str],
    base_system_prompt: Optional[str],
    generate: Callable[[str, Sequence[dict], Optional[str]], str],
    correct: Callable[[str, str, Optional[FinancialSummary]], str],
) -> GuardedReply:
    """Full generate -> correct -> validate -> (regenerate | clarify) pipeline.

    ``generate(message, history, system_prompt)`` produces one raw reply;
    ``correct(reply, message, summary)`` is the deterministic reply fix-up. An
    invalid response is rejected and regenerated with ``feedback_text`` feedback,
    up to :data:`MAX_REGENERATION_ATTEMPTS` times. If it still fails, an honest
    clarification is returned - the invalid response is never passed through.
    """
    system_prompt = base_system_prompt
    for attempt in range(MAX_REGENERATION_ATTEMPTS + 1):
        raw = generate(message, history, system_prompt)
        reply = correct(raw, message, summary)
        result = validate_response(reply, message, history, agg_summary, retrieved_texts)
        if result.passed:
            return GuardedReply(reply=reply)

        if attempt == MAX_REGENERATION_ATTEMPTS:
            break
        base = base_system_prompt if base_system_prompt else ADVISOR_SYSTEM_PROMPT
        system_prompt = base + "\n\nVALIDATION FEEDBACK\n" + feedback_text(result)

    return GuardedReply(
        reply=guard.build_clarification(agg_summary, message), is_clarification=True
    )


__all__ = [
    "Claim",
    "ClaimCategory",
    "MAX_REGENERATION_ATTEMPTS",
    "ValidationResult",
    "extract_claims",
    "feedback_text",
    "validate_and_refine",
    "validate_response",
]