"""Conversation-intent and financial-claim validation guard (hallucination fix).

The LLM writes the entire reply, so a wrong turn can state an invented figure as
fact (for example ``"You need $75,000 to start the startup."`` after the user
only mentioned a $30,000 budget, or a made-up rate of return). This module is a
deterministic guard that runs *after* the model replies and before anything is
persisted. It is deliberately conservative:

* **Follow-up detection** (:func:`is_follow_up`) recognizes elliptical messages
  such as ``"why?"``, ``"explain"``, ``"I don't understand"``, ``"what do you
  mean?"`` and ``"how?"`` so the conversation brief can point the model at the
  immediately preceding exchange instead of changing topic.
* **Conversation briefing** (:func:`build_conversation_brief`) assembles the
  current context handed to the model: the previous assistant message on
  follow-ups, a topic-continuity instruction otherwise, and any financial facts
  the user provided in *earlier* turns (verified by the deterministic parser).
* **Claim validation** (:func:`guard_final_reply`) checks every candidate reply
  for unsupported financial claims and, when one is found, replaces the whole
  reply with an honest clarification instead of letting an invented figure reach
  the user. A figure is *grounded* when it matches a user-provided value, a value
  the deterministic plan computed from those values, a simple sum/conversion of
  them, or an explicitly labeled monthly multiple (e.g. "3 months of expenses").

This complements (does not replace) :mod:`src.advice.finance_calc`, which
verifies arithmetic and strips invented lifestyle changes; the guard closes the
remaining gap on *claimed requirements, sufficiency judgements, and returns*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from src.advice import finance_calc
from src.advice.finance_calc import FinancialSummary

# ---------------------------------------------------------------------------
# Amount / percent scanning
# ---------------------------------------------------------------------------
_DOLLAR_RE = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)")
_PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)\s*%")
_MONTH_NUM_RE = re.compile(r"\b(\d{1,2})\s*[- ]?month(?:s)?\b")

# ---------------------------------------------------------------------------
# Follow-up intent detection
# ---------------------------------------------------------------------------
_FOLLOW_UP_EXACT = frozenset({
    "why", "why not", "how", "how so", "how come", "explain", "explain more",
    "please explain", "what", "what do you mean", "what does that mean",
    "meaning", "say again", "come again", "more detail", "more details",
    "in other words", "i don t understand", "i dont understand", "i don t get it",
    "i dont get it", "i don t follow", "i am confused", "im confused",
    "not sure i follow", "not following", "can you explain", "could you explain",
    "can you clarify", "could you clarify", "please clarify", "clarify",
    "go on", "so what", "then what", "ok", "okay", "oh", "hmm", "huh",
    "really", "wait", "seriously", "that s it", "what would that look like",
})

_FOLLOW_UP_PREFIXES = (
    "what do you mean", "what does that mean", "what s the", "what is the",
    "can you explain", "could you explain", "can you clarify", "could you clarify",
    "please clarify", "why is that", "why would that", "why do you say",
    "why exactly", "why not", "why should i", "why do", "why does", "but why",
    "and why", "so why", "how come", "how is that", "how does that",
    "how would that", "how exactly", "how so", "explain that", "explain this",
    "explain why", "explain more", "explain to me", "i don t understand",
    "i dont understand", "i don t get", "i dont get", "i don t follow",
    "i am confused", "im confused", "clarify", "is that because", "are you saying",
    "so you re saying", "you re saying", "in other words", "meaning", "again",
    "okay but", "ok but", "i mean", "wait ,",
)


_FOLLOW_UP_MARK = "MESSAGE-DETECTED-AS-FOLLOW-UP"


def _norm(text: str) -> str:
    lowered = (text or "").strip().lower().replace("'", "").replace("\u2019", "")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip(" ?!,.;:")


def is_follow_up(message: str) -> bool:
    """Whether ``message`` elliptically refers back to the previous exchange.

    Full new questions such as ``"How do I begin investing?"`` or ``"I want to
    open a startup."`` are **not** follow-ups; short wh-phrases and clarification
    requests are.
    """
    lowered = _norm(message)
    if not lowered:
        return False
    if lowered in _FOLLOW_UP_EXACT:
        return True
    if any(lowered.startswith(prefix) for prefix in _FOLLOW_UP_PREFIXES):
        return True
    tokens = lowered.split()
    if len(tokens) <= 2 and tokens and tokens[-1] in ("why", "how", "what", "explain"):
        return True
    return False


# ---------------------------------------------------------------------------
# Conversation brief
# ---------------------------------------------------------------------------
def _facts_block(summary: FinancialSummary) -> Optional[str]:
    """Render a compact facts list for the brief (None when nothing to state)."""
    if summary is None:
        return None
    lines: List[str] = []
    if summary.income is not None:
        lines.append(f"- Income: {finance_calc.format_amount(summary.income)}")
    for item in summary.expenses:
        lines.append(f"- {item.category}: {finance_calc.format_amount(item.amount)}")
    if summary.expenses:
        lines.append(
            f"- Total monthly expenses: {finance_calc.format_amount(summary.total_monthly_expenses)}"
        )
    if summary.savings_balance is not None:
        lines.append(f"- Savings: {finance_calc.format_amount(summary.savings_balance)}")
    if summary.savings_contributions:
        lines.append(
            f"- Monthly savings contributions: "
            f"{finance_calc.format_amount(sum(summary.savings_contributions))}"
        )
    for debt in summary.debts:
        line = f"- {debt.description}: {finance_calc.format_amount(debt.amount)}"
        if debt.apr is not None:
            line += f" ({debt.apr:g}% APR)"
        lines.append(line)
    if summary.debt_payments:
        lines.append(
            f"- Monthly debt payments: {finance_calc.format_amount(sum(summary.debt_payments))}"
        )
    if summary.monthly_surplus is not None:
        lines.append(f"- Monthly surplus: {finance_calc.format_amount(summary.monthly_surplus)}")
    return "\n".join(lines) or None


def _last_assistant_message(history: Sequence[dict]) -> Optional[str]:
    for turn in reversed(history or ()):
        if turn.get("role") == "assistant":
            content = (turn.get("content") or "").strip()
            return content or None
    return None


def build_conversation_brief(
    message: str,
    history: Sequence[dict],
    agg_prior: FinancialSummary,
) -> Optional[str]:
    """A system-prompt context block for the current turn, or ``None``.

    * ``None`` for a first message with no earlier financial facts (no change to
      current behavior).
    * Follow-up messages get the previous assistant message plus a "stay on
      topic / explicitly correct any wrong earlier figure" instruction.
    * Other multi-turn messages get a topic-continuity instruction. A user who
      clearly states a new topic is still free to switch.
    """
    follow_up = is_follow_up(message)
    facts = _facts_block(agg_prior)
    if not history and not facts:
        return None

    previous = _last_assistant_message(history)
    parts: List[str] = []
    if follow_up and previous:
        parts.append(
            _FOLLOW_UP_MARK + "\n"
            "The user's latest message is a follow-up to your immediately previous "
            "answer. Interpret it using that exchange alone. Do not change topic. "
            "If your previous answer contained an unsupported or incorrect financial "
            "figure, EXPLICITLY CORRECT OR RETRACT that figure before continuing."
        )
        parts.append(f"Your previous assistant message was:\n{previous}")
    else:
        parts.append(
            "CONVERSATION CONTEXT (for this message):\n"
            "Continue the current conversation topic. Stay on the topic of the "
            "immediately preceding exchange unless the user's message clearly asks "
            "about a new topic."
        )
    if facts:
        parts.append(
            "Financial information the user previously provided (verified by the "
            "system; use exactly these figures or transparent calculations from "
            "them):\n" + facts
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Financial summary aggregation across turns
# ---------------------------------------------------------------------------
def aggregate_financials(texts: Sequence[str]) -> FinancialSummary:
    """Merge several user messages into one :class:`FinancialSummary`.

    Later messages win for ``income`` and ``savings_balance``; expense categories,
    debts, contributions, payments and goals are unioned (deduplicated). Totals
    and surplus are recomputed by the summary properties.
    """
    merged = FinancialSummary()
    seen_categories = set()
    seen_debts = set()
    seen_goals = set()
    for text in texts or ():
        summary = finance_calc.extract_financials(text)
        if summary.income is not None:
            merged.income = summary.income
        if summary.savings_balance is not None:
            merged.savings_balance = summary.savings_balance
        for item in summary.expenses:
            key = item.category.strip().lower()
            if key in seen_categories:
                continue
            seen_categories.add(key)
            merged.expenses.append(item)
        merged.savings_contributions.extend(summary.savings_contributions)
        for debt in summary.debts:
            key = (debt.description.strip().lower(), round(debt.amount, 2))
            if key in seen_debts:
                continue
            seen_debts.add(key)
            merged.debts.append(debt)
        merged.debt_payments.extend(summary.debt_payments)
        for goal in summary.goals:
            key = goal.strip().lower()
            if key in seen_goals:
                continue
            seen_goals.add(key)
            merged.goals.append(goal)
    return merged


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------
def _append_unique(values: List[float], amount: Optional[float]) -> None:
    if amount is None:
        return
    value = round(float(amount), 2)
    if any(abs(value - existing) < 0.01 for existing in values):
        return
    values.append(value)


def grounded_amounts(summary: FinancialSummary) -> List[float]:
    """All dollar figures the system can vouch for (user values + computed plan)."""
    values: List[float] = []
    if summary is None:
        return values
    _append_unique(values, summary.income)
    _append_unique(values, summary.savings_balance)
    for item in summary.expenses:
        _append_unique(values, item.amount)
    _append_unique(values, summary.total_monthly_expenses if summary.expenses else None)
    _append_unique(values, summary.monthly_surplus)
    for contribution in summary.savings_contributions:
        _append_unique(values, contribution)
    _append_unique(values, sum(summary.savings_contributions) if summary.savings_contributions else None)
    for debt in summary.debts:
        _append_unique(values, debt.amount)
    for payment in summary.debt_payments:
        _append_unique(values, payment)
    _append_unique(values, sum(summary.debt_payments) if summary.debt_payments else None)

    plan = finance_calc.build_plan(summary)
    for attr in (
        "emergency_fund_target", "emergency_fund_gap", "near_term_buffer",
        "recommended_initial_payment", "retained_buffer",
        "remaining_high_interest_debt", "high_interest_debt_total",
        "other_debt_total", "recommended_monthly_debt_payment",
        "recommended_monthly_savings", "recommended_monthly_investing",
    ):
        _append_unique(values, getattr(plan, attr, None) or None)
    return values


def grounded_percents(summary: FinancialSummary) -> List[float]:
    """Percent values that are grounded (only APRs stated by the user)."""
    if summary is None:
        return []
    return [debt.apr for debt in summary.debts if debt.apr is not None]


def is_grounded_number(value: float, grounded: Sequence[float], sentence: str = "") -> bool:
    """Whether ``value`` is a transparent derivation of grounded figures.

    Exact matches, pairwise sums, annual/monthly conversions, and (only when the
    sentence explicitly mentions months, e.g. "3 months of expenses") simple
    monthly multiples all count as grounded.
    """
    if not grounded:
        return False
    tolerance = 1.0
    for g in grounded:
        if g is None:
            continue
        if abs(value - g) <= tolerance:
            return True
        if abs(value * 12 - g) <= tolerance or abs(value - g * 12) <= tolerance:
            return True
    for i, g1 in enumerate(grounded):
        if g1 is None:
            continue
        for g2 in grounded[:i]:
            if g2 is None:
                continue
            if abs(value - (g1 + g2)) <= tolerance:
                return True
    months = _MONTH_NUM_RE.search(sentence or "")
    if months:
        multiplier = int(months.group(1))
        if 2 <= multiplier <= 24:
            for g in grounded:
                if g is None:
                    continue
                if abs(value - g * multiplier) <= tolerance:
                    return True
    return False


def is_grounded_percent(value: float, percents: Sequence[float]) -> bool:
    if not percents:
        return False
    return any(abs(value - p) <= 0.01 for p in percents if p is not None)


# ---------------------------------------------------------------------------
# Unsupported-claim detection
# ---------------------------------------------------------------------------
# Requirement / threshold / cost claim words (only meaningful with a dollar amount).
_REQUIREMENT_RE = re.compile(
    r"\b(need|needs|needed|require|requires|required|requirement|must|mandatory|"
    r"minimum|upfront|capital|to start|will cost|costs? (?:about|around|almost|"
    r"at least|roughly|over|under)|average (?:startup|business|cost|costs)|"
    r"typically requires?|usually requires?)\b",
    re.IGNORECASE,
)
# Sufficiency judgements about "enough"/"insufficient" (require a dollar amount +
# a cost context to avoid flagging neutral statements).
_INSUFFICIENCY_RE = re.compile(
    r"\b(not enough|isn'?t enough|is not enough|insufficient|won'?t be enough|"
    r"won'?t cover|can'?t (?:start|cover|afford)|will need more|lack the "
    r"(?:funds|capital|money|budget))\b",
    re.IGNORECASE,
)
_COST_CONTEXT_RE = re.compile(
    r"\b(start|startup|business|venture|invest|investment|project|plan|set ?up|"
    r"open|company|product|build)\b",
    re.IGNORECASE,
)
# Return / rate-of-return / market-statistic words (only meaningful with a percent).
_RETURN_RE = re.compile(
    r"\b(roi|return|yield|interest rate|annual return|average return|historically|"
    r"guarantee(?:d)?|expect to (?:earn|return)|earns?|generates?|grow(?:s)? at|"
    r"per year|a year|annually|annum)\b",
    re.IGNORECASE,
)


def _sentences(reply: str) -> List[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n", reply or "") if part.strip()]
    return parts


def _sentence_figures(sentence: str) -> tuple:
    dollars = [float(re.sub(r"[,$]", "", d)) for d in _DOLLAR_RE.findall(sentence)]
    pcts = [float(p) for p in _PERCENT_RE.findall(sentence)]
    return dollars, pcts


def requirement_claim_sentences(reply: str, grounded: Sequence[float]) -> List[str]:
    """Sentences that assert an unsupported requirement / cost / threshold claim
    (e.g. ``"You will need $75,000 in starting capital."``). Returns are excluded."""
    claims: List[str] = []
    for sentence in _sentences(reply):
        dollars, _ = _sentence_figures(sentence)
        ungrounded = [
            d for d in dollars if not is_grounded_number(d, grounded, sentence)
        ]
        if ungrounded and _REQUIREMENT_RE.search(sentence):
            claims.append(sentence)
    return claims


def insufficiency_claim_sentences(reply: str) -> List[str]:
    """Sentences judging a stated amount as insufficient (``"Your $30,000 is not
    enough to start a startup."``)."""
    claims: List[str] = []
    for sentence in _sentences(reply):
        dollars, _ = _sentence_figures(sentence)
        if dollars and _INSUFFICIENCY_RE.search(sentence) and _COST_CONTEXT_RE.search(sentence):
            claims.append(sentence)
    return claims


def find_unsupported_claims(
    reply: str,
    grounded: Sequence[float],
    percents: Sequence[float],
) -> List[str]:
    """Return the reply sentences that contain unsupported financial claims."""
    claims: List[str] = []
    claims.extend(requirement_claim_sentences(reply, grounded))
    claims.extend(insufficiency_claim_sentences(reply))
    for sentence in _sentences(reply):
        _, pcts = _sentence_figures(sentence)
        ungrounded_pcts = [p for p in pcts if not is_grounded_percent(p, percents)]
        if ungrounded_pcts and _RETURN_RE.search(sentence):
            claims.append(sentence)
    return claims


# ---------------------------------------------------------------------------
# Guard entry point
# ---------------------------------------------------------------------------
def _budget_amount(summary: FinancialSummary) -> Optional[float]:
    if summary is None:
        return None
    if summary.savings_balance is not None:
        return summary.savings_balance
    return summary.income


def build_clarification(summary: FinancialSummary, message: str) -> str:
    """An honest, non-invented reply used when the model made an unsupported claim."""
    budget = _budget_amount(summary)
    block = (
        "I don't have enough information to determine that reliably, and I won't "
        "invent a financial figure to fill the gap."
    )
    if budget:
        block += (
            f"\n\nUsing the {finance_calc.format_amount(budget)} figure you mentioned, "
            "I can help you explore options that realistically fit that amount - but "
            "I will only work from numbers you provided or that we can calculate from "
            "them."
        )
    block += (
        "\n\nIf you share the specific details (for example, the costs involved, your "
        "available budget, or your income and expenses), I can reason from those exact "
        "numbers."
    )
    return block


@dataclass
class GuardedReply:
    """Outcome of guarding one model reply."""

    reply: str
    is_clarification: bool = False


def guard_final_reply(
    reply: str,
    user_message: str,
    summary: Optional[FinancialSummary] = None,
) -> GuardedReply:
    """Run the full guard on a candidate reply.

    When an unsupported financial claim is found the entire reply is replaced
    with a clarification (never leaving a partial hallucinated figure), otherwise
    the reply is returned unchanged.
    """
    summary = summary or aggregate_financials([user_message])
    claims = find_unsupported_claims(
        reply, grounded_amounts(summary), grounded_percents(summary)
    )
    if claims:
        return GuardedReply(
            reply=build_clarification(summary, user_message), is_clarification=True
        )
    return GuardedReply(reply=reply)


__all__ = [
    "GuardedReply",
    "aggregate_financials",
    "build_clarification",
    "build_conversation_brief",
    "find_unsupported_claims",
    "guard_final_reply",
    "grounded_amounts",
    "grounded_percents",
    "insufficiency_claim_sentences",
    "is_follow_up",
    "is_grounded_number",
    "is_grounded_percent",
    "requirement_claim_sentences",
]