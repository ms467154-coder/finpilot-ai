"""Deterministic financial parsing, verification, and advice reasoning context.

Root cause of the observed problems
-----------------------------------
The LLM (``inference.generate_advice``) writes the entire reply, including any
arithmetic. That produced two failure modes:

1. **Wrong math**: nothing verified the model's figures, so a budget with $1,500
   income and $1,150 of expenses could be answered with "$1,000 - $1,500 =
   -$500/month", plus duplicated sections and echoed prompt text.
2. **Poor reasoning**: the model was given only three verified numbers (income /
   expenses / surplus) and none of the user's savings balance, debt (with APR),
   or goals - so it invented lifestyle changes (move, roommate), random
   percentage cuts, and impossible promises.

This module is the verification + reasoning layer. It:

* parses the user's figures (income, expense categories, savings balance,
  debt balances with APR, monthly debt payments, goals) from arbitrary text;
* computes the *only* reliable numbers::

      total_monthly_expenses = sum(valid expense categories)
      monthly_surplus       = monthly_income - total_monthly_expenses

* computes a **deterministic financial plan** (:func:`build_plan`) that turns the
  verified figures into concrete, checkable recommendations: keep an emergency
  reserve instead of draining savings, prioritize high-interest debt, allocate
  the *real* monthly surplus, estimate debt-payoff timelines using the stated
  APR, and defer investing until the debt and emergency priorities are handled;
* builds a **reasoning brief** for the model that separates facts, calculations,
  the computed plan, and assumptions (no invented facts, no arbitrary
  percentage or expense cuts, no lifestyle changes the user never requested);
* corrects the model's reply: removes echoed user text, duplicated lines and
  section headers, unverified calculation lines, invented expense/lifestyle
  changes and unrealistic promises, then inserts the verified summary + computed
  plan exactly once.

Everything here is deterministic and hardcodes no example values; it works with
arbitrary financial input. When a complete budget cannot be parsed, the reply is
returned with only light de-duplication (there is nothing to verify).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Amount handling
# ---------------------------------------------------------------------------
_PAIR_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9 &'()/%-]{1,40}?)[ \t]*[:=][ \t]*\$?[ \t]*([\d,]+(?:\.\d{1,2})?)\b"
)
_ANNUAL_HINT_RE = re.compile(r"\b(year|annual(?:ly)?|per year|a year)\b", re.IGNORECASE)
_MONTHLY_HINT_RE = re.compile(
    r"(/month|/mo|monthly|per month|each month|a month|every month)", re.IGNORECASE
)
_APR_RE = re.compile(r"\b(\d{1,3}(?:\.\d{1,2})?)\s*%", re.IGNORECASE)

_INCOME_LABELS = frozenset({
    "income", "monthly income", "net income", "gross income", "salary",
    "monthly salary", "wages", "take-home", "take home pay", "earnings",
    "pay", "monthly pay", "paycheck",
})
_DEBT_LABELS = frozenset({
    "debt", "debt payment", "debt payments", "loan", "loans", "loan payment",
    "loan payments", "credit card", "credit card debt", "credit cards",
    "credit card payment", "credit card payments", "minimum payment",
    "minimum payments", "mortgage", "student loan", "student loans",
    "car loan", "car payment", "car payments", "balance", "balances",
})
_SAVING_LABELS = frozenset({
    "savings", "saving", "savings balance", "savings contribution",
    "savings contributions", "retirement contribution", "contribution",
    "contributions", "investments", "investment", "401k", "ira",
    "emergency fund", "savings rate",
})

# Prose fallbacks for income when the "Label: $amount" block format is not used.
_INCOME_PHRASE_RE = re.compile(
    r"\b(?:my\s+|our\s+|i\s+(?:make|earn|bring in|take home)\s+)?"
    r"(?:income|salary|wages|take[- ]home pay|paycheck)\b"
    r"[^\d$]{0,40}?\$?\s*([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_INCOME_VERB_RE = re.compile(
    r"\b(?:i|we)\s+(?:make|earn|bring in|take home)\s+"
    r"\$?\s*([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

# "I have $X" statements without a Label: $amount block (e.g. an available lump
# sum or budget). The negative lookahead avoids mistaking "credit/debt/loan..."
# balances for available savings.
_HAVE_FUNDS_RE = re.compile(
    r"\b(?:i|we)\s+have\s+\$?\s*([\d,]+(?:\.\d{1,2})?)(?![\d,])"
    r"(?!\s*(?:of|in|on|for)?\s*(?:a|an|some)?\s*(?:credit|debt|loan|mortgage|"
    r"card(?:s)?|balance|owed|owing)\b)",
    re.IGNORECASE,
)

# Goal detection.
_GOAL_HEADER_RE = re.compile(
    r"^\s*(?:my\s+)?(?:goals?|what i want(?: to accomplish)?):?\s*$", re.IGNORECASE
)
_GOAL_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s*)(.+)$")
_GOAL_MARKER_RE = re.compile(
    r"\b(pay off|pay down|build|invest(?:ing)?|save (?:up )?for|become|buy|"
    r"retire|get out of debt|debt[- ]free|emergency fund|reduce debt|reach|"
    r"increase|create|set up|start)\b",
    re.IGNORECASE,
)


def _parse_amount(raw: str) -> Optional[float]:
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return None
    if value < 0 or value > 1_000_000_000:
        return None
    return value


def format_amount(value: float) -> str:
    """Format a number as a currency string (no hardcoded example values)."""
    value = round(float(value), 2)
    sign = "-" if value < 0 else ""
    value = abs(value)
    whole = int(value)
    cents = round((value - whole) * 100)
    if cents == 100:
        whole += 1
        cents = 0
    text = f"{whole:,}" if cents == 0 else f"{whole:,}.{cents:02d}"
    return f"{sign}${text}"


def _to_monthly(text: str, after_pos: int, amount: float) -> float:
    window = text[after_pos:after_pos + 80]
    if _ANNUAL_HINT_RE.search(window):
        return amount / 12.0
    return amount


def _window(text: str, start: int, end: int, padding: int = 40) -> str:
    return text[max(0, start - padding):end + padding]


def _normalize_label(label: str) -> str:
    norm = label.strip().lower()
    norm = re.sub(r"\s*[/(](monthly|per month|/month|/mo|month)\)?\s*$", "", norm)
    norm = re.sub(r"\s+", " ", norm)
    return norm.strip(" :\t-")


def _is_income(norm: str) -> bool:
    return any(norm == lbl or norm.endswith(" " + lbl) for lbl in _INCOME_LABELS)


def _is_debt(norm: str) -> bool:
    return any(norm == lbl or norm.endswith(" " + lbl) for lbl in _DEBT_LABELS)


def _is_saving(norm: str) -> bool:
    return any(norm == lbl or norm.endswith(" " + lbl) for lbl in _SAVING_LABELS)


def _has_amount(line: str) -> bool:
    return "$" in line or bool(_PAIR_RE.search(line))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
@dataclass
class ExpenseItem:
    """A single labeled expense amount extracted from the user's message."""

    category: str
    amount: float


@dataclass
class DebtItem:
    """A debt balance, optionally with an interest rate."""

    description: str
    amount: float
    apr: Optional[float] = None


@dataclass
class FinancialSummary:
    """The user's stated financial figures, verified by deterministic parsing."""

    income: Optional[float] = None
    expenses: List[ExpenseItem] = field(default_factory=list)
    savings_balance: Optional[float] = None
    savings_contributions: List[float] = field(default_factory=list)
    debts: List[DebtItem] = field(default_factory=list)
    debt_payments: List[float] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)

    @property
    def total_monthly_expenses(self) -> float:
        return round(sum(item.amount for item in self.expenses), 2)

    @property
    def monthly_surplus(self) -> Optional[float]:
        """``monthly_income - total_monthly_expenses`` (None when not computable)."""
        if self.income is None or not self.expenses:
            return None
        return round(self.income - self.total_monthly_expenses, 2)

    @property
    def is_complete_budget(self) -> bool:
        """True when income and at least one expense were parsed."""
        return self.income is not None and bool(self.expenses)

    def _summary_lines(self) -> List[str]:
        lines = [
            f"- Income: {format_amount(self.income)}",
            f"- Total monthly expenses: {format_amount(self.total_monthly_expenses)}",
        ]
        if self.savings_balance is not None:
            lines.append(f"- Savings: {format_amount(self.savings_balance)}")
        if self.savings_contributions:
            lines.append(
                f"- Monthly savings contributions: {format_amount(sum(self.savings_contributions))}"
            )
        for debt in self.debts:
            line = f"- {debt.description}: {format_amount(debt.amount)}"
            if debt.apr is not None:
                line += f" at {debt.apr:g}% APR"
            lines.append(line)
        if self.debt_payments:
            lines.append(f"- Monthly debt payments: {format_amount(sum(self.debt_payments))}")
        if self.monthly_surplus is not None:
            lines.append(f"- Monthly surplus: {format_amount(self.monthly_surplus)}")
        return lines

    def verified_block(self) -> Optional[str]:
        """A clean, verified summary for the final reply (None if not computable)."""
        if not self.is_complete_budget:
            return None
        return "Verified monthly summary:\n" + "\n".join(self._summary_lines())


# ---------------------------------------------------------------------------
# Deterministic financial plan
# ---------------------------------------------------------------------------
# Advisory defaults. Each one is a stated *assumption* in the plan text (never a
# fact invented about the user).
HIGH_INTEREST_APR_THRESHOLD = 10.0  # APR at/above this is treated as high-interest
EMERGENCY_FUND_MONTHS = 3           # months of expenses targeted for the emergency fund
NEAR_TERM_BUFFER_MONTHS = 1         # minimum reserve kept instead of drained to debt
DEBT_SURPLUS_SHARE = 0.7            # share of surplus to high-interest debt while it remains


@dataclass
class FinancialPlan:
    """A deterministic, number-driven recommendation computed from a summary.

    Only the user's own figures are used; every "default" is surfaced as an
    explicit assumption. No expense category is ever cut and no percentage is
    applied to spending.
    """

    has_budget: bool = False
    emergency_fund_months: float = EMERGENCY_FUND_MONTHS
    emergency_fund_target: Optional[float] = None
    emergency_fund_gap: Optional[float] = None
    near_term_buffer: Optional[float] = None
    high_interest_threshold_apr: float = HIGH_INTEREST_APR_THRESHOLD
    high_interest_debts: List[DebtItem] = field(default_factory=list)
    other_debts: List[DebtItem] = field(default_factory=list)
    high_interest_debt_total: float = 0.0
    other_debt_total: float = 0.0
    available_lump_sum: float = 0.0
    retained_buffer: Optional[float] = None
    recommended_initial_payment: float = 0.0
    remaining_high_interest_debt: float = 0.0
    recommended_monthly_debt_payment: float = 0.0
    recommended_monthly_savings: float = 0.0
    recommended_monthly_investing: float = 0.0
    estimated_months_to_debt_free: Optional[int] = None
    estimated_months_to_emergency_target: Optional[int] = None


def _months_to_pay_off(
    balance: float, apr: Optional[float], monthly_payment: float
) -> Optional[int]:
    """Estimate months to clear a balance at an APR with a fixed monthly payment."""
    if balance <= 0:
        return 0
    if apr is None or monthly_payment is None or monthly_payment <= 0:
        return None
    monthly_rate = apr / 100.0 / 12.0
    if monthly_payment <= balance * monthly_rate:
        return None  # payment does not even cover monthly interest
    n = -math.log(1 - balance * monthly_rate / monthly_payment) / math.log(1 + monthly_rate)
    return math.ceil(n)


def build_plan(summary: FinancialSummary) -> FinancialPlan:
    """Compute a concrete recommendation plan from verified figures.

    Order of priorities (stated in the plan text):
      1. keep a near-term emergency buffer (never drain all savings);
      2. attack high-interest (APR >= threshold) debt with available savings and
         the monthly surplus;
      3. build the emergency fund to the target from the surplus;
      4. only then discuss investing.

    No expense is ever cut and no arbitrary percentage is applied to spending -
    recommendations allocate only the verified monthly surplus and the existing
    savings balance.
    """
    plan = FinancialPlan(has_budget=summary.is_complete_budget)
    if not summary.is_complete_budget:
        return plan

    monthly_expenses = summary.total_monthly_expenses
    surplus = summary.monthly_surplus
    savings = summary.savings_balance if summary.savings_balance is not None else 0.0

    plan.emergency_fund_target = round(EMERGENCY_FUND_MONTHS * monthly_expenses, 2)
    plan.emergency_fund_gap = round(max(0.0, plan.emergency_fund_target - savings), 2)
    plan.near_term_buffer = round(NEAR_TERM_BUFFER_MONTHS * monthly_expenses, 2)

    for debt in summary.debts:
        if debt.apr is not None and debt.apr >= plan.high_interest_threshold_apr:
            plan.high_interest_debts.append(debt)
        else:
            plan.other_debts.append(debt)
    plan.high_interest_debt_total = round(
        sum(item.amount for item in plan.high_interest_debts), 2
    )
    plan.other_debt_total = round(sum(item.amount for item in plan.other_debts), 2)

    if savings > plan.near_term_buffer and plan.high_interest_debt_total > 0:
        plan.available_lump_sum = round(savings - plan.near_term_buffer, 2)
        plan.recommended_initial_payment = round(
            min(plan.available_lump_sum, plan.high_interest_debt_total), 2
        )
    plan.retained_buffer = round(savings - plan.recommended_initial_payment, 2)
    plan.remaining_high_interest_debt = round(
        plan.high_interest_debt_total - plan.recommended_initial_payment, 2
    )

    buffer_gap = round(max(0.0, plan.near_term_buffer - plan.retained_buffer), 2)
    ef_gap_after_reserve = round(
        max(0.0, plan.emergency_fund_target - plan.retained_buffer), 2
    )

    if surplus is None or surplus <= 0:
        return plan

    saved_during_debt_phase = 0.0

    if buffer_gap > 0:
        # No near-term cushion yet: top up the buffer before anything else.
        plan.recommended_monthly_savings = round(min(surplus, buffer_gap), 2)
        if plan.remaining_high_interest_debt > 0:
            plan.recommended_monthly_debt_payment = round(
                surplus - plan.recommended_monthly_savings, 2
            )
    elif plan.remaining_high_interest_debt > 0:
        apr = max((d.apr or 0.0) for d in plan.high_interest_debts)
        plan.recommended_monthly_debt_payment = round(surplus * DEBT_SURPLUS_SHARE, 2)
        plan.recommended_monthly_savings = round(
            surplus - plan.recommended_monthly_debt_payment, 2
        )
        months = _months_to_pay_off(
            plan.remaining_high_interest_debt, apr, plan.recommended_monthly_debt_payment
        )
        plan.estimated_months_to_debt_free = months
        if months is not None:
            saved_during_debt_phase = round(
                months * plan.recommended_monthly_savings, 2
            )
    elif plan.other_debt_total > 0 or ef_gap_after_reserve > 0:
        plan.recommended_monthly_savings = surplus
        if ef_gap_after_reserve > 0:
            plan.estimated_months_to_emergency_target = math.ceil(
                ef_gap_after_reserve / surplus
            )
    else:
        plan.recommended_monthly_investing = surplus

    if ef_gap_after_reserve > 0:
        remaining = round(max(0.0, ef_gap_after_reserve - saved_during_debt_phase), 2)
        if remaining > 0:
            months_after_debt = math.ceil(remaining / surplus)
            if plan.estimated_months_to_debt_free is not None:
                plan.estimated_months_to_emergency_target = (
                    plan.estimated_months_to_debt_free + months_after_debt
                )
            else:
                plan.estimated_months_to_emergency_target = months_after_debt

    return plan


def _debt_label(items: List[DebtItem]) -> str:
    parts = []
    for item in items:
        text = f"{item.description} {format_amount(item.amount)}"
        if item.apr is not None:
            text += f" ({item.apr:g}% APR)"
        parts.append(text)
    return "; ".join(parts)


def _months_text(n: Optional[int]) -> str:
    if n is None:
        return "an unknown number of"
    return f"about {n}"


def render_plan_text(plan: FinancialPlan, summary: FinancialSummary) -> str:
    """Render the computed plan as prose for the model's reasoning brief.

    Every number comes from :func:`build_plan`; assumptions are called out
    explicitly instead of being presented as facts about the user.
    """
    if not plan.has_budget:
        return ""

    lines: List[str] = []
    savings = summary.savings_balance if summary.savings_balance is not None else 0.0

    if summary.savings_balance is None:
        lines.append(
            "- Assumption: no savings balance was stated, so this plan assumes $0 saved "
            "while you build a first buffer."
        )
    lines.append(
        "- Assumption: emergency-fund target of "
        f"{format_amount(plan.emergency_fund_target)} "
        f"({plan.emergency_fund_months:g} months of expenses). Adjust if the user states a "
        "different target."
    )

    if plan.high_interest_debts:
        lines.append(
            f"- High-interest debt (APR >= {plan.high_interest_threshold_apr:g}%): "
            f"{_debt_label(plan.high_interest_debts)}."
        )
    elif summary.debts:
        lines.append(f"- Debt: {_debt_label(list(summary.debts))}.")

    if plan.recommended_initial_payment > 0:
        lines.append(
            f"- Recommendation: pay {format_amount(plan.recommended_initial_payment)} "
            f"of the {format_amount(savings)} savings balance toward high-interest debt now, "
            "keeping "
            f"{format_amount(plan.retained_buffer)} in savings as a reserve (assumption: "
            f"a minimum reserve of {format_amount(plan.near_term_buffer)}, "
            f"{NEAR_TERM_BUFFER_MONTHS:g} month of expenses)."
        )

    if plan.remaining_high_interest_debt > 0:
        line = (
            f"- Recommendation: from the {format_amount(summary.monthly_surplus)} monthly "
            "surplus, pay "
            f"{format_amount(plan.recommended_monthly_debt_payment)}/month toward the "
            f"remaining high-interest debt"
        )
        if plan.recommended_monthly_savings > 0:
            line += (
                f" and add {format_amount(plan.recommended_monthly_savings)}/month to savings"
            )
        apr = max((d.apr or 0.0) for d in plan.high_interest_debts)
        line += (
            f". Estimated debt-free in {_months_text(plan.estimated_months_to_debt_free)}"
            f" month(s) accounting for {apr:g}% APR."
        )
        lines.append(line)
    elif plan.recommended_monthly_savings > 0:
        lines.append(
            f"- Recommendation: add all {format_amount(summary.monthly_surplus)} of the "
            "monthly surplus to savings."
        )
    elif plan.recommended_monthly_investing > 0:
        lines.append(
            f"- Recommendation: with the emergency fund in place, direct the "
            f"{format_amount(summary.monthly_surplus)} monthly surplus toward investing."
        )

    if plan.estimated_months_to_emergency_target is not None:
        lines.append(
            f"- Estimated emergency fund reaches {format_amount(plan.emergency_fund_target)} "
            f"in {_months_text(plan.estimated_months_to_emergency_target)} month(s) total, "
            "assuming the surplus stays constant."
        )
    elif plan.emergency_fund_gap and plan.emergency_fund_gap > 0:
        lines.append(
            f"- Emergency-fund gap: {format_amount(plan.emergency_fund_gap)} "
            "(target minus current savings)."
        )

    lines.append(
        "- Investing: discuss only after high-interest debt is paid and the emergency "
        "fund target is reached; acknowledge the exact order can depend on circumstances."
    )
    return "\n".join(lines)


def _extract_goals(text: str, summary: FinancialSummary) -> None:
    in_goals_section = False
    for raw in text.splitlines():
        line = raw.strip()
        if _GOAL_HEADER_RE.match(line):
            in_goals_section = True
            continue

        item = _GOAL_ITEM_RE.match(line)
        if in_goals_section:
            if item and not _has_amount(line):
                goal = item.group(1).strip(" .:-")
                if goal and goal not in summary.goals:
                    summary.goals.append(goal)
            continue

        # Standalone goals (no "Goals:" header).
        if _has_amount(line):
            continue
        if item:
            goal = item.group(1).strip(" .:-")
            if goal and _GOAL_MARKER_RE.search(goal) and goal not in summary.goals:
                summary.goals.append(goal)
        elif len(line.split()) >= 3 and _GOAL_MARKER_RE.search(line):
            goal = line.strip(" .:-")
            if goal not in summary.goals:
                summary.goals.append(goal)


def extract_financials(text: str) -> FinancialSummary:
    """Parse income, expenses, savings, debt (with APR), and goals from text."""
    summary = FinancialSummary()
    if not text or not text.strip():
        return summary

    seen_categories = set()

    def add_expense(label: str, amount: float) -> None:
        key = _normalize_label(label)
        if not key or key in seen_categories:
            return
        seen_categories.add(key)
        summary.expenses.append(ExpenseItem(category=label.strip(), amount=round(amount, 2)))

    for match in _PAIR_RE.finditer(text):
        amount = _parse_amount(match.group(2))
        if amount is None:
            continue
        norm = _normalize_label(match.group(1))
        window = _window(text, match.start(), match.end())

        if _is_income(norm):
            value = round(_to_monthly(text, match.end(), amount), 2)
            if summary.income is None or value > summary.income:
                summary.income = value
        elif _is_saving(norm):
            if _MONTHLY_HINT_RE.search(window):
                summary.savings_contributions.append(round(amount, 2))
            elif summary.savings_balance is None or amount > summary.savings_balance:
                summary.savings_balance = round(amount, 2)
        elif _is_debt(norm):
            if "payment" in norm or _MONTHLY_HINT_RE.search(window):
                summary.debt_payments.append(round(amount, 2))
            else:
                apr_match = _APR_RE.search(window)
                summary.debts.append(
                    DebtItem(
                        description=match.group(1).strip(),
                        amount=round(amount, 2),
                        apr=float(apr_match.group(1)) if apr_match else None,
                    )
                )
        else:
            add_expense(match.group(1), amount)

    if summary.income is None:
        for regex in (_INCOME_PHRASE_RE, _INCOME_VERB_RE):
            for match in regex.finditer(text):
                amount = _parse_amount(match.group(1))
                if amount is None:
                    continue
                value = round(_to_monthly(text, match.end(), amount), 2)
                if summary.income is None or value > summary.income:
                    summary.income = value

    # Available funds stated as "I have $X" (no labeled block) count as savings
    # when nothing more specific was parsed.
    if summary.savings_balance is None:
        for match in _HAVE_FUNDS_RE.finditer(text):
            amount = _parse_amount(match.group(1))
            if amount is None:
                continue
            if summary.savings_balance is None or amount > summary.savings_balance:
                summary.savings_balance = round(amount, 2)

    _extract_goals(text, summary)
    return summary


# ---------------------------------------------------------------------------
# Reasoning brief (system-prompt context)
# ---------------------------------------------------------------------------
_REASONING_RULES = """Reasoning rules you must follow:
- Base every recommendation only on the verified facts and the computed plan below. Never invent facts about the user's lifestyle, housing, job, family, location, or ability to make changes. Never assume the user can move, get a roommate, change jobs, or cut any specific expense without evidence.
- Clearly distinguish: facts provided by the user, calculations, the computed plan, your recommendations, and assumptions. State every assumption explicitly instead of inventing facts.
- Use the exact amounts from the "Recommended plan" section for the monthly allocation and the debt/emergency-fund strategy. Do not propose different income, expense, surplus, savings, or debt figures.
- Do not propose arbitrary percentage reductions (e.g. "cut food by 20%") or invent savings or expense reductions unless there is a concrete, explained reason. No specific expense cut is recommended by this plan.
- Prioritize high-interest debt: high-APR debt (e.g. a 22% credit card) should be a major priority, typically before investing.
- Keep an emergency reserve: do not recommend spending the entire savings balance. Use the computed amount toward urgent high-interest debt while retaining a buffer, and say so.
- Use the calculated monthly surplus as the basis for the monthly allocation. Never use a different surplus figure.
- Never make unrealistic promises about becoming debt-free. Give realistic estimates and clearly label them as estimates.
- When estimating debt-payoff timelines, account for the stated APR/interest.
- Discuss investing only after addressing high-interest debt and an emergency fund, while acknowledging the exact order can depend on the user's situation.
- If important information is missing (e.g. emergency-fund target, risk tolerance, time horizon), state the assumption you are making rather than inventing facts."""

_RESPONSE_STRUCTURE = """Structure your answer using these sections:
1. Financial situation
2. What I would prioritize
3. Recommended monthly allocation
4. Debt repayment strategy
5. Emergency fund strategy
6. Spending adjustments
7. Investing strategy
8. 6-month action plan
9. Why this plan makes sense

Keep the response concise but useful, specific to the user's actual numbers, and professional - like a personal-finance assistant, not a calculator. Do not echo the user's original prompt. A verified monthly summary of the key figures will be shown with your answer; reference it instead of repeating every number."""


def finance_context(summary: FinancialSummary) -> Optional[str]:
    """A system-prompt brief: verified facts, calculations, and reasoning rules.

    Returns None when there is no complete budget to verify.
    """
    if not summary.is_complete_budget:
        return None

    facts = [f"- Monthly income: {format_amount(summary.income)}"]
    for item in summary.expenses:
        facts.append(f"- {item.category}: {format_amount(item.amount)}/month")
    facts.append(f"- Total monthly expenses: {format_amount(summary.total_monthly_expenses)}")
    if summary.savings_balance is not None:
        facts.append(f"- Savings balance: {format_amount(summary.savings_balance)}")
    if summary.savings_contributions:
        facts.append(
            f"- Monthly savings contributions: {format_amount(sum(summary.savings_contributions))}"
        )
    for debt in summary.debts:
        line = f"- {debt.description}: {format_amount(debt.amount)}"
        if debt.apr is not None:
            line += f" ({debt.apr:g}% APR)"
        facts.append(line)
    if summary.debt_payments:
        facts.append(f"- Monthly debt payments: {format_amount(sum(summary.debt_payments))}")
    if summary.goals:
        facts.append("- Goals: " + "; ".join(summary.goals))

    calculation = (
        "- Monthly surplus = monthly income - total monthly expenses = "
        f"{format_amount(summary.income)} - {format_amount(summary.total_monthly_expenses)} "
        f"= {format_amount(summary.monthly_surplus)}"
    )

    plan = build_plan(summary)
    plan_text = render_plan_text(plan, summary)

    return (
        "The user provided the following financial information, already verified by "
        "the system. Use exactly these figures and do not recompute or restate them "
        "in a different way.\n\n"
        "Facts provided by the user:\n" + "\n".join(facts) +
        "\n\nCalculations:\n" + calculation +
        (("\n\nRecommended plan (computed from the verified numbers; use these exact "
          "numbers):\n" + plan_text) if plan_text else "") +
        "\n\n" + _REASONING_RULES +
        "\n\n" + _RESPONSE_STRUCTURE
    )


# ---------------------------------------------------------------------------
# Reply correction
# ---------------------------------------------------------------------------
_MATH_LINE_RE = re.compile(
    r"\$\s*[\d,]+(?:\.\d+)?\s*[-−–]\s*\$?\s*[\d,]+(?:\.\d+)?"
    r"\s*=\s*[-−–]?\$?\s*[\d,]+(?:\.\d+)?"
)
_SUMMARY_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?"
    r"(?:income|(?:total\s+)?(?:monthly\s+)?(?:expense|expenses|expenditure|spending)"
    r"|(?:monthly\s+)?surplus|left\s*over|leftover|remaining|balance|savings|debt\s*payments?)"
    r"\s*[:=]?\s*\$?\s*[-−–]?\s*[\d,]+",
    re.IGNORECASE,
)
# Repeated section headers such as "Monthly Income and Expenses 1".
_SECTION_HEADER_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?"
    r"(?:monthly\s+)?(?:income|budget|expense(?:s)?)"
    r"(?:\s+and\s+(?:expense(?:s)?|income))?"
    r"(?:\s+(?:overview|summary|breakdown|details))?\b",
    re.IGNORECASE,
)
# Lifestyle changes the user never requested or indicated were possible.
_LIFESTYLE_INVENTION_RE = re.compile(
    r"\b(roommate|move (?:out|in|to|into)|relocate|downsize|change (?:your )?(?:job|career)|"
    r"get a (?:better|new|second|part[- ]time) job|side hustle|"
    r"ask (?:for|your employer for) a raise|sell (?:your|the) (?:car|home|house))\b",
    re.IGNORECASE,
)
# Arbitrary percentage-based expense reductions ("cut food by 20%").
_ARBITRARY_PERCENT_CUT_RE = re.compile(
    r"\b(cut|reduce|lower|trim|slash|decrease|drop)\b[^.!?\n]{0,60}?\b\d{1,3}\s*%\b",
    re.IGNORECASE,
)
# Invented cuts to specific expense categories (with or without a number).
_INVENTED_EXPENSE_CUT_RE = re.compile(
    r"\b(cut|reduce|reduction|reducing|cutting|lower|trim|decrease|drop)\b"
    r"[^.!?\n]{0,40}?\b"
    r"(rent|utilities|utility bills?|housing|groceries|grocery|food|entertainment|"
    r"transport(?:ation)?(?: costs?)?|commute|dining|eating out|subscriptions?|"
    r"expenses?|spending)\b",
    re.IGNORECASE,
)
# Unrealistic same-month / impossible debt-free promises.
_UNREALISTIC_DEBT_FREE_RE = re.compile(
    r"\b(pay off (?:your |the )?(?:entire |all(?: of)? |whole |full )?|debt[- ]?free)"
    r"[^.!?\n]{0,30}?\b(this month|next month|within a month|in (?:a )?single month|"
    r"immediately|overnight)\b",
    re.IGNORECASE,
)
# The nine response-structure section headers (optionally numbered).
_SECTION_TITLE_RE = re.compile(
    r"^\s*(?:\d+[.)]\s*)?"
    r"(financial situation|what i would prioritize|recommended monthly allocation|"
    r"debt repayment strategy|emergency fund strategy|spending adjustments|"
    r"investing strategy|6[- ]?month action plan|why this plan makes sense)"
    r"\s*:?\s*$",
    re.IGNORECASE,
)


def _split_lines(text: str) -> List[str]:
    return [line.rstrip() for line in (text or "").splitlines()]


def _join_lines(lines: List[str]) -> str:
    out: List[str] = []
    prev_blank = True
    for line in lines:
        if not line.strip():
            if not prev_blank:
                out.append("")
            prev_blank = True
            continue
        out.append(line)
        prev_blank = False
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out).strip()


def _normalize_for_dedupe(line: str) -> str:
    text = (line or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b\d{1,2}\s*$", "", text)  # trailing section number ("... 1")
    text = re.sub(r"^[-*•·\d.)\s]+", "", text)  # leading bullets / list numbers
    return text.strip(" .:")


def _remove_echoed_lines(lines: List[str], user_message: str) -> List[str]:
    """Drop reply lines that are near-verbatim copies of the user's message."""
    user_norms = {
        _normalize_for_dedupe(line)
        for line in (user_message or "").splitlines()
        if len(_normalize_for_dedupe(line)) >= 4
    }
    out = []
    for line in lines:
        if line.strip() and _normalize_for_dedupe(line) in user_norms:
            continue
        out.append(line)
    return out


def _dedupe_lines(lines: List[str]) -> List[str]:
    """Remove duplicate lines and repeated section headers (keep first)."""
    seen = set()
    out = []
    for line in lines:
        norm = _normalize_for_dedupe(line)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(line)
    return out


def _strip_calculation_lines(lines: List[str]) -> List[str]:
    """Remove the model's own (untrusted) math and income/expense/surplus totals."""
    out = []
    for line in lines:
        if (
            _MATH_LINE_RE.search(line)
            or _SUMMARY_LINE_RE.match(line)
            or _SECTION_HEADER_RE.match(line)
        ):
            continue
        out.append(line)
    return out


def _strip_invented_changes(lines: List[str]) -> List[str]:
    """Drop lines that invent lifestyle changes, expense cuts, or false promises.

    These are the failure modes the user reported: "reduce rent and utilities",
    "use a roommate", random percentage reductions, "$X net reduction" math, and
    promises like "pay off your credit card this month". None of them are
    supported by the verified figures, so they are removed outright.
    """
    out = []
    for line in lines:
        if (
            _LIFESTYLE_INVENTION_RE.search(line)
            or _ARBITRARY_PERCENT_CUT_RE.search(line)
            or _INVENTED_EXPENSE_CUT_RE.search(line)
            or _UNREALISTIC_DEBT_FREE_RE.search(line)
        ):
            continue
        out.append(line)
    return out


def _clean_sections(lines: List[str]) -> List[str]:
    """Drop empty response sections and renumber the remaining headers 1..N.

    After invented lines are removed a section can be left with a header but no
    body (e.g. "Spending adjustments" when no cuts were recommended). Dropping
    the header and renumbering keeps the reply a single clean, coherent read.
    """
    out: List[str] = []
    count = 0
    for i, line in enumerate(lines):
        match = _SECTION_TITLE_RE.match(line)
        if not match:
            out.append(line)
            continue
        next_non_blank = next(
            (lines[j] for j in range(i + 1, len(lines)) if lines[j].strip()), None
        )
        if next_non_blank is None or _SECTION_TITLE_RE.match(next_non_blank):
            continue  # header with no body -> drop it
        count += 1
        title = match.group(1).strip().rstrip(":")
        out.append(f"{count}. {title}")
    return out


def correct_financial_response(
    reply: str,
    user_message: str,
    summary: Optional[FinancialSummary] = None,
) -> str:
    """Clean and verify a model reply against the user's stated figures.

    * removes echoed user text and duplicated lines/sections;
    * removes invented lifestyle changes, expense cuts, and unrealistic promises;
    * when a complete budget was parsed, strips the model's own (unverified)
      arithmetic/totals and inserts one verified summary block;
    * otherwise returns the cleaned reply unchanged.
    """
    summary = summary or extract_financials(user_message)
    lines = _split_lines(reply)
    lines = _remove_echoed_lines(lines, user_message)
    lines = _dedupe_lines(lines)
    lines = _strip_invented_changes(lines)
    lines = _strip_calculation_lines(lines)
    lines = _clean_sections(lines)

    block = summary.verified_block()
    if block is None:
        return _join_lines(lines)

    return _join_lines([block, ""] + lines)


__all__ = [
    "DEBT_SURPLUS_SHARE",
    "DebtItem",
    "EMERGENCY_FUND_MONTHS",
    "ExpenseItem",
    "FinancialPlan",
    "FinancialSummary",
    "HIGH_INTEREST_APR_THRESHOLD",
    "NEAR_TERM_BUFFER_MONTHS",
    "build_plan",
    "correct_financial_response",
    "extract_financials",
    "finance_context",
    "format_amount",
    "render_plan_text",
]
