"""Tests for the deterministic financial plan and clean-response guarantees.

Covers the requested scenarios:

1. High-interest debt recommendation
2. Emergency fund reasoning
3. Monthly surplus allocation
4. No invented financial facts
5. No arbitrary percentage reductions
6. No duplicated assistant response
7. No echoed user prompt
8. Clean final API response

The tests exercise :func:`src.advice.finance_calc.build_plan`,
:func:`src.advice.finance_calc.render_plan_text`,
:func:`src.advice.finance_calc.finance_context` and
:func:`src.advice.finance_calc.correct_financial_response` at the unit level and
end-to-end through ``POST /chat`` (inference stubbed via the ``client`` fixture).
"""

from __future__ import annotations

import re

from src.advice import finance_calc
from src.inference import inference

USER_BUDGET = """Income: $3,000/month

Expenses:

* Rent: $900
* Food: $400
* Transportation: $200
* Utilities: $150
* Entertainment: $250
* Other: $200

Savings: $4,000

Credit card debt: $3,000 at 22% APR

Goals:

1. Pay off credit card debt
2. Build an emergency fund
3. Start investing after becoming financially stable
"""


def _summary():
    return finance_calc.extract_financials(USER_BUDGET)


def _plan():
    return finance_calc.build_plan(_summary())


# ---------------------------------------------------------------------------
# 1. High-interest debt recommendation
# ---------------------------------------------------------------------------
def test_high_interest_debt_is_prioritized():
    plan = _plan()
    # The 22% card is classified as high-interest and paid before investing.
    assert plan.high_interest_debt_total == 3000
    assert plan.recommended_initial_payment == 1900
    assert plan.remaining_high_interest_debt == 1100
    assert plan.recommended_monthly_debt_payment > 0
    assert plan.recommended_monthly_debt_payment >= plan.recommended_monthly_savings
    # Investing only appears after debt + emergency fund are handled.
    assert plan.recommended_monthly_investing == 0


def test_plan_context_states_the_high_interest_priority():
    ctx = finance_calc.finance_context(_summary())
    assert "High-interest debt (APR >= 10%): Credit card debt $3,000 (22% APR)" in ctx
    assert "pay $1,900 of the $4,000 savings balance toward high-interest debt" in ctx
    assert "investing" in ctx.lower()


# ---------------------------------------------------------------------------
# 2. Emergency fund reasoning
# ---------------------------------------------------------------------------
def test_emergency_fund_reserve_is_kept():
    plan = _plan()
    assert plan.emergency_fund_target == 6300  # 3 x $2,100 expenses
    assert plan.near_term_buffer == 2100
    # Savings are NOT drained: $2,100 stays behind.
    assert plan.retained_buffer == 2100
    assert plan.recommended_initial_payment < plan.emergency_fund_target


def test_plan_context_explains_the_buffer():
    ctx = finance_calc.finance_context(_summary())
    assert "keeping $2,100 in savings as a reserve" in ctx
    assert "minimum reserve of $2,100" in ctx
    assert "Keep an emergency reserve" in ctx


# ---------------------------------------------------------------------------
# 3. Monthly surplus allocation
# ---------------------------------------------------------------------------
def test_surplus_allocation_uses_the_verified_surplus_exactly():
    plan = _plan()
    plan_text = finance_calc.render_plan_text(plan, _summary())
    assert plan.recommended_monthly_debt_payment == 630
    assert plan.recommended_monthly_savings == 270
    assert plan.recommended_monthly_debt_payment + plan.recommended_monthly_savings == 900
    assert "$900 monthly surplus" in plan_text or "$900" in plan_text


def test_debt_free_estimate_accounts_for_interest():
    plan = _plan()
    assert plan.estimated_months_to_debt_free == 2
    plan_text = finance_calc.render_plan_text(plan, _summary())
    assert "accounting for 22% APR" in plan_text
    assert "about 2 month(s)" in plan_text


def test_no_surplus_means_no_monthly_allocation():
    summary = finance_calc.extract_financials(
        "Income: $2,000\nRent: $2,000\nCredit card debt: $5,000 at 20% APR"
    )
    plan = finance_calc.build_plan(summary)
    assert plan.has_budget
    assert plan.recommended_monthly_debt_payment == 0
    assert plan.recommended_monthly_savings == 0
    assert plan.recommended_monthly_investing == 0


# ---------------------------------------------------------------------------
# 4. No invented financial facts
# ---------------------------------------------------------------------------
def test_plan_never_invents_lifestyle_changes():
    plan_text = finance_calc.render_plan_text(_plan(), _summary())
    for banned in ("roommate", "move to", "change jobs", "side hustle", "ask for a raise"):
        assert banned not in plan_text.lower()


def test_plan_contains_only_user_figures_and_stated_assumptions():
    plan_text = finance_calc.render_plan_text(_plan(), _summary())
    for figure in ("$3,000", "$2,100", "$900", "$4,000", "$3,000 (22% APR)"):
        assert figure in plan_text
    # Assumptions are labeled, never presented as facts.
    assert "Assumption" in plan_text


def test_missing_savings_is_stated_as_an_assumption_not_a_fact():
    summary = finance_calc.extract_financials(
        "Income: $2,000\nRent: $800\nFood: $500\nCredit card debt: $1,000 at 19% APR"
    )
    plan = finance_calc.build_plan(summary)
    plan_text = finance_calc.render_plan_text(plan, summary)
    assert "no savings balance was stated" in plan_text.lower()
    assert "assumes $0 saved" in plan_text.lower()


# ---------------------------------------------------------------------------
# 5. No arbitrary percentage reductions
# ---------------------------------------------------------------------------
def test_plan_never_cuts_expenses_or_uses_percentage_cuts():
    plan_text = finance_calc.render_plan_text(_plan(), _summary())
    assert not re.search(r"\b(cut|reduce|lower|trim|decrease)\b", plan_text, re.IGNORECASE)
    # The only percentages allowed are the APR and the high-interest threshold.
    assert not re.search(r"\b\d{1,3}\s*%\b[^\n]*\b(cut|reduce|lower)\b", plan_text, re.IGNORECASE)
    ctx = finance_calc.finance_context(_summary())
    assert "Do not propose arbitrary percentage reductions" in ctx


def test_correction_strips_invented_cuts_and_lifestyle_changes():
    bad_reply = (
        "Here are some ideas.\n"
        "Find a roommate to split the rent.\n"
        "Cut food by 20% and reduce entertainment by 10%.\n"
        "A reduction of transportation by $50 nets you $55 more.\n"
        "Keep paying the card with $630 a month from your surplus.\n"
    )
    corrected = finance_calc.correct_financial_response(bad_reply, USER_BUDGET)
    for banned in (
        "roommate",
        "Cut food by 20%",
        "reduce entertainment by 10%",
        "nets you $55",
    ):
        assert banned not in corrected
    # Legitimate recommendation survives.
    assert "$630 a month from your surplus" in corrected


def test_correction_strips_unrealistic_debt_free_promises():
    bad = "You can save enough to pay off your credit card this month.\n"
    corrected = finance_calc.correct_financial_response(bad, USER_BUDGET)
    assert "this month" not in corrected


# ---------------------------------------------------------------------------
# 6. No duplicated assistant response
# ---------------------------------------------------------------------------
def test_duplicated_sections_are_collapsed():
    duplicated = (
        "What I would prioritize\n"
        "Pay off the credit card first because it is 22% APR.\n"
        "What I would prioritize\n"
        "Pay off the credit card first because it is 22% APR.\n"
    )
    corrected = finance_calc.correct_financial_response(duplicated, USER_BUDGET)
    assert corrected.count("What I would prioritize") == 1
    assert corrected.count("Pay off the credit card first because it is 22% APR.") == 1


def test_verified_summary_appears_exactly_once():
    duplicated = (
        "Financial situation\n"
        "Your income is $3,000 and your expenses total $2,100.\n"
        "Monthly surplus: $900\n"
        "Financial situation\n"
        "Your income is $3,000 and your expenses total $2,100.\n"
    )
    corrected = finance_calc.correct_financial_response(duplicated, USER_BUDGET)
    assert corrected.count("Verified monthly summary:") == 1
    assert corrected.count("Monthly surplus: $900") == 1


# ---------------------------------------------------------------------------
# 7. No echoed user prompt
# ---------------------------------------------------------------------------
def test_not_echoes_user_prompt():
    reply = (
        "Income: $3,000/month\n"
        "Expenses:\n"
        "* Rent: $900\n"
        "Here is my advice for your situation.\n"
        "Pay the card down using your surplus.\n"
    )
    corrected = finance_calc.correct_financial_response(reply, USER_BUDGET)
    # The prompt's structure and category bullets are not echoed back.
    assert "Income: $3,000/month" not in corrected
    assert "Expenses:" not in corrected
    assert "* Rent: $900" not in corrected
    assert "Here is my advice for your situation." in corrected
    # Verified numbers may legitimately appear once inside the summary, but the
    # prompt itself is never reproduced as a block.
    assert corrected.count("- Savings: $4,000") <= 1


# ---------------------------------------------------------------------------
# 8. Clean final API response
# ---------------------------------------------------------------------------
def test_chat_reply_is_a_single_clean_response(client, monkeypatch):
    """/chat returns one clean reply string with no internal metadata."""

    def messy_generate(user_message, conversation_history=None, **kwargs):
        return (
            "1. Financial situation\n"
            "Income: $3,000 a month and expenses are $2,100.\n"
            "2. What I would prioritize\n"
            "Pay off the credit card because it is 22% APR.\n"
            "3. Spending adjustments\n"
            "Use a roommate.\n"
            "Cut food by 20% and reduce rent.\n"
            "4. What I would prioritize\n"
            "Pay off the credit card because it is 22% APR.\n"
            "Income: $3,000/month\n"
            "Savings: $4,000\n"
        )

    monkeypatch.setattr(inference, "generate_advice", messy_generate)

    resp = client.post("/chat", json={"message": USER_BUDGET})
    assert resp.status_code == 200
    data = resp.json()
    reply = data["reply"]

    assert isinstance(reply, str) and reply.strip()
    assert reply.count("Verified monthly summary:") == 1
    assert reply.count("What I would prioritize") == 1
    # No internal metadata leaked into the reply.
    assert "Re:" not in reply
    assert "Structured advice" not in reply
    assert "Key takeaway" not in reply
    # No invented facts or cuts survived.
    assert "roommate" not in reply
    assert "Cut food by 20%" not in reply
    assert "reduce rent" not in reply
    # The stored advice body is the same single response.
    from src.advice import processor

    assert processor.clean_text(reply) == data["advice"]["full_text"]


def test_chat_reply_mentions_the_computed_plan(client, monkeypatch):
    """The model is handed the deterministic plan so it reasons from it."""

    def plan_following_generate(user_message, conversation_history=None, **kwargs):
        prompt = kwargs.get("system_prompt") or ""
        assert "Recommended plan (computed from the verified numbers" in prompt
        assert "$630/month toward the remaining high-interest debt" in prompt
        return (
            "1. Financial situation\n"
            "Your income is $3,000 and your expenses are $2,100.\n"
            "2. What I would prioritize\n"
            "Pay the credit card first because it is 22% APR.\n"
        )

    monkeypatch.setattr(inference, "generate_advice", plan_following_generate)
    resp = client.post("/chat", json={"message": USER_BUDGET})
    assert resp.status_code == 200
    assert "Verified monthly summary:" in resp.json()["reply"]