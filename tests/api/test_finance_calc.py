"""Tests for deterministic financial parsing/correction (response-generation fix).

Covers the required scenarios:

1. Income > expenses   (positive surplus)
2. Income = expenses   (zero surplus)
3. Income < expenses   (negative surplus)
4. Multiple expense categories
5. Debt + savings + expenses together

plus the exact reported bug (wrong model arithmetic, duplicated sections, echoed
user prompt) verified both at the unit level and end-to-end through ``POST /chat``.
"""

from __future__ import annotations

from src.advice import finance_calc
from src.inference import inference

REPORTED_BUDGET = """Income: $1,500/month

Expenses:

* Rent: $450
* Food: $250
* Transportation: $100
* Utilities: $100
* Entertainment: $150
* Other: $100
"""

# The model's buggy reply: wrong arithmetic, duplicated sections, echoed prompt.
BUGGY_REPLY = """Monthly Income and Expenses 1
Income: $1,000
Rent: $450
Food: $250
Transportation: $100
Utilities: $100
Entertainment: $150
Other: $100
$1,000 - $1,500 = -$500/month

Monthly Income and Expenses 2
Income: $1,000
Rent: $450
Food: $250
Transportation: $100
Utilities: $100
Entertainment: $150
Other: $100
$1,000 - $1,500 = -$500/month
"""


# ---------------------------------------------------------------------------
# Parsing / calculation
# ---------------------------------------------------------------------------
def test_reported_budget_parses_correctly():
    summary = finance_calc.extract_financials(REPORTED_BUDGET)
    assert summary.income == 1500
    assert summary.total_monthly_expenses == 1150
    assert summary.monthly_surplus == 350
    assert len(summary.expenses) == 6
    assert {item.category for item in summary.expenses} == {
        "Rent", "Food", "Transportation", "Utilities", "Entertainment", "Other",
    }


def test_income_greater_than_expenses():
    message = "Income: $1,500\nRent: $450\nFood: $250\nTransportation: $100"
    summary = finance_calc.extract_financials(message)
    assert summary.total_monthly_expenses == 800
    assert summary.monthly_surplus == 700
    assert "Monthly surplus: $700" in summary.verified_block()


def test_income_equals_expenses():
    message = "Income: $1,000\nRent: $600\nFood: $250\nOther: $150"
    summary = finance_calc.extract_financials(message)
    assert summary.total_monthly_expenses == 1000
    assert summary.monthly_surplus == 0
    assert "Monthly surplus: $0" in summary.verified_block()


def test_income_less_than_expenses():
    message = "Income: $800\nRent: $600\nFood: $400"
    summary = finance_calc.extract_financials(message)
    assert summary.total_monthly_expenses == 1000
    assert summary.monthly_surplus == -200
    assert "Monthly surplus: -$200" in summary.verified_block()


def test_multiple_expense_categories():
    message = (
        "My income is $3,000. Rent: $900, Food: $400, Transport: $200, "
        "Utilities: $150, Insurance: $100, Entertainment: $120, Other: $130"
    )
    summary = finance_calc.extract_financials(message)
    assert summary.total_monthly_expenses == 2000
    assert summary.monthly_surplus == 1000


def test_debt_savings_and_expenses_together():
    message = """Income: $2,000
Rent: $600
Food: $300
Utilities: $100
Debt payment: $200
Savings: $150
"""
    summary = finance_calc.extract_financials(message)
    assert summary.income == 2000
    assert summary.total_monthly_expenses == 1000  # rent + food + utilities only
    assert summary.debt_payments == [200]
    assert summary.savings_balance == 150
    assert summary.monthly_surplus == 1000
    block = summary.verified_block()
    assert "Income: $2,000" in block
    assert "Total monthly expenses: $1,000" in block
    assert "Monthly debt payments: $200" in block
    assert "Savings: $150" in block
    assert "Monthly surplus: $1,000" in block


def test_monthly_savings_contribution_vs_balance():
    monthly = finance_calc.extract_financials("Income: $2,000\nSavings: $150/month\nRent: $500")
    assert monthly.savings_contributions == [150]
    assert monthly.savings_balance is None

    balance = finance_calc.extract_financials("Income: $2,000\nSavings: $4,000\nRent: $500")
    assert balance.savings_balance == 4000
    assert balance.savings_contributions == []


def test_annual_income_is_converted_to_monthly():
    summary = finance_calc.extract_financials(
        "Income: $18,000/year\nRent: $600\nFood: $400"
    )
    assert summary.income == 1500
    assert summary.monthly_surplus == 500


# ---------------------------------------------------------------------------
# Reply correction
# ---------------------------------------------------------------------------
def test_correct_financial_response_fixes_wrong_math_and_duplicates():
    corrected = finance_calc.correct_financial_response(BUGGY_REPLY, REPORTED_BUDGET)

    assert "Verified monthly summary:" in corrected
    assert "Income: $1,500" in corrected
    assert "Total monthly expenses: $1,150" in corrected
    assert "Monthly surplus: $350" in corrected

    # The wrong calculation and the invented expense total are gone.
    assert "-$500" not in corrected
    assert "Income: $1,000" not in corrected

    # Duplicated section headers are collapsed to at most one occurrence.
    assert corrected.lower().count("monthly income and expenses") <= 1

    # Echoed user prompt lines are removed (the verified block replaces them).
    assert "* Rent: $450" not in corrected
    assert "Income: $1,500/month" not in corrected


def test_correct_financial_response_leaves_plain_replies_unchanged():
    reply = "You should build an emergency fund before investing."
    assert finance_calc.correct_financial_response(reply, "How do I save money?") == reply


def test_no_budget_means_no_verified_block():
    reply = "A budget breakdown can help you track spending."
    corrected = finance_calc.correct_financial_response(reply, "What is a budget?")
    assert "Verified monthly summary:" not in corrected
    assert corrected == reply


def test_format_amount():
    assert finance_calc.format_amount(1500) == "$1,500"
    assert finance_calc.format_amount(1150.5) == "$1,150.50"
    assert finance_calc.format_amount(0) == "$0"
    assert finance_calc.format_amount(-200) == "-$200"


# ---------------------------------------------------------------------------
# Reasoning quality (the brief handed to the model)
# ---------------------------------------------------------------------------
FULL_EXAMPLE = """Income: $3,000/month

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


def _full_summary():
    return finance_calc.extract_financials(FULL_EXAMPLE)


def test_full_example_parses_all_facts():
    summary = _full_summary()
    assert summary.income == 3000
    assert summary.total_monthly_expenses == 2100
    assert summary.monthly_surplus == 900
    assert summary.savings_balance == 4000
    assert len(summary.debts) == 1
    assert summary.debts[0].description == "Credit card debt"
    assert summary.debts[0].amount == 3000
    assert summary.debts[0].apr == 22
    assert summary.goals == [
        "Pay off credit card debt",
        "Build an emergency fund",
        "Start investing after becoming financially stable",
    ]


def test_context_prioritizes_high_interest_debt():
    ctx = finance_calc.finance_context(_full_summary())
    assert "Credit card debt: $3,000 (22% APR)" in ctx
    assert "high-APR debt" in ctx
    assert "major priority" in ctx


def test_context_emergency_fund_reasoning():
    ctx = finance_calc.finance_context(_full_summary())
    assert "Keep an emergency reserve" in ctx
    assert "retaining a buffer" in ctx
    assert "Savings balance: $4,000" in ctx


def test_context_monthly_surplus_allocation():
    ctx = finance_calc.finance_context(_full_summary())
    assert "Monthly surplus = monthly income - total monthly expenses" in ctx
    assert "$3,000 - $2,100 = $900" in ctx
    assert "Use the calculated monthly surplus" in ctx


def test_context_no_invented_facts_and_no_arbitrary_percentages():
    ctx = finance_calc.finance_context(_full_summary())
    assert "Never invent facts" in ctx
    assert "Do not propose arbitrary percentage reductions" in ctx
    # Lifestyle changes are explicitly prohibited, not recommended.
    assert "Never assume the user can move, get a roommate, change jobs" in ctx
    # Only the user's own figures are present - nothing invented.
    for amount in ("$4,000", "$3,000", "$2,100", "$900", "$150", "$250", "$200"):
        assert amount in ctx


def test_context_includes_goals_and_structure():
    ctx = finance_calc.finance_context(_full_summary())
    assert "Goals: Pay off credit card debt; Build an emergency fund" in ctx
    assert "8. 6-month action plan" in ctx
    assert "1. Financial situation" in ctx
    assert "9. Why this plan makes sense" in ctx


def test_corrected_reply_keeps_recommendations_intact():
    summary = _full_summary()
    reply = (
        "Pay off the credit card first because its 22% APR is expensive.\n"
        "Keep about two months of expenses in savings as a buffer.\n"
        "Direct part of your $900 monthly surplus toward the card each month."
    )
    corrected = finance_calc.correct_financial_response(reply, FULL_EXAMPLE, summary=summary)
    assert "Verified monthly summary:" in corrected
    assert "Pay off the credit card first because its 22% APR is expensive." in corrected
    assert "Direct part of your $900 monthly surplus" in corrected


# ---------------------------------------------------------------------------
# Clean single API response (no duplication, no echoed prompt)
# ---------------------------------------------------------------------------
def test_corrected_reply_is_single_clean_response():
    summary = _full_summary()
    duplicated_reply = """Financial situation
Your income is $3,000 and your total expenses are about $2,100.
Monthly surplus: $900

What I would prioritize
Pay off the credit card debt first because the APR is high.

What I would prioritize
Pay off the credit card debt first because the APR is high.

Monthly surplus: $900
"""
    corrected = finance_calc.correct_financial_response(duplicated_reply, FULL_EXAMPLE, summary=summary)
    # Exactly one verified summary, no repeated paragraphs, no stray math.
    assert corrected.count("Verified monthly summary:") == 1
    assert corrected.count("What I would prioritize") == 1
    assert corrected.count("Monthly surplus: $900") == 1
    assert "Re:" not in corrected
    assert "Structured advice" not in corrected
    assert "Key takeaway" not in corrected


def test_chat_corrects_wrong_model_calculation(client, monkeypatch):
    """Even when the model emits bad math, /chat returns verified numbers."""
    seen_kwargs = {}

    def buggy_generate(user_message, conversation_history=None, **kwargs):
        seen_kwargs.update(kwargs)
        return BUGGY_REPLY

    monkeypatch.setattr(inference, "generate_advice", buggy_generate)

    resp = client.post("/chat", json={"message": REPORTED_BUDGET})
    assert resp.status_code == 200
    reply = resp.json()["reply"]

    assert "Income: $1,500" in reply
    assert "Total monthly expenses: $1,150" in reply
    assert "Monthly surplus: $350" in reply
    assert "-$500" not in reply

    # The verified figures are also handed to the model as context.
    assert seen_kwargs.get("system_prompt") is not None
    assert "already verified by the system" in seen_kwargs["system_prompt"]
    assert "$1,150" in seen_kwargs["system_prompt"]


def test_chat_returns_one_clean_response(client, monkeypatch):
    """The /chat reply is a single clean string with no internal metadata."""

    def verbose_generate(user_message, conversation_history=None, **kwargs):
        return (
            "Financial situation\n"
            "Your income is $3,000 a month and your expenses total $2,100.\n"
            "What I would prioritize\n"
            "Pay off the credit card debt first because it is 22% APR.\n"
            "What I would prioritize\n"
            "Pay off the credit card debt first because it is 22% APR.\n"
        )

    monkeypatch.setattr(inference, "generate_advice", verbose_generate)

    resp = client.post("/chat", json={"message": FULL_EXAMPLE})
    assert resp.status_code == 200
    reply = resp.json()["reply"]

    assert reply.count("Verified monthly summary:") == 1
    assert reply.count("What I would prioritize") == 1
    assert "Re:" not in reply
    assert "Structured advice" not in reply
    assert "Key takeaway" not in reply
    assert "$3,000 - $2,100 = $900" not in reply  # model math stripped
    # One clean assistant response string.
    assert isinstance(reply, str) and reply.strip()
    # The advice record's full_text is the same response (whitespace-normalized).
    assert resp.json()["advice"] is not None
    from src.advice import processor

    assert processor.clean_text(reply) == resp.json()["advice"]["full_text"]

