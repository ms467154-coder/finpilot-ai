"""Tests for the conversation-intent / financial-claim safety guard.

Covers the requested behaviors:

1. A startup question with "$30,000" must produce a *startup-related* answer, and
   an invented "$75,000 required capital" claim must never reach the user.
2. ``"why?"`` after a startup message must explain the *previous startup answer*,
   not switch to investment returns.
3. ``"I don't understand."`` must reuse the immediately previous answer.
4. New topics are detected: after an investing answer, ``"I want to open a
   startup."`` is treated as a new topic, not a follow-up.
5. ``"I have $30,000."`` never produces "not enough" / "$75,000 required" claims.
6. The Advice summary / key takeaway derive from the final validated answer.

Unit-level guards live here too (follow-up detection, cross-turn aggregation,
grounding of computed figures, claim rejection), all deterministic with the model
stubbed via the ``client`` fixture.
"""

from __future__ import annotations

from src.advice import finance_calc, guard, processor
from src.inference import inference


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------
def test_follow_up_detection():
    for follow_up in (
        "why?",
        "why",
        "explain",
        "explain more",
        "I don't understand",
        "I don't get it",
        "what do you mean?",
        "what does that mean",
        "how?",
        "how come",
        "can you clarify",
        "why is that",
        "I'm confused",
    ):
        assert guard.is_follow_up(follow_up), follow_up

    for fresh_question in (
        "I want to open a startup.",
        "I have $30,000.",
        "How do I begin investing with a small budget?",
        "What about credit card balances?",
        "How should I budget for a house?",
    ):
        assert not guard.is_follow_up(fresh_question), fresh_question


# ---------------------------------------------------------------------------
# Financial extraction / aggregation
# ---------------------------------------------------------------------------
def test_i_have_parses_as_available_funds():
    summary = finance_calc.extract_financials("I have $30,000 to open a startup.")
    assert summary.savings_balance == 30000


def test_i_have_debt_is_not_treated_as_savings():
    summary = finance_calc.extract_financials("I have $2,000 in credit card debt.")
    assert summary.savings_balance is None


def test_aggregate_merges_funds_across_turns():
    merged = guard.aggregate_financials(
        ["I want to open a startup.", "I have $30,000 to open it."]
    )
    assert merged.savings_balance == 30000
    assert merged.income is None

    merged2 = guard.aggregate_financials(
        ["Income: $2,000\nRent: $500", "Income: $2,500\nRent: $900\nFood: $400"]
    )
    assert merged2.income == 2500
    assert merged2.total_monthly_expenses == 900  # first-seen rent + food


def test_grounded_amounts_include_computed_plan():
    summary = finance_calc.extract_financials(
        "Income: $2,000\nRent: $1,000\nFood: $500\nOther: $500"
    )
    grounded = guard.grounded_amounts(summary)
    assert 2000 in grounded  # income / total expenses
    assert 6000 in grounded  # computed 3-month emergency-fund target


def test_non_grounded_multiple_requires_explicit_months():
    grounded = [2000.0]
    assert guard.is_grounded_number(6000, grounded, "a 3-month emergency fund")
    assert guard.is_grounded_number(6000, grounded, sentence="") is False


# ---------------------------------------------------------------------------
# Conversation brief
# ---------------------------------------------------------------------------
def test_first_message_builds_no_brief():
    brief = guard.build_conversation_brief("How do I pay off debt?", [], guard.aggregate_financials([]))
    assert brief is None


def test_follow_up_brief_points_at_previous_answer():
    history = [
        {"role": "user", "content": "I want to open a startup. I have $30,000."},
        {"role": "assistant", "content": "Here is a realistic startup plan for you."},
    ]
    agg_prior = guard.aggregate_financials(["I want to open a startup. I have $30,000."])
    brief = guard.build_conversation_brief("why is that?", history, agg_prior)

    assert brief is not None
    assert guard._FOLLOW_UP_MARK in brief
    assert "$30,000" in brief
    assert "Here is a realistic startup plan for you." in brief


# ---------------------------------------------------------------------------
# Claim validation (unit)
# ---------------------------------------------------------------------------
def test_guard_rejects_unsupported_required_capital():
    reply = "To start a startup you will need $75,000 in starting capital."
    guarded = guard.guard_final_reply(reply, "I have $30,000 to open a startup.")

    assert guarded.is_clarification
    assert "$75,000" not in guarded.reply
    assert "don't have enough information" in guarded.reply.lower()
    assert "$30,000" in guarded.reply  # grounded figure offered as a starting point


def test_guard_passes_grounded_startup_reply():
    reply = "With $30,000 you can start a service business that fits a small budget."
    guarded = guard.guard_final_reply(reply, "I want to open a startup. I have $30,000.")

    assert not guarded.is_clarification
    assert guarded.reply == reply


def test_guard_blocks_claim_that_budget_is_not_enough():
    reply = "Your $30,000 is not enough to start a startup, so you will need more."
    guarded = guard.guard_final_reply(reply, "I have $30,000.")

    assert guarded.is_clarification
    assert "not enough" not in guarded.reply.lower()
    assert "$75,000" not in guarded.reply


def test_guard_blocks_unsupported_return_claims():
    reply = "Historically, index funds return about 7% a year."
    guarded = guard.guard_final_reply(reply, "How do I begin investing?")
    assert guarded.is_clarification


def test_guard_passes_grounded_percent():
    reply = "Pay off the credit card first because it is 22% APR."
    guarded = guard.guard_final_reply(reply, "Income: $3,000\nCredit card debt: $3,000 at 22% APR")
    assert not guarded.is_clarification


# ---------------------------------------------------------------------------
# End-to-end through POST /chat
# ---------------------------------------------------------------------------
def test_chat_rejects_hallucinated_capital(client, monkeypatch):
    """$75,000 invented capital never reaches the user; no advice is fabricated."""

    def hallucinating(user_message, conversation_history=None, **kwargs):
        return "To open a startup you will need $75,000 in starting capital."

    monkeypatch.setattr(inference, "generate_advice", hallucinating)

    resp = client.post("/chat", json={"message": "I have $30,000 to open a startup."})
    assert resp.status_code == 200
    data = resp.json()
    assert "$75,000" not in data["reply"]
    assert "don't have enough information" in data["reply"].lower()
    assert data["advice"] is None


def test_chat_keeps_grounded_startup_answer(client, monkeypatch):
    grounded = "With $30,000 you can start a service business that fits a small budget."
    monkeypatch.setattr(inference, "generate_advice", lambda *a, **k: grounded)

    resp = client.post("/chat", json={"message": "I want to open a startup. I have $30,000."})
    assert resp.status_code == 200
    assert resp.json()["reply"] == grounded


def test_chat_why_follow_up_stays_on_topic(client, monkeypatch):
    """'why?' after a startup message stays on the startup topic (no ROI drift)."""
    seen = {}

    def follow_up_generate(user_message, conversation_history=None, **kwargs):
        system_prompt = kwargs.get("system_prompt") or ""
        seen["history"] = conversation_history
        seen["system_prompt"] = system_prompt
        if guard._FOLLOW_UP_MARK in system_prompt:
            return (
                "I shouldn't have stated a fixed amount. Startup costs depend on "
                "your model and your expenses."
            )
        return "Here is a realistic startup plan for $30,000."

    monkeypatch.setattr(inference, "generate_advice", follow_up_generate)

    first = client.post("/chat", json={"message": "I want to open a startup. I have $30,000."})
    first_id = first.json()["conversation_id"]

    second = client.post("/chat", json={"message": "why?", "conversation_id": first_id})
    assert second.status_code == 200
    reply = second.json()["reply"]

    # The model saw the prior turns + a follow-up brief pointing at its last answer.
    assert seen["history"] and seen["history"][-1]["role"] == "assistant"
    assert guard._FOLLOW_UP_MARK in seen["system_prompt"]
    assert "Here is a realistic startup plan for $30,000." in seen["system_prompt"]

    # The answer explains the previous startup recommendation, not returns.
    assert "I shouldn't have stated" in reply
    assert "startup" in reply.lower()


def test_chat_understand_follow_up_reuses_previous_answer(client, monkeypatch):
    def gen(user_message, conversation_history=None, **kwargs):
        system_prompt = kwargs.get("system_prompt") or ""
        if guard._FOLLOW_UP_MARK in system_prompt:
            return "I will restate the previous point in a simpler way."
        return "Prioritize the highest interest debt first."

    monkeypatch.setattr(inference, "generate_advice", gen)

    first = client.post("/chat", json={"message": "How should I prioritize my debts?"})
    first_id = first.json()["conversation_id"]

    resp = client.post("/chat", json={"message": "I don't understand.", "conversation_id": first_id})
    assert resp.status_code == 200
    assert "restate the previous point" in resp.json()["reply"]


def test_chat_new_topic_is_not_treated_as_follow_up(client, monkeypatch):
    """After investing advice, a startup question is a NEW topic, not a follow-up."""
    seen = {}

    def gen(user_message, conversation_history=None, **kwargs):
        system_prompt = kwargs.get("system_prompt") or ""
        if "startup" in user_message.lower():
            seen["startup_turn"] = {
                "is_follow_up": guard.is_follow_up(user_message),
                "follow_up_marked": guard._FOLLOW_UP_MARK in system_prompt,
            }
        return "Investing basics" if "investing" in user_message.lower() else "Startup plan"

    monkeypatch.setattr(inference, "generate_advice", gen)

    client.post("/chat", json={"message": "How do I begin investing with a small budget?"})
    resp = client.post("/chat", json={"message": "I want to open a startup."})

    assert resp.status_code == 200
    assert resp.json()["reply"] == "Startup plan"
    assert seen["startup_turn"]["is_follow_up"] is False
    assert seen["startup_turn"]["follow_up_marked"] is False


def test_chat_advice_is_derived_from_final_validated_answer(client, monkeypatch):
    """Summary + key takeaway come from the reply the user actually receives."""
    reply_text = (
        "You should build a $6,000 emergency fund before investing, since your "
        "monthly expenses are $2,000."
    )
    message = "Income: $2,500\nRent: $1,000\nFood: $500\nOther: $500"
    monkeypatch.setattr(inference, "generate_advice", lambda *a, **k: reply_text)

    resp = client.post("/chat", json={"message": message})
    assert resp.status_code == 200
    data = resp.json()
    assert data["advice"] is not None
    advice = data["advice"]

    # The structured advice is literally derived from the final reply string.
    assert advice["full_text"] == processor.clean_text(data["reply"])
    assert advice["key_recommendation"].rstrip("…") in advice["full_text"]
    assert advice["short_title"].rstrip("…") in advice["full_text"]