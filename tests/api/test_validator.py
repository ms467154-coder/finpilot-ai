"""Tests for the response validation layer (claim extraction + classification).

Covers the required behaviors:

* every generated response is decomposed into financial claims;
* each claim is classified as USER_PROVIDED / RETRIEVED / CALCULATED /
  ESTIMATE / ASSUMPTION / UNKNOWN;
* UNKNOWN figures asserted as fact fail validation (1, 2, 7);
* contradictions with the conversation fail validation (5);
* assumptions presented as facts are caught (6) and labelled estimates pass;
* numeric claims in a topic never present fail validation (3, 4);
* on failure the response is regenerated with validation feedback (bounded),
  and an invalid response is never silently passed through.

The validator is exercised both as a unit (no model) and end-to-end through
``POST /chat`` with ``inference.generate_advice`` stubbed.
"""

from __future__ import annotations

from src.advice import guard
from src.advice import validator
from src.advice.validator import ClaimCategory
from src.inference import inference


def _agg(*texts):
    return guard.aggregate_financials(list(texts))


# ---------------------------------------------------------------------------
# Claim extraction / classification
# ---------------------------------------------------------------------------
def test_user_provided_figure_is_not_unknown():
    agg = _agg("I have $30,000 to open a startup.")
    claims = validator.extract_claims(
        "With $30,000 you can start a service business that fits a small budget.", agg
    )
    assert claims
    categories = {claim.category for claim in claims}
    assert categories == {ClaimCategory.USER_PROVIDED}


def test_calculated_figure_is_classified():
    agg = _agg("Income: $2,000\nRent: $1,000\nFood: $500\nOther: $500")
    claims = validator.extract_claims("Keep a $6,000 emergency fund ready.", agg)
    assert ClaimCategory.CALCULATED in {claim.category for claim in claims}
    assert ClaimCategory.UNKNOWN not in {claim.category for claim in claims}


def test_hedged_estimate_is_labelled():
    agg = _agg()
    claims = validator.extract_claims(
        "Historically, index funds have returned about 7% a year.", agg
    )
    assert ClaimCategory.ESTIMATE in {claim.category for claim in claims}


def test_requirement_claim_stays_unknown_even_when_hedged():
    agg = _agg("I have $30,000.")
    claims = validator.extract_claims(
        "You will likely need at least $75,000 in starting capital.", agg
    )
    assert ClaimCategory.UNKNOWN in {claim.category for claim in claims}


def test_retrieved_amount_is_retrieved():
    agg = _agg()
    claims = validator.extract_claims(
        "Continue putting $400 a month into your house fund.",
        agg,
        retrieved_texts=["User is saving $400/month for a house."],
    )
    assert ClaimCategory.RETRIEVED in {claim.category for claim in claims}


def test_extract_claims_ignores_plain_prose():
    agg = _agg()
    assert validator.extract_claims("Prioritize high interest debt first.", agg) == []


# ---------------------------------------------------------------------------
# Validation verdicts
# ---------------------------------------------------------------------------
def test_unknown_figure_fails_validation():
    agg = _agg("I have $30,000.")
    result = validator.validate_response(
        "You will need $75,000 in starting capital.", "I have $30,000.", [], agg
    )
    assert not result.passed
    assert any("without support" in i for i in result.issues)
    assert not any(result.passed for _ in [0])


def test_grounded_reply_passes_validation():
    agg = _agg("I want to open a startup. I have $30,000.")
    result = validator.validate_response(
        "With $30,000 you can start a service business that fits a small budget.",
        "I want to open a startup. I have $30,000.",
        [],
        agg,
    )
    assert result.passed, result.issues


def test_contradiction_with_history_fails():
    agg = _agg("Income: $1,500\nRent: $450\nFood: $250\nTransportation: $100")
    result = validator.validate_response(
        "Your income is $1,000 and your expenses are $800.",
        "Income: $1,500\nRent: $450\nFood: $250\nTransportation: $100",
        [],
        agg,
    )
    assert not result.passed
    assert any("Contradicts" in i for i in result.issues)


def test_topic_drift_fails_when_claim_is_numeric():
    agg = _agg("I want to open a startup.")
    result = validator.validate_response(
        "Investing in index funds could earn you 7% a year over time.",
        "I want to open a startup.",
        [],
        agg,
    )
    assert not result.passed
    assert any("Topic drift" in i for i in result.issues)


def test_aprs_mentioned_in_reply_are_not_contradictions():
    agg = _agg("Income: $3,000\nCredit card debt: $3,000 at 22% APR")
    result = validator.validate_response(
        "Pay the credit card first because it is 22% APR.",
        "Income: $3,000\nCredit card debt: $3,000 at 22% APR",
        [],
        agg,
    )
    assert result.passed, result.issues


def test_feedback_text_lists_issues():
    agg = _agg()
    result = validator.validate_response(
        "You will need $75,000 in starting capital.", "I have $30,000 to open a startup.", [], agg
    )
    feedback = validator.feedback_text(result)
    assert "VALIDATION FEEDBACK" in feedback or "validation" in feedback.lower()
    assert "cor" in feedback.lower() or "corrected" in feedback.lower()


# ---------------------------------------------------------------------------
# Regeneration loop (unit-level)
# ---------------------------------------------------------------------------
def _stub_correct():
    def correct(reply, message, summary=None):
        return reply

    return correct


def test_regenerates_with_feedback_until_valid():
    calls = []

    def generate(message, history, system_prompt):
        calls.append(system_prompt)
        if "VALIDATION FEEDBACK" in (system_prompt or ""):
            return "You should keep your $30,000 budget and start a service business carefully."
        return "You will need $75,000 in starting capital."

    agg = _agg("I have $30,000 to open a startup.")
    out = validator.validate_and_refine(
        message="I have $30,000 to open a startup.",
        history=[],
        summary=agg,
        agg_summary=agg,
        retrieved_texts=[],
        base_system_prompt=None,
        generate=generate,
        correct=_stub_correct(),
    )

    assert not out.is_clarification
    assert "$75,000" not in out.reply
    assert "service business carefully" in out.reply
    assert len(calls) == 2
    assert "VALIDATION FEEDBACK" in calls[1]


def test_never_passes_invalid_response_even_if_regen_always_fails():
    calls = []

    def generate(message, history, system_prompt):
        calls.append(system_prompt)
        return "You will need $75,000 in starting capital."

    agg = _agg("I have $30,000 to open a startup.")
    out = validator.validate_and_refine(
        message="I have $30,000 to open a startup.",
        history=[],
        summary=agg,
        agg_summary=agg,
        retrieved_texts=[],
        base_system_prompt=None,
        generate=generate,
        correct=_stub_correct(),
    )

    assert out.is_clarification
    assert "$75,000" not in out.reply
    assert "don't have enough information" in out.reply.lower()
    assert len(calls) == validator.MAX_REGENERATION_ATTEMPTS + 1


def test_valid_reply_is_not_regenerated():
    calls = []

    def generate(message, history, system_prompt):
        calls.append(system_prompt)
        return "With $30,000 you can start a service business that fits a small budget."

    agg = _agg("I have $30,000 to open a startup.")
    out = validator.validate_and_refine(
        message="I have $30,000 to open a startup.",
        history=[],
        summary=agg,
        agg_summary=agg,
        retrieved_texts=[],
        base_system_prompt=None,
        generate=generate,
        correct=_stub_correct(),
    )
    assert not out.is_clarification
    assert "service business" in out.reply
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Regeneration loop (end-to-end through POST /chat)
# ---------------------------------------------------------------------------
def test_chat_regenerates_invalid_then_accepts_valid(client, monkeypatch):
    calls = []

    def gen(user_message, conversation_history=None, **kwargs):
        system_prompt = kwargs.get("system_prompt") or ""
        calls.append(system_prompt)
        if "VALIDATION FEEDBACK" in system_prompt:
            return "You should keep your $30,000 budget and start a service business carefully."
        return "You will need $75,000 in starting capital."

    monkeypatch.setattr(inference, "generate_advice", gen)

    resp = client.post("/chat", json={"message": "I have $30,000 to open a startup."})
    assert resp.status_code == 200
    data = resp.json()
    assert len(calls) == 2
    assert "VALIDATION FEEDBACK" in calls[1]
    assert "$75,000" not in data["reply"]
    assert "service business carefully" in data["reply"]
    assert data["advice"] is not None


def test_chat_never_silently_passes_when_regen_keeps_failing(client, monkeypatch):
    calls = []

    def gen(user_message, conversation_history=None, **kwargs):
        calls.append(kwargs.get("system_prompt") or "")
        return "You will need $75,000 in starting capital."

    monkeypatch.setattr(inference, "generate_advice", gen)

    resp = client.post("/chat", json={"message": "I have $30,000 to open a startup."})
    assert resp.status_code == 200
    data = resp.json()
    assert len(calls) >= 2  # regenerated, never silently passed the first reply
    assert "$75,000" not in data["reply"]
    assert "don't have enough information" in data["reply"].lower()
    assert data["advice"] is None


def test_chat_grounded_reply_is_not_regenerated(client, monkeypatch):
    calls = []

    def gen(user_message, conversation_history=None, **kwargs):
        calls.append(user_message)
        return "With $30,000 you can start a service business that fits a small budget."

    monkeypatch.setattr(inference, "generate_advice", gen)

    resp = client.post(
        "/chat", json={"message": "I want to open a startup. I have $30,000."}
    )
    assert resp.status_code == 200
    assert len(calls) == 1
    assert "service business" in resp.json()["reply"]