"""Regression tests for the "startup $30,000" prompt-example bug.

The exact input ``"I have $30,000. I want to open a startup. What do you
recommend?"`` previously produced the literal text ``"startup-related answer"``
because the system prompt's TOPIC CONTINUITY EXAMPLE block contained bare,
few-word placeholder utterances that the small instruct model parroted verbatim.

These tests assert the root cause is fixed at the source: the example block now
uses realistic, grounded assistant answers, and the exact input flows through
the guard/validator loop as a grounded startup reply referencing the user's
$30,000 budget (no invented $75,000 requirement, no investment drift).
"""

from __future__ import annotations

from src.inference import inference
from src.inference.prompt_templates import ADVISOR_SYSTEM_PROMPT

STARTUP_QUESTION = "I have $30,000. I want to open a startup. What do you recommend?"

GROUNDED_STARTUP_REPLY = (
    "With your $30,000 budget, focus on a low-cost startup such as a "
    "service-based business or a small online store. List expected setup and "
    "running costs before spending, and keep enough cash aside for the first "
    "months of operating expenses."
)


# ---------------------------------------------------------------------------
# Root-cause regression: the system prompt must not contain the placeholder
# scaffold the model used to parrot.
# ---------------------------------------------------------------------------
def test_prompt_example_has_no_bare_placeholder_utterances():
    prompt = ADVISOR_SYSTEM_PROMPT
    assert "### TOPIC CONTINUITY EXAMPLE" in prompt
    assert "startup-related answer" not in prompt
    assert "startup + $30,000 related answer" not in prompt
    assert "$30,000" in prompt  # the grounded example is still present


def test_prompt_example_assistant_replies_are_grounded():
    prompt = ADVISOR_SYSTEM_PROMPT
    example = prompt.split("### TOPIC CONTINUITY EXAMPLE")[1]
    assert "I have $30,000." in example
    assert "business model" in example
    assert "operating costs" in example


# ---------------------------------------------------------------------------
# The exact input is a grounded startup question -> reply passes untouched.
# ---------------------------------------------------------------------------
def test_chat_grounded_startup_reply_for_exact_input(client, monkeypatch):
    captured_prompt = {}

    def gen(user_message, conversation_history=None, **kwargs):
        captured_prompt["sp"] = kwargs.get("system_prompt") or ""
        return GROUNDED_STARTUP_REPLY

    monkeypatch.setattr(inference, "generate_advice", gen)

    resp = client.post("/chat", json={"message": STARTUP_QUESTION})
    assert resp.status_code == 200
    data = resp.json()

    # The model was never shown the old placeholder scaffold.
    prompt = captured_prompt["sp"]
    assert "startup-related answer" not in prompt
    assert "startup + $30,000 related answer" not in prompt

    # The reply stays on the startup topic, uses the user's $30,000, and does
    # not invent a required amount or drift into investments.
    reply = data["reply"]
    assert "startup" in reply.lower()
    assert "$30,000" in reply
    assert "$75,000" not in reply
    assert "return" not in reply.lower()
    assert "related answer" not in reply.lower()

    # A grounded reply must not be regenerated or downgraded to a clarification.
    assert data["advice"] is not None