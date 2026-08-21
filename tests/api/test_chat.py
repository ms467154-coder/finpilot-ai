"""End-to-end tests for ``POST /chat`` (Phase 12).

Covers the happy path (reply + structured advice + conversation id), multi-turn
continuation, persistence into the store, and input validation. Model inference
is stubbed via the ``client`` fixture (see tests/api/conftest.py).
"""

from __future__ import annotations

from tests.api.conftest import FAKE_REPLY


def test_chat_returns_reply_advice_and_conversation_id(client):
    resp = client.post("/chat", json={"message": "How do I pay off debt?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_id"]
    assert data["reply"] == FAKE_REPLY
    assert data["advice"] is not None
    assert data["advice"]["id"]
    assert data["advice"]["category"]
    assert data["advice"]["full_text"] == FAKE_REPLY


def test_chat_is_multi_turn_with_same_conversation(client):
    first = client.post("/chat", json={"message": "How do I pay off debt?"})
    first_id = first.json()["conversation_id"]

    second = client.post(
        "/chat",
        json={"message": "What about credit card balances?", "conversation_id": first_id},
    )

    assert second.status_code == 200
    assert second.json()["conversation_id"] == first_id

    history = __import__("src.api.store", fromlist=["get_store"]).get_store().get_conversation(first_id)
    assert history is not None
    assert [turn["role"] for turn in history["messages"]] == ["user", "assistant", "user", "assistant"]


def test_chat_persists_structured_advice(client):
    client.post("/chat", json={"message": "How do I pay off debt?"})

    from src.api import store

    items = store.get_store().list_advice()
    assert len(items) == 1
    assert items[0]["full_text"] == FAKE_REPLY
    assert items[0]["conversation_id"]


def test_chat_advice_is_optional(client, monkeypatch):
    """Short, non-actionable replies produce ``advice: null`` (still stored)."""
    import src.inference.inference as inference

    monkeypatch.setattr(inference, "generate_advice", lambda *args, **kwargs: "Got it.")
    resp = client.post("/chat", json={"message": "ok"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "Got it."
    assert data["advice"] is None


def test_chat_rejects_empty_message(client):
    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 422


def test_chat_rejects_missing_message(client):
    resp = client.post("/chat", json={})
    assert resp.status_code == 422
