"""End-to-end tests for the conversation list/history endpoints (Phase 14).

Covers ``GET /conversations`` (user-scoped listing with title + last-message
preview) and ``GET /conversations/{id}`` (full message history). Model inference
is stubbed via the ``client`` fixture (see tests/api/conftest.py).
"""

from __future__ import annotations

from tests.api.conftest import FAKE_REPLY


def test_conversations_empty_before_first_chat(client):
    resp = client.get("/conversations")
    assert resp.status_code == 200
    assert resp.json() == []


def test_chat_creates_a_listed_conversation(client):
    resp = client.post("/chat", json={"message": "How do I pay off debt?"})

    assert resp.status_code == 200
    conv_id = resp.json()["conversation_id"]

    listing = client.get("/conversations")
    assert listing.status_code == 200
    items = listing.json()
    assert len(items) == 1
    item = items[0]
    assert item["id"] == conv_id
    assert item["title"] == "How do I pay off debt?"
    assert item["last_message"] == FAKE_REPLY
    assert item["updated_at"]


def test_conversation_detail_returns_full_history(client):
    first = client.post("/chat", json={"message": "How do I pay off debt?"}).json()
    conv_id = first["conversation_id"]
    client.post(
        "/chat",
        json={"message": "What about credit cards?", "conversation_id": conv_id},
    )

    detail = client.get(f"/conversations/{conv_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["id"] == conv_id
    assert body["title"] == "How do I pay off debt?"
    assert [turn["role"] for turn in body["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert body["messages"][0]["content"] == "How do I pay off debt?"
    assert body["messages"][1]["content"] == FAKE_REPLY
    assert body["messages"][3]["content"] == FAKE_REPLY


def test_conversation_detail_404_for_unknown(client):
    resp = client.get("/conversations/does-not-exist")
    assert resp.status_code == 404


def test_conversation_list_is_scoped_by_user(client):
    alice = client.post("/chat", json={"message": "Budgeting help?", "user_id": "alice"})
    assert alice.status_code == 200

    bob = client.get("/conversations", params={"user_id": "bob"})
    assert bob.status_code == 200
    assert bob.json() == []

    alice_list = client.get("/conversations", params={"user_id": "alice"})
    assert alice_list.status_code == 200
    assert len(alice_list.json()) == 1


def test_conversation_title_is_derived_from_first_user_message(client):
    long_question = "I want to understand the best way to allocate my monthly savings " * 3
    resp = client.post("/chat", json={"message": long_question})
    assert resp.status_code == 200
    conv_id = resp.json()["conversation_id"]

    title = client.get(f"/conversations/{conv_id}").json()["title"]
    assert len(title) <= 49
    assert title.startswith("I want to understand the best way")
    assert title.endswith("…")


def test_rename_conversation_updates_title_and_persists(client):
    conv_id = client.post("/chat", json={"message": "How do I pay off debt?"}).json()[
        "conversation_id"
    ]

    resp = client.patch(f"/conversations/{conv_id}", json={"title": "Debt payoff plan"})
    assert resp.status_code == 200
    assert resp.json() == {"id": conv_id, "title": "Debt payoff plan"}

    detail = client.get(f"/conversations/{conv_id}").json()
    assert detail["title"] == "Debt payoff plan"

    listing = client.get("/conversations").json()
    assert listing[0]["title"] == "Debt payoff plan"


def test_rename_conversation_rejects_blank_title(client):
    conv_id = client.post("/chat", json={"message": "How do I pay off debt?"}).json()[
        "conversation_id"
    ]
    assert client.patch(f"/conversations/{conv_id}", json={"title": "   "}).status_code == 422
    assert client.patch(f"/conversations/{conv_id}", json={"title": ""}).status_code == 422


def test_rename_conversation_404_for_unknown(client):
    resp = client.patch("/conversations/does-not-exist", json={"title": "Rename me"})
    assert resp.status_code == 404


def test_delete_conversation_removes_it_and_its_advice(client):
    conv_id = client.post("/chat", json={"message": "How do I pay off debt?"}).json()[
        "conversation_id"
    ]
    from src.api import store

    assert len(store.get_store().list_advice()) == 1

    resp = client.delete(f"/conversations/{conv_id}")
    assert resp.status_code == 200
    assert resp.json() == {"id": conv_id, "deleted": True}

    assert client.get(f"/conversations/{conv_id}").status_code == 404
    assert client.get("/conversations").json() == []
    assert store.get_store().list_advice() == []


def test_delete_conversation_respects_owner(client):
    alice = client.post("/chat", json={"message": "Budgeting help?", "user_id": "alice"}).json()
    conv_id = alice["conversation_id"]

    resp = client.delete(f"/conversations/{conv_id}", params={"user_id": "bob"})
    assert resp.status_code == 404
    assert client.get(f"/conversations/{conv_id}").status_code == 200

    assert client.delete(f"/conversations/{conv_id}", params={"user_id": "alice"}).status_code == 200


def test_delete_conversation_404_for_unknown(client):
    assert client.delete("/conversations/does-not-exist").status_code == 404
