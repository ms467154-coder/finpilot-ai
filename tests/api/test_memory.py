"""End-to-end tests for conversational memory (Phase 13).

Covers the required behaviors:

1. a memory created in one conversation is retrievable and influences a later
   ``/chat`` response in a *new* conversation for the same user;
2. a near-duplicate memory is deduped/updated rather than stored twice;
3. an irrelevant query does not retrieve unrelated memories;
4. ``DELETE /memory/{id}`` removes the memory (not retrieved, not listed);

plus user-scoping/privacy and raw-identifier redaction.

Model inference is stubbed (see tests/api/conftest.py); these tests re-stub
``generate_advice`` so the stub can observe whether the memory context reached
the model and reply accordingly.
"""

from __future__ import annotations

from src.inference import inference
from src.memory import retriever as memory_retriever

from tests.api.conftest import FAKE_REPLY


def _fake_generate_with_memory(marker: str, memory_reply: str):
    """Stub that returns ``memory_reply`` when the memory context is present."""

    def fake(user_message, conversation_history=None, **kwargs):
        context = kwargs.get("system_prompt") or ""
        if marker in context:
            return memory_reply
        return FAKE_REPLY

    return fake


def test_memory_influences_later_conversation(client, monkeypatch):
    monkeypatch.setattr(
        inference,
        "generate_advice",
        _fake_generate_with_memory(
            "house", "Since you are saving for a house, focus on a dedicated savings plan."
        ),
    )

    conv1 = client.post("/chat", json={"message": "I'm saving up for a house.", "user_id": "u1"})
    assert conv1.status_code == 200
    conv1_id = conv1.json()["conversation_id"]

    # Memory was extracted and stored.
    memories = client.get("/memory", params={"user_id": "u1"}).json()
    assert len(memories) == 1
    assert "house" in memories[0]["content"]
    assert memories[0]["category"] == "goal"

    # A new conversation for the same user uses the memory.
    conv2 = client.post(
        "/chat", json={"message": "How should I budget for it?", "user_id": "u1"}
    )
    assert conv2.status_code == 200
    data = conv2.json()
    assert data["conversation_id"] != conv1_id
    assert "saving for a house" in data["reply"]
    used = data["used_memories"] or []
    assert any("house" in mem["content"] for mem in used)


def test_near_duplicate_memory_is_updated_not_duplicated(client, monkeypatch):
    monkeypatch.setattr(
        inference, "generate_advice", _fake_generate_with_memory("never-match", FAKE_REPLY)
    )

    client.post("/chat", json={"message": "I want to save for a car.", "user_id": "u1"})
    client.post("/chat", json={"message": "Actually I'm now saving for a house.", "user_id": "u1"})

    memories = client.get("/memory", params={"user_id": "u1"}).json()
    assert len(memories) == 1
    assert "house" in memories[0]["content"]
    assert "car" not in memories[0]["content"]


def test_irrelevant_query_does_not_retrieve(client, monkeypatch):
    monkeypatch.setattr(
        inference,
        "generate_advice",
        _fake_generate_with_memory("debt", "focus on paying down your debt first."),
    )

    client.post(
        "/chat",
        json={"message": "I want to pay off my debt before investing.", "user_id": "u1"},
    )

    # Direct retriever checks.
    assert memory_retriever.retrieve_relevant_memories("u1", "what is the weather like today", top_k=3) == []
    relevant = memory_retriever.retrieve_relevant_memories("u1", "how can I pay off my debt", top_k=3)
    assert relevant and "debt" in relevant[0].content

    # HTTP level: unrelated question uses no memories.
    data = client.post("/chat", json={"message": "What is the weather like today?", "user_id": "u1"}).json()
    assert (data.get("used_memories") or []) == []
    assert data["reply"] == FAKE_REPLY


def test_delete_memory_removes_it(client, monkeypatch):
    monkeypatch.setattr(
        inference, "generate_advice", _fake_generate_with_memory("house", FAKE_REPLY)
    )

    client.post("/chat", json={"message": "I'm saving for a house.", "user_id": "u1"})
    memory = client.get("/memory", params={"user_id": "u1"}).json()[0]

    resp = client.delete(f"/memory/{memory['id']}", params={"user_id": "u1"})
    assert resp.status_code == 200
    assert resp.json() == {"id": memory["id"], "deleted": True}

    assert client.get("/memory", params={"user_id": "u1"}).json() == []

    # No longer retrieved into a /chat context.
    data = client.post(
        "/chat", json={"message": "How should I budget for a house?", "user_id": "u1"}
    ).json()
    assert (data.get("used_memories") or []) == []

    # Deleting again 404s.
    assert client.delete(f"/memory/{memory['id']}", params={"user_id": "u1"}).status_code == 404


def test_memories_are_scoped_per_user(client, monkeypatch):
    monkeypatch.setattr(
        inference,
        "generate_advice",
        _fake_generate_with_memory("house", "Since you are saving for a house, use a dedicated plan."),
    )

    client.post("/chat", json={"message": "I'm saving for a house.", "user_id": "u1"})
    assert len(client.get("/memory", params={"user_id": "u1"}).json()) == 1
    assert client.get("/memory", params={"user_id": "u2"}).json() == []

    data = client.post("/chat", json={"message": "How should I budget?", "user_id": "u2"}).json()
    assert (data.get("used_memories") or []) == []
    assert "saving for a house" not in data["reply"]

    # A different user cannot delete u1's memory.
    memory = client.get("/memory", params={"user_id": "u1"}).json()[0]
    assert client.delete(f"/memory/{memory['id']}", params={"user_id": "u2"}).status_code == 404


def test_sensitive_identifiers_are_redacted(client, monkeypatch):
    monkeypatch.setattr(
        inference, "generate_advice", _fake_generate_with_memory("never-match", FAKE_REPLY)
    )

    client.post(
        "/chat",
        json={
            "message": "I want to pay off my credit card 4111 1111 1111 1111 soon.",
            "user_id": "u1",
        },
    )
    memories = client.get("/memory", params={"user_id": "u1"}).json()
    assert memories
    content = memories[0]["content"]
    assert "4111" not in content
    assert "redacted" in content
