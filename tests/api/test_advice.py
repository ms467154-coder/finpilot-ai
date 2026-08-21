"""End-to-end tests for the advice endpoints (Phase 12).

Covers ``GET /advice`` (list + category filter), ``GET /advice/{id}``, and
``POST /advice/{id}/save`` including 404 behaviour. Advice rows are created by
calling ``POST /chat`` (with model inference stubbed via the ``client`` fixture).
"""

from __future__ import annotations

UNKNOWN_ID = "does-not-exist"


def _post_chat(client, message: str = "How do I pay off debt?"):
    return client.post("/chat", json={"message": message})


def test_advice_empty_when_nothing_chatted(client):
    resp = client.get("/advice")
    assert resp.status_code == 200
    assert resp.json() == []


def test_advice_lists_stored_items(client):
    first_id = _post_chat(client).json()["advice"]["id"]
    second_id = _post_chat(client, "Best way to start investing?").json()["advice"]["id"]

    items = client.get("/advice").json()

    assert {item["id"] for item in items} == {first_id, second_id}
    assert all(item["category"] for item in items)
    assert all(item["saved"] is False for item in items)
    assert all(item["conversation_id"] for item in items)
    # newest-first (timestamp DESC)
    assert [item["timestamp"] for item in items] == sorted(
        [item["timestamp"] for item in items], reverse=True
    )


def test_advice_filters_by_category(client):
    _post_chat(client)
    all_items = client.get("/advice").json()
    category = all_items[0]["category"]

    matched = client.get(f"/advice?category={category}").json()
    assert len(matched) == 1
    assert matched[0]["id"] == all_items[0]["id"]

    assert client.get("/advice?category=NoSuchCategory").json() == []


def test_advice_get_one(client):
    advice_id = _post_chat(client).json()["advice"]["id"]

    resp = client.get(f"/advice/{advice_id}")
    assert resp.status_code == 200
    item = resp.json()
    assert item["id"] == advice_id
    assert item["short_title"]
    assert item["key_recommendation"]


def test_advice_get_missing_returns_404(client):
    resp = client.get(f"/advice/{UNKNOWN_ID}")
    assert resp.status_code == 404


def test_advice_save_marks_saved(client):
    advice_id = _post_chat(client).json()["advice"]["id"]

    save = client.post(f"/advice/{advice_id}/save")
    assert save.status_code == 200
    assert save.json() == {"id": advice_id, "saved": True}

    assert client.get(f"/advice/{advice_id}").json()["saved"] is True

    listed = client.get("/advice").json()
    assert any(item["id"] == advice_id and item["saved"] is True for item in listed)


def test_advice_save_missing_returns_404(client):
    resp = client.post(f"/advice/{UNKNOWN_ID}/save")
    assert resp.status_code == 404
