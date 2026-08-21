"""Shared fixtures for the API endpoint tests (Phase 12).

The full stack is exercised (FastAPI -> :mod:`src.api.chat_service` -> the SQLite
:mod:`src.api.store`) so ``/chat``, ``/advice`` and ``/advice/{id}/save`` are
covered end-to-end. The single exception is ``inference.generate_advice`` which
is stubbed so tests never load the (multi-GB) model.

Phase 13 additionally points the memory store (:mod:`src.memory.store`) at the
same per-test temporary database so conversational-memory tests stay isolated.

Each test gets a fresh temporary database via ``tmp_path`` + ``reset_store``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import main as backend_main
from src.api import store
from src.inference import inference
from src.memory import store as memory_store

# > 12 words so should_extract_advice() always returns True -> an Advice item is
# produced from every /chat call, letting the advice endpoints be tested too.
FAKE_REPLY = (
    "You should pay off high interest debt before investing any spare cash, "
    "and build a small emergency fund to protect your budget."
)


@pytest.fixture
def fake_generate(monkeypatch):
    """Replace model inference with a canned reply for fast, deterministic tests."""
    monkeypatch.setattr(inference, "generate_advice", lambda *args, **kwargs: FAKE_REPLY)
    return FAKE_REPLY


@pytest.fixture
def client(tmp_path, fake_generate):
    """TestClient against a fresh app instance backed by a temp SQLite DB."""
    store.reset_store()
    store.get_store(str(tmp_path / "test.db"))
    memory_store.reset_memory_store()
    memory_store.get_memory_store(str(tmp_path / "test.db"))
    app = backend_main.create_app()
    with TestClient(app) as test_client:
        yield test_client
    memory_store.reset_memory_store()
    store.reset_store()
