"""FastAPI application entry point (Phase 9).

Wires the routes from :mod:`backend.routes`, enables CORS for the frontend
(origins from ``CHATBOT_CORS_ORIGINS``, comma-separated; ``*`` by default), and
initializes the SQLite store on startup.

Run::

    uvicorn backend.main:app --host 127.0.0.1 --port 8000

No frontend/dashboard UI is built here yet - only the API surface.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import store

from .routes import router


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.get_store().init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial Advice Chatbot API",
        version="0.1.0",
        description="Multi-turn financial advice chat + structured advice storage.",
        lifespan=lifespan,
    )

    origins = [o.strip() for o in os.environ.get("CHATBOT_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/health", summary="Liveness check")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()

__all__ = ["app", "create_app"]