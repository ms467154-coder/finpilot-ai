# FinAdvise — Financial Advice Chatbot

A conversational financial advisor that answers money questions in plain
language and turns actionable replies into **structured, dashboard-able advice**.

- **Chat** (Phase 10) — multi-turn chat with a fine-tuned Qwen2.5-0.5B-Instruct
  model (LoRA adapter). Replies that contain actionable guidance are structured
  into advice records.
- **Dashboard** (Phase 11) — organizes every piece of stored advice: recent
  items, chronological history (paginated), categories, key recommendations,
  a topic tag-cloud, and saved advice.
- **Integration & polish** (Phase 12) — shared navigation/layout, global error
  handling and loading states, responsive styling, backend endpoint tests, and
  this documentation.

The app is **educational and qualitative**. It deliberately does **not** collect
or analyze any personal financial-profile data (no salary / income / expense /
net-worth / portfolio inputs anywhere).

## Architecture overview

```
┌───────────────────────── Frontend (React + Vite, :5173) ─────────────────────────┐
│                                                                                  │
│  App (view router)                                                               │
│   ├── components/Layout  AppShell · NavBar · ApiStatus (global banner/loading)   │
│   │                     · ErrorBoundary                                          │
│   ├── pages/Dashboard   DashboardPage (Recent · History · Categories · Topics ·  │
│   │                                          Key Recommendations · Saved)        │
│   ├── components/Chat   ChatWindow (bubbles · advice cards · input)              │
│   └── api/client        sendMessage · listAdvice · getAdvice · saveAdvice        │
│              │  fetch (JSON, CORS enabled)                                       │
└──────────────┼───────────────────────────────────────────────────────────────────┘
               ▼
┌───────────────────────── Backend (FastAPI, :8000) ───────────────────────────────┐
│  backend/routes.py      POST /chat · GET /advice · GET /advice/{id}              │
│                         POST /advice/{id}/save · GET /health                    │
│  backend/schemas.py     Pydantic request/response models                        │
│  src/api/chat_service   orchestrates a chat turn + advice extraction            │
│  src/inference          generate_advice (fine-tuned model, lazy-loaded)          │
│  src/advice/processor   raw reply → structured Advice (category, title, key)     │
│  src/api/store          SQLite persistence (conversations + advice)              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Request flow (`POST /chat`):**
route → `chat_service.chat()` → `inference.generate_advice()` (with stored
history for multi-turn) → conversation persisted → if the reply is actionable,
`advice.processor` builds a structured `Advice` which is stored and returned
alongside the reply.

**Data storage (SQLite):**
- `conversations` — id, timestamps, JSON message history.
- `advice` — structured advice records plus a `saved` flag (for the dashboard's
  Saved Advice section).
- The file lives at `data/chatbot.db` by default; override with `CHATBOT_DB_PATH`.

**Theme:** dark/black background with a yellow `#ffd21f` accent, defined as CSS
custom properties in `frontend/src/styles/theme.css` and shared across the Chat,
Dashboard, and Layout components.

## Repository layout

```
backend/            FastAPI app (main, routes, schemas)
src/
  api/              chat_service + SQLite store
  advice/           categorizer + processor + advice schemas
  data/             dataset loading / cleaning / splitting
  evaluation/       metrics + evaluation scripts
  inference/        prompt templates + generation
  model/            base-model loader
  training/         dataset prep + LoRA training
frontend/           React + Vite app (chat + dashboard)
tests/              pytest suite (tests/api covers the HTTP endpoints)
scripts/            verification scripts
configs/            model / training configuration (YAML)
models/             base model cache + LoRA adapters (models/fine_tuned/final)
docs/               documentation
```

## Setup

Prerequisites: **Python 3.11+**, **Node.js 18+** (npm 9+).

```bash
# 1. Backend dependencies
python -m pip install -r requirements.txt

# 2. Model artifacts (base model is cached under models/base; the LoRA adapter
#    lives at models/fine_tuned/final). If the base model is missing it will be
#    downloaded from Hugging Face on first use.

# 3. Frontend dependencies
cd frontend
npm install
cd ..
```

## Running

### Backend (from the repository root)

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The API is then live at `http://127.0.0.1:8000` (docs at `/docs`). Liveness
check: `curl http://127.0.0.1:8000/health` → `{"status":"ok"}`.

The model is loaded lazily on the first `/chat` request, so the first reply can
take a few seconds/minutes (downloads + CPU/GPU load) — subsequent requests are
fast.

### Frontend

```bash
cd frontend
npm run dev          # http://localhost:5173
```

The frontend expects the backend at `http://127.0.0.1:8000`. To point it
elsewhere, set `VITE_API_BASE_URL` (see Configuration).

Production build:

```bash
npm run build        # outputs to frontend/dist
npm run preview      # serve the built app locally
```

## Testing

```bash
# Backend — API endpoint tests (model inference is stubbed in tests/api/conftest.py)
python -m pytest tests/

# Frontend — component + integration tests (jsdom)
cd frontend
npm test
```

## API reference

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/chat` | Multi-turn chat. Body: `{"message", "conversation_id"?}` → `{conversation_id, reply, advice?}` |
| `GET` | `/advice` | Stored advice history, newest first. Optional `?category=` filter |
| `GET` | `/advice/{id}` | A single advice record (404 if unknown) |
| `POST` | `/advice/{id}/save` | Mark an advice record as saved |
| `GET` | `/health` | Liveness check |

## Configuration

| Variable | Where | Default | Purpose |
| --- | --- | --- | --- |
| `CHATBOT_DB_PATH` | backend | `data/chatbot.db` | SQLite database path |
| `CHATBOT_CORS_ORIGINS` | backend | `*` | Comma-separated allowed CORS origins |
| `VITE_API_BASE_URL` | frontend | `http://127.0.0.1:8000` | Backend base URL used by the API client |

Model behaviour (base model, precision, generation params, adapter path) is
configured in `configs/model_config.yaml`.

## Troubleshooting

- **Frontend shows a red banner / chat errors** — the backend is not running or
  unreachable. Start it (see Running) or check `VITE_API_BASE_URL`.
- **First `/chat` reply is slow** — the model is loading; later requests reuse
  the cached model.
- **Dashboard shows no advice** — advice records are only created when a chat
  reply is actionable; start a conversation in the Chat tab first.
