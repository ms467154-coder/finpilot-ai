# FinAdvise — PowerShell Runbook

All commands below are written for **Windows PowerShell 5.1+ / PowerShell 7** and should be run
from the project root unless a `cd` is shown.

```
C:\Users\AbdElhalk\OneDrive\Desktop\NLP Projects\Financial Advice Chatbot
```

---

## Prerequisites

| Tool                           | Version      | Check            |
| ------------------------------ | ------------ | ---------------- |
| Python                         | 3.10+        | `python --version` |
| Node.js / npm                  | 18+          | `node --version`  |

---

## 1. Backend setup

```powershell
# Install backend dependencies
python -m pip install -r requirements.txt
```

### Start the backend only (development, auto-reload)

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify it is up:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected: `status = ok`.

Data location:

```powershell
# Optional: pick a custom DB file before starting the backend
$env:CHATBOT_DB_PATH = "C:\path\to\chatbot.db"
```

---

## 2. Frontend setup

```powershell
cd frontend
npm install
cd ..
```

### Start the frontend only (dev server on port 5173)

```powershell
cd frontend
npm run dev
cd ..
```

The app opens at **http://localhost:5173**. The frontend talks to the backend at
`http://127.0.0.1:8000` by default.

---

## 3. One-command start

```powershell
# Start backend + frontend in separate windows, then open the browser
.\start.ps1

# Same, but do not open the browser
.\start.ps1 -NoBrowser

# Same, but before launching, run the full test suites and exit
.\start.ps1 -Test
```

---

## 4. Running the tests

### Backend (71 tests — chat, memory, advice, conversations)

```powershell
python -m pytest tests -q
```

### Frontend (46 tests — components, pages, integration)

```powershell
cd frontend
npm run test
cd ..
```

Equivalent one-liner:

```powershell
cd frontend; npx vitest run; cd ..
```

### Full regression in one shot

```powershell
.\start.ps1 -Test
```

---

## 5. Production build

```powershell
cd frontend
npm run build
cd ..
```

If `npm run build` misbehaves on your machine, call vite directly:

```powershell
cd frontend
npx vite build
cd ..
```

The bundle is written to `frontend\dist\`. To preview it afterwards:

```powershell
cd frontend
npx vite preview
cd ..
```

---

## 6. Data & reset

All SQLite data (conversations, advice, memory) lives in one file:

```
data\chatbot.db
```

Reset everything (safe to delete — it is recreated on backend startup):

```powershell
Remove-Item -LiteralPath "data\chatbot.db" -ErrorAction SilentlyContinue
```

---

## 7. HTTP API reference (all JSON, base = `http://127.0.0.1:8000`)

| Method | Endpoint                          | Purpose                              |
| ------ | --------------------------------- | ------------------------------------ |
| GET    | `/health`                         | Liveness check                       |
| POST   | `/chat`                           | Send a message (`message`, `conversation_id`) |
| GET    | `/conversations`                  | List conversation summaries          |
| GET    | `/conversations/{id}`             | Full conversation history            |
| PATCH  | `/conversations/{id}`             | Rename (body `{"title": "..."}`)     |
| DELETE | `/conversations/{id}`             | Delete conversation (+ its advice)   |
| GET    | `/advice`                         | List structured advice               |
| GET    | `/advice/{id}`                    | One advice record                    |
| POST   | `/advice/{id}/save`               | Mark advice as saved                 |
| GET    | `/memory`                         | List stored user memories            |

Quick smoke test:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

---

## 8. Ports & CORS

| Service   | URL                   | Notes                          |
| --------- | --------------------- | ------------------------------ |
| Backend   | `http://127.0.0.1:8000` | API; DB via `data\chatbot.db`  |
| Frontend  | `http://localhost:5173`  | Vite dev server (auto-open)    |

CORS defaults to `*`. To lock it down to the dev origin(s):

```powershell
$env:CHATBOT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
```

(Set this before starting the backend.)

---

## 9. Stopping the app

Each server runs in its own terminal window; just close those windows. To be safe,
kill by port:

```powershell
# Find PIDs listening on the dev ports
Get-NetTCPConnection -LocalPort 8000,5173 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess

# Then stop them (replace <PID> with the number(s) shown)
Stop-Process -Id <PID> -Force
```