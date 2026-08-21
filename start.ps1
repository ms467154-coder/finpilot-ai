# start.ps1 - Launch the FinAdvise full stack (FastAPI backend + React/Vite frontend).
#
#   .\start.ps1             Start both servers in their own windows and open the app.
#   .\start.ps1 -NoBrowser  Start both servers without opening the browser.
#   .\start.ps1 -Test       Run the backend + frontend test suites, then exit.
#
# For the step-by-step first-time setup, see RUNBOOK.md.

[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$Test
)

$ErrorActionPreference = "Stop"

$Root     = Split-Path -Parent $MyInvocation.MyCommand.Path
$Frontend = Join-Path $Root "frontend"
$DataDir  = Join-Path $Root "data"

function Write-Step([string]$message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Yellow
}

function Confirm-Command([string]$name, [string]$label) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Host "$label was not found on PATH."
        Write-Host "See RUNBOOK.md for setup instructions."
        Write-Error "Missing prerequisite: $label"
    }
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
Write-Step "Checking prerequisites..."

Confirm-Command "python" "Python 3.10+"
Confirm-Command "npm" "Node.js / npm"

if (-not (Test-Path -LiteralPath (Join-Path $Frontend "node_modules"))) {
    Write-Host "frontend\node_modules is missing. Run 'npm install' inside the frontend folder first."
    Write-Host "See RUNBOOK.md under 'Frontend setup'."
    Write-Error "Frontend dependencies are not installed."
}

# ---------------------------------------------------------------------------
# Shared data location (persistence for chat + advice + memory)
# ---------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir | Out-Null
}
$env:CHATBOT_DB_PATH = Join-Path $DataDir "chatbot.db"

# ---------------------------------------------------------------------------
# Test mode: run the full regression suites and exit.
# ---------------------------------------------------------------------------
if ($Test) {
    # npm/python write progress to stderr; keep that from aborting under
    # $ErrorActionPreference = "Stop". Native exit codes are checked manually.
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    Write-Step "Running backend tests (pytest)..."
    & python -m pytest tests -q
    $backendCode = $LASTEXITCODE

    Write-Step "Running frontend tests (vitest)..."
    Push-Location $Frontend
    try {
        & npm.cmd run test
        $frontendCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }

    $ErrorActionPreference = $oldEap

    if ($backendCode -ne 0) {
        Write-Host "Backend tests FAILED." -ForegroundColor Red
        exit $backendCode
    }
    if ($frontendCode -ne 0) {
        Write-Host "Frontend tests FAILED." -ForegroundColor Red
        exit $frontendCode
    }
    Write-Host ""
    Write-Host "All done." -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------------------
# Normal mode: launch the app.
# ---------------------------------------------------------------------------
Write-Step "Starting FinAdvise..."
Write-Host "  Backend  -> http://127.0.0.1:8000  (FastAPI, --reload)"
Write-Host "  Frontend -> http://localhost:5173  (React + Vite)"
Write-Host "  DB       -> $env:CHATBOT_DB_PATH"
Write-Host ""

# Backend in its own window (auto-reloads on source changes during development).
Start-Process -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload" `
    -WorkingDirectory $Root

# Frontend dev server in its own window.
Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory $Frontend

# Open the app in the default browser once the dev server has had time to boot.
if (-not $NoBrowser) {
    Start-Sleep -Seconds 4
    Start-Process "http://localhost:5173"
}

Write-Host ""
Write-Host "Both servers started. Close their windows to stop them."
Write-Host "Run '.\start.ps1 -Test' to run the full test suite." -ForegroundColor Green