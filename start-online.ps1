# Puts the SatQuery-AI backend online from this laptop: starts the API (port 8000, reachable only from this
# machine) and the ngrok tunnel that gives it its public HTTPS address, each in its own window.
# To go offline, close both windows (or press Ctrl+C in each).
# The deployed website (Vercel) only works while both are running.
param(
    [string]$Domain = "endpoint-widen-blinked.ngrok-free.dev",
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "backend"

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "Port $Port is already in use (is the backend already running?). Stop it first." -ForegroundColor Red
    exit 1
}
if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
    Write-Host "ngrok is not installed. Install it with: winget install ngrok.ngrok" -ForegroundColor Red
    exit 1
}

$envFile = Join-Path $backend ".env"
if (-not (Test-Path $envFile) -or -not (Select-String -Path $envFile -Pattern '^\s*CORS_ORIGINS\s*=\s*\S' -Quiet)) {
    Write-Host "Warning: CORS_ORIGINS is not set in backend\.env, so the website will not be allowed to call the backend." -ForegroundColor Yellow
}

# The model needs about 5 GB of the GPU: warn if something else (e.g. a training run) is already using it.
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $used, $total = (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits).Split(",") | ForEach-Object { [int]$_.Trim() }
    if ($total - $used -lt 5000) {
        Write-Host "Warning: only $($total - $used) MB of GPU memory is free; the model needs about 5000 MB. Stop other GPU work first." -ForegroundColor Yellow
    }
}

Write-Host "Starting the backend (the model takes about a minute to load)..."
Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "main_api:app", "--host", "127.0.0.1", "--port", $Port -WorkingDirectory $backend

Write-Host "Starting the ngrok tunnel..."
Start-Process -FilePath "ngrok" -ArgumentList "http", "--url=https://$Domain", $Port

Write-Host ""
Write-Host "Backend address: https://$Domain" -ForegroundColor Green
Write-Host "The website works once the backend window shows 'Application startup complete'."
Write-Host "To go offline, close the two new windows."
