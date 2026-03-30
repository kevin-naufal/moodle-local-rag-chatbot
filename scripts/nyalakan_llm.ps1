param(
    [int]$Port = 8501,
    [switch]$SkipModelPull
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
Set-Location $projectRoot

$chatModel = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"
$embedModel = "nomic-embed-text"
$ollamaBaseUrl = "http://127.0.0.1:11434"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

function Test-OllamaReachable {
    param([int]$TimeoutSec = 2)
    try {
        Invoke-RestMethod -Uri "$ollamaBaseUrl/api/tags" -Method Get -TimeoutSec $TimeoutSec | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Wait-Ollama {
    param(
        [int]$MaxWaitSec = 25
    )

    $elapsed = 0
    while ($elapsed -lt $MaxWaitSec) {
        if (Test-OllamaReachable -TimeoutSec 2) {
            return $true
        }
        Start-Sleep -Seconds 1
        $elapsed++
    }
    return $false
}

Write-Host "== Menyalakan LLM project =="
Write-Host "Folder: $projectRoot"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Perintah 'ollama' tidak ditemukan. Install Ollama dulu: https://ollama.com/download"
}

if (-not (Test-OllamaReachable)) {
    Write-Host "Ollama belum aktif. Menjalankan 'ollama serve' di background..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Minimized | Out-Null
}

if (-not (Wait-Ollama -MaxWaitSec 25)) {
    throw "Ollama belum bisa diakses di $ollamaBaseUrl. Coba jalankan manual: ollama serve"
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment belum ada. Membuat .venv..."
    py -3 -m venv .venv
}

if (-not (Test-Path $venvPython)) {
    throw "Gagal menemukan Python di .venv\Scripts\python.exe"
}

Write-Host "Memastikan dependency Python terpasang..."
& $venvPython -m pip install -q -r requirements.txt

if (-not $SkipModelPull) {
    $installedModels = ollama list | Out-String
    if ($installedModels -notmatch [regex]::Escape($chatModel)) {
        Write-Host "Mengunduh chat model: $chatModel"
        ollama pull $chatModel
    }
    if ($installedModels -notmatch [regex]::Escape($embedModel)) {
        Write-Host "Mengunduh embedding model: $embedModel"
        ollama pull $embedModel
    }
}

$url = "http://127.0.0.1:$Port"
Write-Host "Membuka browser: $url"
Start-Process $url | Out-Null

Write-Host "Menjalankan Streamlit (tekan Ctrl+C untuk stop)..."
& $venvPython -m streamlit run app/chatbot_ui.py --server.headless true --server.port $Port
