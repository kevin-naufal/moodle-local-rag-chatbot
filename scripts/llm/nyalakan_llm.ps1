param(
    [switch]$SkipModelPull
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
Set-Location $projectRoot

$chatModel = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"
$defaultEmbedModel = "nomic-embed-text"
$embedModel = $defaultEmbedModel
$embedBackend = "auto"
$ollamaBaseUrl = "http://127.0.0.1:11434"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$envFile = Join-Path $projectRoot ".env"

function Import-DotEnvFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return
    }

    foreach ($line in Get-Content $Path) {
        $trimmed = [string]$line
        if ([string]::IsNullOrWhiteSpace($trimmed)) {
            continue
        }
        $trimmed = $trimmed.Trim()
        if ($trimmed.StartsWith("#")) {
            continue
        }
        $parts = $trimmed -split "=", 2
        if ($parts.Count -ne 2) {
            continue
        }
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ($value.Length -ge 2) {
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        if ($name) {
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

Import-DotEnvFile -Path $envFile

if ($env:OLLAMA_BASE_URL) {
    $ollamaBaseUrl = $env:OLLAMA_BASE_URL.TrimEnd("/")
}
if ($env:EMBED_MODEL) {
    $embedModel = $env:EMBED_MODEL.Trim()
}
if ($env:EMBED_BACKEND) {
    $embedBackend = $env:EMBED_BACKEND.Trim().ToLowerInvariant()
}

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

function Start-OllamaDebugServer {
    Write-Host "Membuka terminal Ollama debug server..."

    $running = Get-Process ollama -ErrorAction SilentlyContinue
    if ($running) {
        Write-Host "Restart Ollama agar log HTTP/API tampil di terminal debug..."
        $running | Stop-Process -Force
        Start-Sleep -Seconds 2
    }

    $serveCommand = @"
`$Host.UI.RawUI.WindowTitle = 'Ollama debug server'
`$env:OLLAMA_DEBUG = '1'
`$env:OLLAMA_HOST = '$ollamaBaseUrl'
Write-Host '== Ollama debug server =='
Write-Host 'Press Ctrl+C here to stop Ollama.'
ollama serve
"@

    Start-Process -FilePath "powershell" -ArgumentList @(
        "-NoExit",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", $serveCommand
    ) | Out-Null

    if (-not (Wait-Ollama -MaxWaitSec 30)) {
        throw "Ollama debug server belum bisa diakses di $ollamaBaseUrl."
    }
}

function Start-ChatModel {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Model,
        [string]$KeepAlive = "30m"
    )

    Write-Host "Menyalakan chat model: $Model"
    $body = @{
        model = $Model
        prompt = "Reply with OK."
        stream = $false
        keep_alive = $KeepAlive
        options = @{
            num_predict = 1
            temperature = 0
        }
    } | ConvertTo-Json -Depth 4

    try {
        Invoke-RestMethod -Uri "$ollamaBaseUrl/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120 | Out-Null
    } catch {
        throw "Gagal menyalakan chat model '$Model'. Details: $($_.Exception.Message)"
    }
}

function Show-ChatModelStatus {
    Write-Host ""
    Write-Host "== Status LLM aktif =="
    ollama ps
}

Write-Host "== Menyalakan LLM project =="
Write-Host "Folder: $projectRoot"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Perintah 'ollama' tidak ditemukan. Install Ollama dulu: https://ollama.com/download"
}

Start-OllamaDebugServer

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment belum ada. Membuat .venv..."
    py -3 -m venv .venv
}

if (-not (Test-Path $venvPython)) {
    throw "Gagal menemukan Python di .venv\Scripts\python.exe"
}

Write-Host "Memastikan dependency Python terpasang..."
& $venvPython -m pip install -r requirements.txt

if (-not $SkipModelPull) {
    $installedModels = ollama list | Out-String
    if ($installedModels -notmatch [regex]::Escape($chatModel)) {
        Write-Host "Mengunduh chat model: $chatModel"
        ollama pull $chatModel
    }
    if ($embedBackend -in @("auto", "ollama") -and $installedModels -notmatch [regex]::Escape($embedModel)) {
        Write-Host "Mengunduh embedding model: $embedModel"
        ollama pull $embedModel
    } elseif ($embedBackend -eq "bert") {
        Write-Host "Embedding backend BERT aktif; skip pull embedding model Ollama."
    }
}

Start-ChatModel -Model $chatModel
Show-ChatModelStatus
Write-Host "Environment LLM siap."
