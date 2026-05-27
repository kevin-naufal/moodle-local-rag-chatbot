param(
    [switch]$StopServer
)

$ErrorActionPreference = "Stop"

$chatModels = "qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b"
$embedModel = "nomic-embed-text"

if ($env:DEMO_EVAL_CHAT_MODELS) {
    $chatModels = $env:DEMO_EVAL_CHAT_MODELS.Trim()
} elseif ($env:CHAT_MODEL) {
    $chatModels = $env:CHAT_MODEL.Trim()
}

function Split-CsvList {
    param([string]$Value)
    return @([string]$Value -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

Write-Host "== Mematikan model LLM =="

foreach ($model in @((Split-CsvList $chatModels) + $embedModel)) {
    try {
        ollama stop $model | Out-Null
        Write-Host "Stopped: $model"
    } catch {
        Write-Host "Skip (tidak aktif): $model"
    }
}

if ($StopServer) {
    Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "Process Ollama dimatikan."
}

Write-Host "Selesai."
