param(
    [switch]$StopServer
)

$ErrorActionPreference = "Stop"

$chatModel = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"
$embedModel = "nomic-embed-text"

Write-Host "== Mematikan model LLM =="

foreach ($model in @($chatModel, $embedModel)) {
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
