param(
    [string]$Dataset = ".\data\answer_run_questions\ch03_running_time_in_scope_30q.json",
    [string]$DataDir = ".\data\eval_ch03_only",
    [int]$Runs = 3,
    [string]$Modes = "llm_only,rag_ollama,rag_bert",
    [string]$ExistingAnswerRuns = "",
    [switch]$UseExistingAnswerRuns,
    [switch]$SkipModelPull,
    [switch]$SkipPreparse,
    [switch]$SkipSystemPlots,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
Set-Location $projectRoot

$chatModel = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"
$defaultEmbedModel = "nomic-embed-text"
$embedModel = $defaultEmbedModel
$embedBackend = "auto"
$ollamaBaseUrl = "http://127.0.0.1:11434"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
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

function Resolve-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return (Resolve-Path $PathValue).Path
    }
    return (Resolve-Path (Join-Path $projectRoot $PathValue)).Path
}

function Convert-ToAbsoluteProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $PathValue))
}

function Get-EnvBoolean {
    param(
        [string]$Value,
        [bool]$Default = $false
    )

    $text = [string]$Value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $Default
    }

    switch ($text.Trim().ToLowerInvariant()) {
        "1" { return $true }
        "true" { return $true }
        "yes" { return $true }
        "y" { return $true }
        "on" { return $true }
        "0" { return $false }
        "false" { return $false }
        "no" { return $false }
        "n" { return $false }
        "off" { return $false }
        default { return $Default }
    }
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

Import-DotEnvFile -Path $envFile

if (-not $PSBoundParameters.ContainsKey("Dataset") -and $env:DEMO_EVAL_DATASET) {
    $Dataset = $env:DEMO_EVAL_DATASET
}
if (-not $PSBoundParameters.ContainsKey("DataDir") -and $env:DEMO_EVAL_DATA_DIR) {
    $DataDir = $env:DEMO_EVAL_DATA_DIR
}
if (-not $PSBoundParameters.ContainsKey("Runs") -and $env:DEMO_EVAL_RUNS) {
    $Runs = [int]$env:DEMO_EVAL_RUNS
}
if (-not $PSBoundParameters.ContainsKey("Modes") -and $env:DEMO_EVAL_MODES) {
    $Modes = $env:DEMO_EVAL_MODES
}
if (-not $PSBoundParameters.ContainsKey("ExistingAnswerRuns") -and $env:DEMO_EVAL_EXISTING_ANSWER_RUNS) {
    $ExistingAnswerRuns = $env:DEMO_EVAL_EXISTING_ANSWER_RUNS
}
if (-not $PSBoundParameters.ContainsKey("UseExistingAnswerRuns")) {
    $UseExistingAnswerRuns = Get-EnvBoolean -Value $env:DEMO_EVAL_USE_EXISTING_ANSWER_RUNS -Default $false
}
if ($env:OLLAMA_BASE_URL) {
    $ollamaBaseUrl = $env:OLLAMA_BASE_URL.TrimEnd("/")
}
if ($env:EMBED_MODEL) {
    $embedModel = $env:EMBED_MODEL.Trim()
}
if ($env:EMBED_BACKEND) {
    $embedBackend = $env:EMBED_BACKEND.Trim().ToLowerInvariant()
}

function Wait-Ollama {
    param([int]$MaxWaitSec = 25)

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

function Start-TraceMonitor {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TraceLogPath
    )

    $watchScript = Join-Path $projectRoot "scripts\tools\watch_chatbot_logs.ps1"
    if (-not (Test-Path $watchScript)) {
        Write-Host "Monitoring trace tidak ditemukan: $watchScript" -ForegroundColor Yellow
        return
    }

    $traceDir = Split-Path -Parent $TraceLogPath
    if ($traceDir) {
        New-Item -ItemType Directory -Force -Path $traceDir | Out-Null
    }
    if (-not (Test-Path $TraceLogPath)) {
        New-Item -ItemType File -Force -Path $TraceLogPath | Out-Null
    }

    Write-Host ""
    Write-Host "== Monitoring komunikasi backend <-> LLM =="
    Write-Host "Log Python trace : $TraceLogPath"
    Write-Host "Membuka terminal monitoring. Tekan Ctrl+C di terminal monitoring untuk berhenti."

    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $watchScript,
        "python",
        "50"
    )
    Start-Process -FilePath "powershell" -ArgumentList $arguments | Out-Null
}

function Ensure-ToolingReady {
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
        $modeList = @($Modes.ToLowerInvariant().Split(",") | ForEach-Object { $_.Trim() })
        $needsOllamaEmbedding = $embedBackend -in @("auto", "ollama") -or $modeList -contains "rag_ollama"
        if ($needsOllamaEmbedding -and $installedModels -notmatch [regex]::Escape($embedModel)) {
            Write-Host "Mengunduh embedding model: $embedModel"
            ollama pull $embedModel
        } elseif (-not $needsOllamaEmbedding) {
            Write-Host "Embedding Ollama tidak dibutuhkan untuk mode/backend saat ini."
        }
    }

    Start-ChatModel -Model $chatModel
    Show-ChatModelStatus
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )

    $display = $ArgumentList | ForEach-Object {
        if ($_ -match '\s') {
            '"' + $_ + '"'
        } else {
            $_
        }
    }

    Write-Host ""
    Write-Host "== $Label =="
    Write-Host ($display -join " ")

    if ($DryRun) {
        return
    }

    & $venvPython @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Label' gagal dengan exit code $LASTEXITCODE."
    }
}

function Snapshot-AnswerRunsFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    Write-Host ""
    Write-Host "== Snapshot answer runs =="
    Write-Host "$SourcePath -> $TargetPath"

    if ($DryRun) {
        return
    }

    $targetDir = Split-Path -Parent $TargetPath
    if ($targetDir) {
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    }
    Copy-Item -LiteralPath $SourcePath -Destination $TargetPath -Force
}


$datasetPath = Resolve-ProjectPath $Dataset

if (-not (Test-Path $datasetPath)) {
    throw "Dataset tidak ditemukan: $datasetPath"
}
$dataDirPath = ""
if (-not $UseExistingAnswerRuns) {
    $dataDirPath = Resolve-ProjectPath $DataDir
    if (-not (Test-Path $dataDirPath)) {
        throw "Folder corpus tidak ditemukan: $dataDirPath"
    }
} else {
    if ($DataDir) {
        try {
            $dataDirPath = Resolve-ProjectPath $DataDir
        } catch {
            $dataDirPath = $DataDir
        }
    }
}

$answerRunsPath = Join-Path $projectRoot "data\answer_runs\demo_answer_runs_$timestamp.jsonl"
$answerRunsSource = "generated"
$datasetStem = [System.IO.Path]::GetFileNameWithoutExtension($datasetPath)
$safeDatasetStem = [regex]::Replace($datasetStem, "[^A-Za-z0-9_-]+", "_")
if (-not $safeDatasetStem) {
    $safeDatasetStem = "dataset"
}
$resumeAnswerRunsPath = Join-Path $projectRoot "data\answer_runs\demo_answer_runs_resume_${safeDatasetStem}.jsonl"
$evalOutputDir = Join-Path $projectRoot "data\eval_results\demo_eval_${safeDatasetStem}_$timestamp"
$answerRunsSnapshotPath = Join-Path $evalOutputDir "answer_runs.jsonl"
$systemRunsPath = Join-Path $evalOutputDir "system_eval_runs.jsonl"
$systemSummaryPath = Join-Path $evalOutputDir "system_eval_summary.json"
$systemPlotDir = Join-Path $evalOutputDir "system_eval_plots"
$judgedRunsPath = Join-Path $evalOutputDir "judged_runs.jsonl"
$qualityRunsPath = Join-Path $evalOutputDir "quality_eval_runs.jsonl"
$qualitySummaryPath = Join-Path $evalOutputDir "quality_eval_summary.json"
$qualityPlotDir = Join-Path $evalOutputDir "quality_eval_plots"
$traceLogPath = Join-Path "C:\xampp\moodledata\local_chatbot\logs" "e2e_trace_python.jsonl"

if ($UseExistingAnswerRuns) {
    if (-not $ExistingAnswerRuns) {
        throw "UseExistingAnswerRuns aktif, tapi path ExistingAnswerRuns belum diisi."
    }
    $candidateAnswerRunsPath = Convert-ToAbsoluteProjectPath $ExistingAnswerRuns
    if (Test-Path $candidateAnswerRunsPath) {
        $answerRunsPath = $candidateAnswerRunsPath
        $answerRunsSource = "existing"
    } else {
        throw "File answer-runs tidak ditemukan: $candidateAnswerRunsPath"
    }
} else {
    $answerRunsPath = $resumeAnswerRunsPath
    if (Test-Path $answerRunsPath) {
        $answerRunsSource = "resume_partial"
    } else {
        $answerRunsSource = "resume_new"
    }
}

Write-Host "== Demo Evaluation Runner =="
Write-Host "Project : $projectRoot"
Write-Host "Dataset : $datasetPath"
Write-Host "Corpus  : $dataDirPath"
Write-Host "Modes   : $Modes"
Write-Host "Runs    : $Runs"
Write-Host "Answers : $answerRunsSource"
Write-Host "Output  : $evalOutputDir"
if ($DryRun) {
    Write-Host "Mode    : dry-run"
}

if (-not $DryRun) {
    Ensure-ToolingReady
    Start-TraceMonitor -TraceLogPath $traceLogPath
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $evalOutputDir | Out-Null
}

if ($UseExistingAnswerRuns) {
    Write-Host ""
    Write-Host "== Generate answer runs =="
    Write-Host "Skip generate. Menggunakan file existing: $answerRunsPath"
} else {
    $runEvalArgs = @(
        ".\scripts\eval\run_eval.py",
        "--dataset", $datasetPath,
        "--data-dir", $dataDirPath,
        "--output", $answerRunsPath,
        "--runs", $Runs.ToString(),
        "--modes", $Modes,
        "--trace-log", $traceLogPath
    )
    if ($SkipPreparse) {
        $runEvalArgs += "--skip-preparse"
    }
    $runEvalArgs += "--resume"

    Invoke-Step -Label "Generate answer runs" -ArgumentList $runEvalArgs
}

Snapshot-AnswerRunsFile -SourcePath $answerRunsPath -TargetPath $answerRunsSnapshotPath

$systemEvalArgs = @(
    ".\scripts\eval\evaluate_system_performance.py",
    "--questions", $datasetPath,
    "--answer-runs", $answerRunsSnapshotPath,
    "--output-runs", $systemRunsPath,
    "--output-summary", $systemSummaryPath
)
Invoke-Step -Label "Evaluate system performance" -ArgumentList $systemEvalArgs

if (-not $SkipSystemPlots) {
    $systemPlotArgs = @(
        ".\scripts\eval\plot_system_eval.py",
        "--summary", $systemSummaryPath,
        "--output-dir", $systemPlotDir
    )
    Invoke-Step -Label "Generate system plots" -ArgumentList $systemPlotArgs
}

$judgeArgs = @(
    ".\scripts\eval\auto_judge_answer_quality.py",
    "--questions", $datasetPath,
    "--answer-runs", $answerRunsSnapshotPath,
    "--output", $judgedRunsPath
)
Invoke-Step -Label "Auto-judge answer quality" -ArgumentList $judgeArgs

$qualityEvalArgs = @(
    ".\scripts\eval\evaluate_answer_quality.py",
    "--judged-runs", $judgedRunsPath,
    "--output-runs", $qualityRunsPath,
    "--output-summary", $qualitySummaryPath,
    "--plot",
    "--plot-output-dir", $qualityPlotDir
)
Invoke-Step -Label "Summarize answer quality" -ArgumentList $qualityEvalArgs

Write-Host ""
Write-Host "== Output Penting =="
Write-Host "eval_output_dir      : $evalOutputDir"
Write-Host "answer_runs          : $answerRunsSnapshotPath"
Write-Host "system_eval_runs     : $systemRunsPath"
Write-Host "system_eval_summary  : $systemSummaryPath"
if (-not $SkipSystemPlots) {
    Write-Host "system_eval_plots    : $systemPlotDir"
}
Write-Host "judged_runs          : $judgedRunsPath"
Write-Host "quality_eval_runs    : $qualityRunsPath"
Write-Host "quality_eval_summary : $qualitySummaryPath"
Write-Host "quality_eval_plots   : $qualityPlotDir"
Write-Host ""
Write-Host "Demo evaluation completed successfully."
