param(
    [int]$Tail = 30,
    [string]$RequestId = ''
)

$phpLog = 'C:\xampp\moodledata\local_chatbot\logs\e2e_trace_php.jsonl'
$pythonLog = 'C:\xampp\moodledata\local_chatbot\logs\e2e_trace_python.jsonl'

function Format-TraceLine {
    param(
        [string]$Raw,
        [string]$FallbackLayer
    )

    $line = ($Raw | ForEach-Object { $_.Trim() })
    if (-not $line) { return $null }

    try {
        $obj = $line | ConvertFrom-Json
    } catch {
        return "[$FallbackLayer] RAW $line"
    }

    if ($RequestId -and [string]$obj.request_id -ne $RequestId) {
        return $null
    }

    $ts = [string]($obj.timestamp)
    $layer = [string]($obj.layer)
    $event = [string]($obj.event)
    $rid = [string]($obj.request_id)
    $q = if ($null -ne $obj.question_number -and "$($obj.question_number)" -ne '') { [string]$obj.question_number } else { '-' }
    $a = if ($null -ne $obj.attempt -and "$($obj.attempt)" -ne '') { [string]$obj.attempt } else { '-' }
    $dur = if ($null -ne $obj.duration_ms -and "$($obj.duration_ms)" -ne '') { " dur=$($obj.duration_ms)ms" } else { '' }

    $extra = ''
    if ($null -ne $obj.prompt_chars -and "$($obj.prompt_chars)" -ne '') { $extra += " prompt=$($obj.prompt_chars)" }
    if ($null -ne $obj.answer_chars -and "$($obj.answer_chars)" -ne '') { $extra += " answer=$($obj.answer_chars)" }
    if ($null -ne $obj.sources_count -and "$($obj.sources_count)" -ne '') { $extra += " sources=$($obj.sources_count)" }
    if ($null -ne $obj.error -and [string]$obj.error -ne '') {
        $err = [string]$obj.error
        if ($err.Length -gt 120) { $err = $err.Substring(0,120) + '...(truncated)' }
        $extra += " error=$err"
    }

    return "[$ts] [$layer] $event rid=$rid q=$q a=$a$dur$extra"
}

function Start-TraceTailJob {
    param(
        [string]$Name,
        [string]$Path,
        [int]$TailLines
    )

    Start-Job -Name $Name -ArgumentList $Path, $TailLines -ScriptBlock {
        param($path, $tailLines)
        while (-not (Test-Path -LiteralPath $path)) {
            Start-Sleep -Seconds 1
        }
        Get-Content -LiteralPath $path -Tail $tailLines -Wait
    }
}

$jobs = @()
$jobs += Start-TraceTailJob -Name 'php' -Path $phpLog -TailLines $Tail
$jobs += Start-TraceTailJob -Name 'python' -Path $pythonLog -TailLines $Tail

Write-Host "Watching unified trace in one terminal (tail=$Tail)" -ForegroundColor Cyan
if ($RequestId) {
    Write-Host "Filter request_id=$RequestId" -ForegroundColor Cyan
}
Write-Host 'Press Ctrl+C to stop.' -ForegroundColor Cyan
Write-Host ''

try {
    while ($true) {
        foreach ($job in $jobs) {
            $lines = Receive-Job -Job $job -ErrorAction SilentlyContinue
            if (-not $lines) { continue }
            foreach ($line in $lines) {
                $formatted = Format-TraceLine -Raw ([string]$line) -FallbackLayer $job.Name
                if (-not $formatted) { continue }
                if ($formatted -match '\[php\]') {
                    Write-Host $formatted -ForegroundColor Green
                } elseif ($formatted -match '\[python\]') {
                    Write-Host $formatted -ForegroundColor Yellow
                } else {
                    Write-Host $formatted
                }
            }
        }
        Start-Sleep -Milliseconds 150
    }
} finally {
    foreach ($job in $jobs) {
        try { Stop-Job -Job $job -ErrorAction SilentlyContinue | Out-Null } catch {}
        try { Remove-Job -Job $job -ErrorAction SilentlyContinue | Out-Null } catch {}
    }
}
