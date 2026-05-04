param(
    [ValidateSet('all', 'php', 'python')]
    [string]$Mode = 'all',
    [int]$Tail = 50
)

$phpLog = 'C:\xampp\moodledata\local_chatbot\logs\e2e_trace_php.jsonl'
$pythonLog = 'C:\xampp\moodledata\local_chatbot\logs\e2e_trace_python.jsonl'

$sources = @()
if ($Mode -eq 'all' -or $Mode -eq 'php') {
    $sources += [pscustomobject]@{ Name = 'PHP'; Path = $phpLog }
}
if ($Mode -eq 'all' -or $Mode -eq 'python') {
    $sources += [pscustomobject]@{ Name = 'PYTHON'; Path = $pythonLog }
}

if ($sources.Count -eq 0) {
    Write-Host 'No log source selected.' -ForegroundColor Yellow
    exit 1
}

$jobs = @()

foreach ($src in $sources) {
    $jobs += Start-Job -ArgumentList $src.Name, $src.Path, $Tail -ScriptBlock {
        param($name, $path, $tail)

        while (-not (Test-Path -LiteralPath $path)) {
            Write-Output "[$name] waiting for log file: $path"
            Start-Sleep -Seconds 2
        }

        Get-Content -LiteralPath $path -Tail $tail -Wait | ForEach-Object {
            "[$name] $_"
        }
    }
}

Write-Host "Watching logs mode=$Mode tail=$Tail" -ForegroundColor Cyan
Write-Host 'Press Ctrl+C to stop.' -ForegroundColor Cyan

try {
    while ($true) {
        foreach ($job in $jobs) {
            $lines = Receive-Job -Job $job -ErrorAction SilentlyContinue
            foreach ($line in $lines) {
                if ($line -like '[PHP]*') {
                    Write-Host $line -ForegroundColor Green
                } elseif ($line -like '[PYTHON]*') {
                    Write-Host $line -ForegroundColor Yellow
                } else {
                    Write-Host $line
                }
            }
        }
        Start-Sleep -Milliseconds 250
    }
} finally {
    foreach ($job in $jobs) {
        try {
            Stop-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
        } catch {}
        try {
            Remove-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
        } catch {}
    }
}
