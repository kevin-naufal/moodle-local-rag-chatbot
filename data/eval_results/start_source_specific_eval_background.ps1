$ErrorActionPreference = "Stop"

$repo = "C:\Users\Kevin\Downloads\my-llm"
$runScript = Join-Path $repo "data\eval_results\run_source_specific_eval.ps1"
$stdout = Join-Path $repo "data\eval_results\source_specific_eval_background_stdout.log"
$stderr = Join-Path $repo "data\eval_results\source_specific_eval_background_stderr.log"

$process = Start-Process powershell `
  -WorkingDirectory $repo `
  -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runScript) `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -PassThru

Write-Host "Started PID=$($process.Id)"
Write-Host "STDOUT=$stdout"
Write-Host "STDERR=$stderr"
