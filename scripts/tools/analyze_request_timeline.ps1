param(
  [string]$RequestId = '',
  [int]$Latest = 1,
  [switch]$ShowContent
)

$phpLog = 'C:\xampp\moodledata\local_chatbot\logs\e2e_trace_php.jsonl'
$pyLog = 'C:\xampp\moodledata\local_chatbot\logs\e2e_trace_python.jsonl'

function Read-Jsonl($path) {
  if (-not (Test-Path -LiteralPath $path)) { return @() }
  $rows = @()
  Get-Content -LiteralPath $path | ForEach-Object {
    $line = $_.Trim()
    if ($line) {
      try { $rows += ($line | ConvertFrom-Json) } catch {}
    }
  }
  return $rows
}

$phpRows = Read-Jsonl $phpLog
$pyRows = Read-Jsonl $pyLog
$all = @($phpRows + $pyRows) | Where-Object { $_.request_id }
if (-not $all -or $all.Count -eq 0) {
  Write-Host 'No trace rows found.' -ForegroundColor Yellow
  exit 1
}

if (-not $RequestId) {
  $ids = $all |
    Group-Object -Property request_id |
    ForEach-Object {
      [pscustomobject]@{
        request_id = $_.Name
        last_ts = ($_.Group | Sort-Object ts_ms | Select-Object -Last 1).ts_ms
      }
    } |
    Sort-Object last_ts -Descending |
    Select-Object -First ([Math]::Max(1, $Latest))
  foreach ($id in $ids) {
    Write-Host $id.request_id
  }
  exit 0
}

$rows = $all | Where-Object { $_.request_id -eq $RequestId } | Sort-Object ts_ms
if (-not $rows -or $rows.Count -eq 0) {
  Write-Host "request_id not found: $RequestId" -ForegroundColor Yellow
  exit 1
}

$firstTs = [int64]($rows[0].ts_ms)
$phpChatSuccess = $rows | Where-Object { $_.event -eq 'chat_request_success' } | Select-Object -First 1
$phpRunnerSuccess = $rows | Where-Object { $_.event -eq 'php_runner_exec_success' } | Select-Object -First 1
$pyReqSuccess = $rows | Where-Object { $_.event -eq 'python_request_success' } | Select-Object -First 1
$ollamaCalls = $rows | Where-Object { $_.event -eq 'ollama_llm_invoke_success' }
$health = $rows | Where-Object { $_.event -like 'ollama_healthcheck_*' } | Select-Object -First 1

Write-Host "request_id: $RequestId" -ForegroundColor Cyan
Write-Host "question: $($rows[0].question_number) attempt: $($rows[0].attempt)" -ForegroundColor Cyan
Write-Host ''
Write-Host 'Timeline:' -ForegroundColor Green

foreach ($r in $rows) {
  $delta = [int64]$r.ts_ms - $firstTs
  $dur = if ($null -ne $r.duration_ms -and "$($r.duration_ms)" -ne '') { " dur=$([int64]$r.duration_ms)ms" } else { '' }
  Write-Host ("+{0,7}ms  [{1}] {2}{3}" -f $delta, $r.layer, $r.event, $dur)
}

Write-Host ''
Write-Host 'Summary:' -ForegroundColor Green
if ($phpChatSuccess) { Write-Host ("- Moodle(PHP) total: {0} ms" -f $phpChatSuccess.duration_ms) }
if ($phpRunnerSuccess) { Write-Host ("- PHP -> Python exec: {0} ms" -f $phpRunnerSuccess.duration_ms) }
if ($pyReqSuccess) { Write-Host ("- Python total: {0} ms" -f $pyReqSuccess.duration_ms) }
if ($health) { Write-Host ("- Ollama healthcheck: {0} ms" -f ($health.duration_ms)) }

if ($ollamaCalls -and $ollamaCalls.Count -gt 0) {
  $sum = ($ollamaCalls | Measure-Object -Property duration_ms -Sum).Sum
  Write-Host ("- Ollama invoke calls: {0} call(s), total {1} ms" -f $ollamaCalls.Count, $sum)
  $i = 1
  foreach ($c in $ollamaCalls) {
    Write-Host ("  call#{0}: {1} ms (prompt_chars={2}, answer_chars={3})" -f $i, $c.duration_ms, $c.prompt_chars, $c.answer_chars)
    $i++
  }
}

if ($phpChatSuccess -and $ollamaCalls -and $pyReqSuccess) {
  $sumCalls = ($ollamaCalls | Measure-Object -Property duration_ms -Sum).Sum
  $pyNonLlm = [int64]$pyReqSuccess.duration_ms - [int64]$sumCalls
  Write-Host ("- Python non-LLM overhead: {0} ms" -f $pyNonLlm)
  $phpOver = [int64]$phpChatSuccess.duration_ms - [int64]$pyReqSuccess.duration_ms
  Write-Host ("- PHP overhead outside Python: {0} ms" -f $phpOver)
}

if ($ShowContent) {
  Write-Host ''
  Write-Host 'Content Trace:' -ForegroundColor Green
  $q = $rows | Where-Object { $_.event -eq 'python_query_text' } | Select-Object -First 1
  if ($q) {
    Write-Host '[python_query_text]'
    Write-Host ($q.query_text)
    Write-Host ''
  }

  $prompts = $rows | Where-Object { $_.event -eq 'ollama_llm_prompt' }
  $idx = 1
  foreach ($p in $prompts) {
    Write-Host ("[ollama_llm_prompt #{0}] chars={1}" -f $idx, $p.prompt_chars)
    Write-Host ($p.prompt_text)
    Write-Host ''
    $idx++
  }

  $raws = $rows | Where-Object { $_.event -eq 'ollama_llm_raw_response' }
  $idx = 1
  foreach ($rw in $raws) {
    Write-Host ("[ollama_llm_raw_response #{0}] raw_chars={1} cleaned_chars={2}" -f $idx, $rw.raw_answer_chars, $rw.cleaned_answer_chars)
    Write-Host '-- cleaned answer --'
    Write-Host ($rw.cleaned_answer_text)
    Write-Host ''
    $idx++
  }

  $finalPy = $rows | Where-Object { $_.event -eq 'python_response_emit' } | Select-Object -Last 1
  if ($finalPy) {
    Write-Host '[python_response_emit]'
    Write-Host ($finalPy.answer_text)
    Write-Host ''
  }

  $finalPhp = $rows | Where-Object { $_.event -eq 'chat_request_success' } | Select-Object -Last 1
  if ($finalPhp -and $null -ne $finalPhp.answer_text) {
    Write-Host '[php chat_request_success answer_text]'
    Write-Host ($finalPhp.answer_text)
    Write-Host ''
  }
}
