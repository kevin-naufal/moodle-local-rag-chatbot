$ErrorActionPreference = "Stop"

$repo = "C:\Users\Kevin\Downloads\my-llm"
Set-Location $repo

powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\scripts\run_demo_eval.ps1" `
  -Profile prod `
  -Dataset "$repo\data\answer_run_questions\ch03_running_time_source_specific_3q.json" `
  -DataDir "$repo\data\eval_ch03_only" `
  -Runs 3 `
  -Modes "llm_only,rag_bert,rag_msmarco" `
  -ChatModels "qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b" `
  -ForceNewAnswerRuns `
  -SkipPreparse `
  -SkipModelPull `
  -CleanupAfterRun
