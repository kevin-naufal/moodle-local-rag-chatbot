@echo off
setlocal

set "REPO=C:\Users\Kevin\Downloads\my-llm"
set "RUN_SCRIPT=%REPO%\data\eval_results\run_source_specific_eval.ps1"
set "STDOUT=%REPO%\data\eval_results\source_specific_eval_background_stdout.log"
set "STDERR=%REPO%\data\eval_results\source_specific_eval_background_stderr.log"

cd /d "%REPO%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%RUN_SCRIPT%" 1> "%STDOUT%" 2> "%STDERR%"
