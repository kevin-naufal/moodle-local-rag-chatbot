@echo off
setlocal

set "REPO=C:\Users\Kevin\Downloads\my-llm"
set "WORKER=%REPO%\data\eval_results\run_source_specific_eval_worker.cmd"
set "STDOUT=%REPO%\data\eval_results\source_specific_eval_background_stdout.log"
set "STDERR=%REPO%\data\eval_results\source_specific_eval_background_stderr.log"

cd /d "%REPO%"
start "source-specific-eval" /min "%WORKER%"

echo Started detached eval window.
echo STDOUT=%STDOUT%
echo STDERR=%STDERR%
