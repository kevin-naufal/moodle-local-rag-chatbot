@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS1=%SCRIPT_DIR%watch_chatbot_logs.ps1"

if not exist "%PS1%" (
  echo [ERROR] Script not found: %PS1%
  pause
  exit /b 1
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"

set "TAIL=%~2"
if "%TAIL%"=="" set "TAIL=50"

echo Starting chatbot log watcher...
echo Mode: %MODE%
echo Tail: %TAIL%
echo.
echo Press Ctrl+C to stop.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" -Mode %MODE% -Tail %TAIL%

endlocal
