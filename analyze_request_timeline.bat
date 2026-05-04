@echo off
setlocal
set "PS1=%~dp0analyze_request_timeline.ps1"
if "%~1"=="" (
  echo Latest request_id:
  powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" -Latest 1
  echo.
  echo Usage:
  echo   %~n0 request_id
  exit /b 0
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" -RequestId "%~1"
endlocal
