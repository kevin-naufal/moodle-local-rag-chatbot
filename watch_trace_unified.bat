@echo off
setlocal
set "PS1=%~dp0watch_trace_unified.ps1"
if not exist "%PS1%" (
  echo [ERROR] %PS1% not found
  pause
  exit /b 1
)
set "TAIL=%~1"
if "%TAIL%"=="" set "TAIL=30"
set "RID=%~2"
if "%RID%"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" -Tail %TAIL%
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" -Tail %TAIL% -RequestId "%RID%"
)
endlocal
