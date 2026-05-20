@echo off
setlocal

set "PS1=%~dp0run_demo_eval.ps1"
if not exist "%PS1%" (
  echo [ERROR] %PS1% not found
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo.
  echo [ERROR] Demo evaluation failed with exit code %EXITCODE%.
  pause
  exit /b %EXITCODE%
)

echo.
echo Demo evaluation finished.
pause
endlocal
