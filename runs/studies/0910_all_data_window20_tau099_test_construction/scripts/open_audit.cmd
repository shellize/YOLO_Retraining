@echo off
setlocal

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found on PATH.
  echo Open a Conda prompt and run this file again.
  exit /b 1
)

python "%~dp0serve_audit.py" %*
exit /b %errorlevel%
