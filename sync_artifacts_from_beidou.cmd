@echo off
setlocal

"%SystemRoot%\System32\wsl.exe" --cd "%~dp0." -- bash ./sync_artifacts_from_beidou.sh %*
set "SYNC_EXIT=%ERRORLEVEL%"

endlocal & exit /b %SYNC_EXIT%
