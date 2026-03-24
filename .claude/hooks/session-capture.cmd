@echo off
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0session-capture.ps1" 2>nul
exit /b 0
