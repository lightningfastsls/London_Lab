@echo off
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0session-orient.ps1" 2>nul
exit /b 0
