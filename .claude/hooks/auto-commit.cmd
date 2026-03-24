@echo off
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0auto-commit.ps1" 2>nul
exit /b 0
