@echo off
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0check_plan_mode.ps1" 2>nul
exit /b 0
