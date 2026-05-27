@echo off
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0check_agents_tag.ps1" 2>nul
exit /b 0
