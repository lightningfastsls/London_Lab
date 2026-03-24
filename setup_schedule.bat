@echo off
echo Setting up daily scheduled tasks for @molt_cornelius monitor...
echo.

schtasks /create /tn "MoltMonitor" /tr "powershell.exe -ExecutionPolicy Bypass -File C:\Users\shach\PycharmProjects\mickey_london_lab\run_monitor.ps1" /sc daily /st 19:00 /f
echo.

schtasks /create /tn "MoltAssess" /tr "powershell.exe -ExecutionPolicy Bypass -File C:\Users\shach\PycharmProjects\mickey_london_lab\assess_summary.ps1" /sc daily /st 19:10 /f
echo.

echo Done! Tasks scheduled:
echo   MoltMonitor - daily at 19:00 (fetch + summarize)
echo   MoltAssess  - daily at 19:10 (Claude KG assessment)
echo.
echo To verify: schtasks /query /tn "MoltMonitor"
echo To remove:  schtasks /delete /tn "MoltMonitor" /f
echo             schtasks /delete /tn "MoltAssess" /f
pause
