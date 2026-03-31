# run_monitor.ps1 — Fetch and summarize @molt_cornelius posts
Set-Location "\\wsl$\Ubuntu\home\shachar\projects\mickey_london_lab"
& ".venv\bin\python" monitor.py 2>&1 | Tee-Object -FilePath "monitor_log.txt"
