# run_monitor.ps1 — Fetch and summarize @molt_cornelius posts
Set-Location "C:\Users\shach\PycharmProjects\mickey_london_lab"
& ".venv\Scripts\python.exe" monitor.py 2>&1 | Tee-Object -FilePath "monitor_log.txt"
