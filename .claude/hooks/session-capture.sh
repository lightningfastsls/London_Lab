#!/bin/bash
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$(dirname "$0")/session-capture.ps1" 2>/dev/null
exit 0
