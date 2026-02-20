#!/bin/bash
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$(dirname "$0")/check_agents_tag.ps1" 2>/dev/null
exit 0
