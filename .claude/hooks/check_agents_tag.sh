#!/usr/bin/env bash
# Hook: Check if response ends with **Agents:** tag
# Triggered on: Stop event

raw=$(cat)
[[ -z "$raw" ]] && exit 0

transcript_path=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('transcript_path',''))" "$raw" 2>/dev/null)
[[ -z "$transcript_path" ]] && exit 0

transcript=$(cat "$transcript_path" 2>/dev/null || echo "")
if [[ -n "$transcript" ]] && ! echo "$transcript" | grep -q '\*\*Agents:\*\*'; then
    echo "[HOOK] Remember to end your response with: **Agents:** [list or None]"
fi

exit 0
