#!/usr/bin/env bash
# Hook: Remind about plan mode before editing code files
# Triggered on: PreToolUse (Edit|Write)

raw=$(cat)
[[ -z "$raw" ]] && exit 0

path=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('tool_input',{}).get('file_path',''))" "$raw" 2>/dev/null)
[[ -z "$path" ]] && exit 0

# Only remind for Python files, not plan files
if [[ "$path" =~ \.py$ ]] && [[ ! "$path" =~ plans/ ]]; then
    echo "[HOOK] Editing code file - did you use EnterPlanMode for non-trivial tasks?"
fi

exit 0
