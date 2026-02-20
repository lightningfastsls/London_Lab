#!/bin/bash
# Auto-Commit Hook - runs PostToolUse on Write (async)
VAULT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ ! -f "$VAULT_ROOT/.arscontexta" ]; then exit 0; fi
FILE_PATH="${CLAUDE_TOOL_INPUT_FILE_PATH:-}"
if [[ "$FILE_PATH" != *"/notes/"* ]] && [[ "$FILE_PATH" != *"/ops/"* ]] && [[ "$FILE_PATH" != *"/inbox/"* ]] && [[ "$FILE_PATH" != *"/self/"* ]] && [[ "$FILE_PATH" != *"\notes\\"* ]] && [[ "$FILE_PATH" != *"\ops\\"* ]] && [[ "$FILE_PATH" != *"\inbox\\"* ]] && [[ "$FILE_PATH" != *"\self\\"* ]]; then
  exit 0
fi
cd "$VAULT_ROOT" || exit 0
git add "$FILE_PATH" 2>/dev/null
if git diff --cached --quiet 2>/dev/null; then exit 0; fi
BASENAME=$(basename "$FILE_PATH")
git commit -m "vault: update $BASENAME" --no-verify 2>/dev/null
exit 0
