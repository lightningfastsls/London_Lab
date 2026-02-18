#\!/bin/bash
# Session Capture Hook - runs at Stop
VAULT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ \! -f "$VAULT_ROOT/.arscontexta" ]; then exit 0; fi
SESSION_DIR="$VAULT_ROOT/ops/sessions"
mkdir -p "$SESSION_DIR"
SESSION_ID="${CLAUDE_CONVERSATION_ID:-$(date +%Y%m%d-%H%M%S)}"
SESSION_FILE="$SESSION_DIR/${SESSION_ID}.json"
if [ \! -f "$SESSION_FILE" ]; then
  printf '{
  "session_id": "%s",
  "start_time": "%s",
  "notes_created": [],
  "notes_modified": [],
  "discoveries": [],
  "status": "completed"
}
' "$SESSION_ID" "$(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)" > "$SESSION_FILE"
fi
