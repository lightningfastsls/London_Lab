#!/usr/bin/env bash
# Hook: Session capture - save session metadata and bridging context on Stop
# Triggered on: Stop event (receives stdin JSON with transcript_path)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Guard: only run in arscontexta vaults
[[ -f "$VAULT_ROOT/.arscontexta" ]] || exit 0

# Read stdin
raw=$(cat)
[[ -z "$raw" ]] && exit 0

# Parse JSON with python3
transcript_path=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('transcript_path',''))" "$raw" 2>/dev/null)
[[ -z "$transcript_path" ]] && exit 0

SESSION_DIR="$VAULT_ROOT/ops/sessions"
mkdir -p "$SESSION_DIR"

# Session ID
session_id="${CLAUDE_CONVERSATION_ID:-$(date +%Y%m%d-%H%M%S)}"
timestamp=$(date '+%Y-%m-%dT%H:%M:%S')

# Save session metadata JSON
cat > "$SESSION_DIR/$session_id.json" <<ENDJSON
{
  "session_id": "$session_id",
  "timestamp": "$timestamp",
  "transcript_path": "$transcript_path",
  "status": "completed"
}
ENDJSON

# --- Write ops/last-session.md ---
LAST_SESSION="$VAULT_ROOT/ops/last-session.md"
summary=("Session $session_id ($timestamp)")

git_status=$(git -C "$VAULT_ROOT" status --porcelain 2>/dev/null || echo "")
if [[ -n "$git_status" ]]; then
    total=$(echo "$git_status" | wc -l)
    src_count=$(echo "$git_status" | grep -c "src/" 2>/dev/null || echo 0)
    test_count=$(echo "$git_status" | grep -c "tests/" 2>/dev/null || echo 0)
    note_count=$(echo "$git_status" | grep -c "notes/" 2>/dev/null || echo 0)
    ops_count=$(echo "$git_status" | grep -c "ops/" 2>/dev/null || echo 0)
    other_count=$((total - src_count - test_count - note_count - ops_count))

    parts=()
    [[ $src_count -gt 0 ]] && parts+=("$src_count src")
    [[ $test_count -gt 0 ]] && parts+=("$test_count tests")
    [[ $note_count -gt 0 ]] && parts+=("$note_count notes")
    [[ $ops_count -gt 0 ]] && parts+=("$ops_count ops")
    [[ $other_count -gt 0 ]] && parts+=("$other_count other")

    if [[ ${#parts[@]} -gt 0 ]]; then
        parts_str=$(IFS=', '; echo "${parts[*]}")
        summary+=("Files changed: $parts_str ($total total)")
    fi
else
    summary+=("No uncommitted changes at session end.")
fi

# Recent commits
recent=$(git -C "$VAULT_ROOT" log --oneline -3 --since="2 hours ago" 2>/dev/null || echo "")
if [[ -n "$recent" ]]; then
    summary+=("Recent commits:")
    while IFS= read -r c; do
        summary+=("  $c")
    done <<< "$recent"
fi

# Goals updated?
goals_diff=$(git -C "$VAULT_ROOT" diff --name-only HEAD -- "ops/goals.md" 2>/dev/null || echo "")
goals_staged=$(git -C "$VAULT_ROOT" diff --name-only --cached -- "ops/goals.md" 2>/dev/null || echo "")
if [[ -n "$goals_diff" || -n "$goals_staged" ]]; then
    summary+=("Goals updated: yes")
fi

printf '%s\n' "${summary[@]}" > "$LAST_SESSION"

# --- State Update Rule enforcement ---
all_changed=$(git -C "$VAULT_ROOT" diff --name-only HEAD 2>/dev/null || echo "")
all_staged=$(git -C "$VAULT_ROOT" diff --name-only --cached 2>/dev/null || echo "")
all_files="$all_changed"$'\n'"$all_staged"

has_substantial=$(echo "$all_files" | grep -c "^src/\|^notes/" 2>/dev/null || echo 0)
goals_updated=$(echo "$all_files" | grep -c "ops/goals.md" 2>/dev/null || echo 0)

if [[ $has_substantial -gt 0 ]] && [[ $goals_updated -eq 0 ]]; then
    echo "[HOOK] Session had substantial changes to src/ or notes/ but ops/goals.md was not updated. Did you complete a milestone?" >&2
fi

exit 0
