#!/usr/bin/env bash
# Hook: Session orientation - show vault state at session start
# Triggered on: SessionStart (no stdin)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Guard: only run in arscontexta vaults
[[ -f "$VAULT_ROOT/.arscontexta" ]] || exit 0

TODAY=$(date +%Y-%m-%d)

echo "=== Session Orient ==="

# --- Active Goals ---
GOALS_FILE="$VAULT_ROOT/ops/goals.md"
GOAL_LINES=()
if [[ -f "$GOALS_FILE" ]]; then
    in_section=false
    while IFS= read -r line; do
        if [[ "$line" == "## Active Threads"* ]]; then
            in_section=true
            continue
        fi
        if $in_section; then
            if [[ "$line" == "## "* ]] && [[ "$line" != "## Active Threads"* ]]; then
                break
            fi
            trimmed=$(echo "$line" | sed 's/^[[:space:]]*//')
            if [[ -n "$trimmed" ]]; then
                GOAL_LINES+=("$line")
            fi
        fi
    done < "$GOALS_FILE"
    if [[ ${#GOAL_LINES[@]} -gt 0 ]]; then
        echo ""
        echo "Current Goals:"
        for gl in "${GOAL_LINES[@]}"; do
            echo "$gl"
        done
    fi
fi

# --- Reminders with overdue detection ---
REMINDERS_FILE="$VAULT_ROOT/ops/reminders.md"
if [[ -f "$REMINDERS_FILE" ]]; then
    overdue_items=()
    soon_items=()
    future_count=0
    today_epoch=$(date -d "$TODAY" +%s 2>/dev/null || date +%s)

    while IFS= read -r line; do
        # Match: - [ ] YYYY-MM-DD: Description
        if [[ "$line" =~ ^-\ \[\ \]\ ([0-9]{4}-[0-9]{2}-[0-9]{2}):\ (.*) ]]; then
            date_str="${BASH_REMATCH[1]}"
            desc="${BASH_REMATCH[2]}"
            reminder_epoch=$(date -d "$date_str" +%s 2>/dev/null || echo "")
            if [[ -n "$reminder_epoch" ]]; then
                days_until=$(( (reminder_epoch - today_epoch) / 86400 ))
                if [[ $days_until -lt 0 ]]; then
                    days_ago=$(( -days_until ))
                    overdue_items+=("  OVERDUE ($days_ago days): $desc")
                elif [[ $days_until -eq 0 ]]; then
                    soon_items+=("  DUE TODAY: $desc")
                elif [[ $days_until -le 3 ]]; then
                    soon_items+=("  DUE in $days_until days: $desc")
                else
                    ((future_count++)) || true
                fi
            else
                ((future_count++)) || true
            fi
        fi
    done < "$REMINDERS_FILE"

    if [[ ${#overdue_items[@]} -gt 0 || ${#soon_items[@]} -gt 0 || $future_count -gt 0 ]]; then
        echo ""
        echo "Reminders:"
        for item in "${overdue_items[@]}"; do echo "$item"; done
        for item in "${soon_items[@]}"; do echo "$item"; done
        if [[ $future_count -gt 0 ]]; then
            echo "  $future_count more scheduled (distant)"
        fi
    fi
fi

# --- Last Session Summary ---
LAST_SESSION_FILE="$VAULT_ROOT/ops/last-session.md"
if [[ -f "$LAST_SESSION_FILE" ]]; then
    shown=0
    in_frontmatter=false
    while IFS= read -r line; do
        trimmed=$(echo "$line" | sed 's/^[[:space:]]*//')
        if [[ "$trimmed" == "---" ]] && [[ $shown -eq 0 ]]; then
            if $in_frontmatter; then in_frontmatter=false; else in_frontmatter=true; fi
            continue
        fi
        if ! $in_frontmatter && [[ -n "$trimmed" ]] && [[ $shown -lt 5 ]]; then
            if [[ $shown -eq 0 ]]; then
                echo ""
                echo "Last session:"
            fi
            echo "  $trimmed"
            ((shown++)) || true
        fi
    done < "$LAST_SESSION_FILE"
fi

# --- Pending Tasks ---
TASKS_FILE="$VAULT_ROOT/ops/tasks.md"
if [[ -f "$TASKS_FILE" ]]; then
    pending_tasks=()
    inprogress_tasks=()
    in_pending=false
    in_inprogress=false

    while IFS= read -r line; do
        if [[ "$line" == "## Pending"* ]]; then in_pending=true; in_inprogress=false; continue; fi
        if [[ "$line" == "## In Progress"* ]]; then in_inprogress=true; in_pending=false; continue; fi
        if [[ "$line" == "## "* ]]; then in_pending=false; in_inprogress=false; continue; fi

        trimmed=$(echo "$line" | sed 's/^[[:space:]]*//')
        if [[ -n "$trimmed" ]] && [[ "$trimmed" != "(none)" ]]; then
            if $in_pending && [[ "$trimmed" == "- "* ]]; then pending_tasks+=("$trimmed"); fi
            if $in_inprogress && [[ "$trimmed" == "- "* ]]; then inprogress_tasks+=("$trimmed"); fi
        fi
    done < "$TASKS_FILE"

    if [[ ${#inprogress_tasks[@]} -gt 0 || ${#pending_tasks[@]} -gt 0 ]]; then
        echo ""
        echo "Tasks:"
        for t in "${inprogress_tasks[@]}"; do echo "  [IN PROGRESS] ${t#- }"; done
        for t in "${pending_tasks[@]}"; do echo "  [PENDING] ${t#- }"; done
    fi
fi

# --- Vault Counts ---
notes_count=0
inbox_count=0
pending_obs_count=0
pending_tension_count=0

NOTES_DIR="$VAULT_ROOT/notes"
INBOX_DIR="$VAULT_ROOT/inbox"
OBS_DIR="$VAULT_ROOT/ops/observations"
TENSION_DIR="$VAULT_ROOT/ops/tensions"

if [[ -d "$NOTES_DIR" ]]; then
    notes_count=$(find "$NOTES_DIR" -name "*.md" -type f 2>/dev/null | wc -l)
fi
if [[ -d "$INBOX_DIR" ]]; then
    inbox_count=$(find "$INBOX_DIR" -name "*.md" -type f 2>/dev/null | wc -l)
fi

# Count pending observations
if [[ -d "$OBS_DIR" ]]; then
    for f in "$OBS_DIR"/*.md; do
        [[ -f "$f" ]] || continue
        if head -20 "$f" | grep -q "status: pending"; then
            ((pending_obs_count++)) || true
        fi
    done
fi

# Count pending tensions
if [[ -d "$TENSION_DIR" ]]; then
    for f in "$TENSION_DIR"/*.md; do
        [[ -f "$f" ]] || continue
        if head -20 "$f" | grep -q "status: pending"; then
            ((pending_tension_count++)) || true
        fi
    done
fi

echo ""
echo "Vault: $notes_count notes | $inbox_count inbox | $pending_obs_count pending observations | $pending_tension_count pending tensions"

# --- Trigger warnings ---
inbox_threshold=3
obs_threshold=10
tension_threshold=5

QUEUE_FILE="$VAULT_ROOT/ops/queue/queue.json"
if [[ -f "$QUEUE_FILE" ]]; then
    thresholds=$(python3 -c "
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    mc=d.get('maintenance_conditions',{})
    print(mc.get('inbox_threshold',''))
    print(mc.get('observation_threshold',''))
    print(mc.get('tension_threshold',''))
except: print('\n\n')
" "$QUEUE_FILE" 2>/dev/null)
    if [[ -n "$thresholds" ]]; then
        it=$(echo "$thresholds" | sed -n '1p')
        ot=$(echo "$thresholds" | sed -n '2p')
        tt=$(echo "$thresholds" | sed -n '3p')
        [[ -n "$it" ]] && inbox_threshold=$it
        [[ -n "$ot" ]] && obs_threshold=$ot
        [[ -n "$tt" ]] && tension_threshold=$tt
    fi
fi

if [[ $inbox_count -ge $inbox_threshold ]]; then echo "TRIGGER: Inbox has $inbox_count items (threshold: $inbox_threshold)"; fi
if [[ $pending_obs_count -ge $obs_threshold ]]; then echo "TRIGGER: $pending_obs_count pending observations (threshold: $obs_threshold)"; fi
if [[ $pending_tension_count -ge $tension_threshold ]]; then echo "TRIGGER: $pending_tension_count pending tensions (threshold: $tension_threshold)"; fi

# --- Knowledge Activation ---
RELEVANCE_FILE="$VAULT_ROOT/ops/session-relevance.md"
VAULT_SEARCH="$VAULT_ROOT/ops/scripts/vault-search.mjs"
INDEX_FILE="$VAULT_ROOT/ops/cache/topic-map-index.json"

if command -v node &>/dev/null && [[ -f "$VAULT_SEARCH" ]] && [[ ${#GOAL_LINES[@]} -gt 0 ]]; then
    # Auto-regenerate index if stale (>24h) or missing
    if [[ ! -f "$INDEX_FILE" ]] || [[ -n "$(find "$INDEX_FILE" -mmin +1440 2>/dev/null)" ]]; then
        if ! node "$VAULT_ROOT/ops/scripts/topic-map-index.mjs" "$VAULT_ROOT/notes" "$INDEX_FILE" 2>/dev/null; then
            # Index generation failed — use stale index if available, or skip Layer 1
            if [[ ! -f "$INDEX_FILE" ]]; then
                echo "Warning: topic-map index generation failed and no stale index available" >&2
            fi
        fi
    fi

    total_notes=0
    thread_count=0
    relevance_content="# Session Relevance Brief\n<!-- Generated: $(date '+%Y-%m-%d %H:%M') -->\n<!-- Method: topic-map-traversal + ripgrep -->\n"

    for gl in "${GOAL_LINES[@]}"; do
        # Extract thread title: - **Title** -- description
        if [[ "$gl" =~ ^[[:space:]]*-[[:space:]]+\*\*(.+)\*\*[[:space:]]*--[[:space:]]*(.+)$ ]]; then
            title="${BASH_REMATCH[1]}"
            desc="${BASH_REMATCH[2]}"
        elif [[ "$gl" =~ ^[[:space:]]*-[[:space:]]+(.+)[[:space:]]*--[[:space:]]*(.+)$ ]]; then
            title="${BASH_REMATCH[1]}"
            desc="${BASH_REMATCH[2]}"
        else
            continue
        fi

        ((thread_count++)) || true
        [[ $thread_count -gt 5 ]] && break

        relevance_content+="\n## $title\n"

        # Search via vault-search.mjs (topic map traversal + ripgrep)
        search_results=$(node "$VAULT_SEARCH" --query "$title" --context "${desc:0:100}" --limit 3 2>/dev/null || echo "[]")
        found_any=false

        if [[ "$search_results" != "[]" ]]; then
            while IFS= read -r line; do
                note_title=$(echo "$line" | sed -n 's/.*"note": *"\([^"]*\)".*/\1/p')
                note_desc=$(echo "$line" | sed -n 's/.*"description": *"\([^"]*\)".*/\1/p')
                note_type=$(echo "$line" | sed -n 's/.*"type": *"\([^"]*\)".*/\1/p')
                note_section=$(echo "$line" | sed -n 's/.*"section": *"\([^"]*\)".*/\1/p')
                note_ctx=$(echo "$line" | sed -n 's/.*"context_phrase": *"\([^"]*\)".*/\1/p')
                if [[ -n "$note_title" ]]; then
                    ctx_display=""
                    [[ -n "$note_ctx" ]] && ctx_display=" -- $note_ctx"
                    type_display=""
                    [[ -n "$note_type" ]] && type_display=" ($note_type)"
                    relevance_content+="- \"$note_title\"${type_display}${ctx_display}\n"
                    ((total_notes++)) || true
                    found_any=true
                fi
            done < <(echo "$search_results" | python3 -c "
import json, sys
try:
    results = json.load(sys.stdin)
    for r in results:
        print(json.dumps(r))
except: pass
" 2>/dev/null)
        fi

        if ! $found_any; then
            relevance_content+="- No strong matches above relevance threshold.\n"
        fi
    done

    echo -e "$relevance_content" > "$RELEVANCE_FILE"
    echo "Knowledge activation: $total_notes notes surfaced for $thread_count threads"

elif command -v qmd &>/dev/null && [[ ${#GOAL_LINES[@]} -gt 0 ]]; then
    # Fallback: qmd (demoted but still available)
    total_notes=0
    thread_count=0
    relevance_content="# Session Relevance Brief\n<!-- Generated: $(date '+%Y-%m-%d %H:%M') -->\n<!-- Method: qmd fallback -->\n"

    for gl in "${GOAL_LINES[@]}"; do
        if [[ "$gl" =~ ^[[:space:]]*-[[:space:]]+\*\*(.+)\*\*[[:space:]]*--[[:space:]]*(.+)$ ]]; then
            title="${BASH_REMATCH[1]}"
            desc="${BASH_REMATCH[2]}"
        elif [[ "$gl" =~ ^[[:space:]]*-[[:space:]]+(.+)[[:space:]]*--[[:space:]]*(.+)$ ]]; then
            title="${BASH_REMATCH[1]}"
            desc="${BASH_REMATCH[2]}"
        else
            continue
        fi

        ((thread_count++)) || true
        [[ $thread_count -gt 5 ]] && break

        relevance_content+="\n## $title\n"
        clean_title=$(echo "$title" | sed 's/[^a-zA-Z0-9 ]//g')
        found_any=false

        kw_results=$(qmd search "$clean_title" --limit 3 --json 2>/dev/null || echo "[]")
        if [[ "$kw_results" == "["* ]]; then
            while IFS= read -r note_title; do
                [[ -n "$note_title" ]] && {
                    relevance_content+="- **$note_title** (keyword)\n"
                    ((total_notes++)) || true
                    found_any=true
                }
            done < <(python3 -c "import json,sys; [print(r.get('title','')) for r in json.loads(sys.argv[1]) if r.get('title')]" "$kw_results" 2>/dev/null)
        fi

        if ! $found_any; then
            relevance_content+="- No strong matches above relevance threshold.\n"
        fi
    done

    echo -e "$relevance_content" > "$RELEVANCE_FILE"
    echo "Knowledge activation (qmd fallback): $total_notes notes surfaced for $thread_count threads"
else
    echo -e "# Session Relevance Brief\n<!-- Generated: $(date '+%Y-%m-%d %H:%M') -->\n\nKnowledge activation unavailable (node and qmd both missing)." > "$RELEVANCE_FILE"
fi

echo "=== End Orient ==="
exit 0
