#\!/bin/bash
# Session Orient Hook - runs at SessionStart
VAULT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ \! -f "$VAULT_ROOT/.arscontexta" ]; then exit 0; fi
echo "=== Session Orient ==="
if [ -f "$VAULT_ROOT/ops/goals.md" ]; then
  echo ""
  echo "Current Goals:"
  grep -A 20 "## Active Threads" "$VAULT_ROOT/ops/goals.md" 2>/dev/null | head -15
fi
if [ -f "$VAULT_ROOT/ops/reminders.md" ]; then
  OVERDUE=$(grep -c "^- \[ \]" "$VAULT_ROOT/ops/reminders.md" 2>/dev/null || echo 0)
  if [ "$OVERDUE" -gt 0 ]; then echo ""; echo "Reminders: $OVERDUE pending"; fi
fi
NOTES_COUNT=$(find "$VAULT_ROOT/notes/" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
INBOX_COUNT=$(find "$VAULT_ROOT/inbox/" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
OBS_COUNT=$(find "$VAULT_ROOT/ops/observations/" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
TENSION_COUNT=$(find "$VAULT_ROOT/ops/tensions/" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "Vault: $NOTES_COUNT notes | $INBOX_COUNT inbox | $OBS_COUNT observations | $TENSION_COUNT tensions"
if [ "$INBOX_COUNT" -ge 3 ]; then echo "TRIGGER: Inbox has $INBOX_COUNT items"; fi
if [ "$OBS_COUNT" -ge 10 ]; then echo "TRIGGER: $OBS_COUNT pending observations"; fi
if [ "$TENSION_COUNT" -ge 5 ]; then echo "TRIGGER: $TENSION_COUNT open tensions"; fi
echo "=== End Orient ==="
