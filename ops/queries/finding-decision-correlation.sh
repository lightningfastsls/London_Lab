#!/bin/bash
# Show which findings led to which decisions
VAULT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "=== Finding-Decision Correlation ==="
echo "Decisions that reference findings:"
grep -rl "^type: decision" "$VAULT_ROOT/notes/" 2>/dev/null | while read f; do
  basename=$(basename "$f" .md)
  finding_links=$(grep -o "\[\[[^]]*\]\]" "$f" 2>/dev/null | tr -d '[]')
  echo "  Decision: $basename"
  echo "    References: $finding_links"
  echo ""
done
