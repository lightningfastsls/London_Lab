#!/bin/bash
# Find hypotheses that haven't been updated or connected
VAULT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "=== Stale Hypotheses ==="
echo "Hypotheses with speculative confidence:"
grep -rl "^type: hypothesis" "$VAULT_ROOT/notes/" 2>/dev/null | while read f; do
  confidence=$(grep "^confidence:" "$f" 2>/dev/null | head -1 | sed 's/confidence: //')
  links=$(grep -c "\[\[" "$f" 2>/dev/null || echo 0)
  basename=$(basename "$f" .md)
  echo "  $basename | confidence: $confidence | links: $links"
done
