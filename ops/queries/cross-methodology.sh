#!/bin/bash
# Find notes that share a method but differ in confidence level
# Surfaces opportunities for validation or replication
VAULT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "=== Cross-Methodology Analysis ==="
echo "Notes by confidence level:"
for level in proven likely experimental speculative; do
  count=$(grep -rl "^confidence: $level" "$VAULT_ROOT/notes/" 2>/dev/null | wc -l | tr -d ' ')
  echo "  $level: $count"
done
echo ""
echo "Method notes with low confidence (candidates for validation):"
grep -rl "^type: method" "$VAULT_ROOT/notes/" 2>/dev/null | xargs grep -l "^confidence: \(experimental\|speculative\)" 2>/dev/null
