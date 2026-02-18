#!/bin/bash
# Show note counts per topic map and identify gaps
VAULT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "=== Topic Coverage ==="
for moc in "$VAULT_ROOT/notes/"*.md; do
  [ -f "$moc" ] || continue
  grep -q "^type: moc" "$moc" || continue
  moc_name=$(basename "$moc" .md)
  count=$(grep -rl "\[\[$moc_name\]\]" "$VAULT_ROOT/notes/" 2>/dev/null | grep -v "$moc" | wc -l | tr -d ' ')
  echo "  $moc_name: $count notes"
done
echo ""
echo "Notes not assigned to any topic map:"
for f in "$VAULT_ROOT/notes/"*.md; do
  [ -f "$f" ] || continue
  grep -q "^type: moc" "$f" && continue
  if ! grep -q "^topics:" "$f" 2>/dev/null; then
    echo "  $(basename "$f" .md)"
  fi
done
