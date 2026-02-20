#!/bin/bash
# Validate Note Hook - runs PostToolUse on Write
# Checks that notes written to notes/ have required schema fields

VAULT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Check if this is an arscontexta vault
if [ ! -f "$VAULT_ROOT/.arscontexta" ]; then
  exit 0
fi

# Get the file path from environment
FILE_PATH="${CLAUDE_TOOL_INPUT_FILE_PATH:-}"

# Only validate files in notes/
case "$FILE_PATH" in
  */notes/*|*\notes\*) ;;
  *) exit 0 ;;
esac

# Only validate .md files
case "$FILE_PATH" in
  *.md) ;;
  *) exit 0 ;;
esac

# Check for YAML frontmatter
if ! head -1 "$FILE_PATH" 2>/dev/null | grep -q '^---$'; then
  echo "WARN: $FILE_PATH missing YAML frontmatter"
  exit 0
fi

# Check required fields
if ! grep -q '^description:' "$FILE_PATH" 2>/dev/null; then
  echo "WARN: $FILE_PATH missing required 'description' field"
fi

if ! grep -q '^topics:' "$FILE_PATH" 2>/dev/null; then
  echo "WARN: $FILE_PATH missing required 'topics' field"
fi

exit 0
