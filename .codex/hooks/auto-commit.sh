#!/usr/bin/env bash
# Hook: Auto-commit vault changes
# Triggered on: PostToolUse (Write, async)

raw=$(cat)
[[ -z "$raw" ]] && exit 0

path=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('tool_input',{}).get('file_path',''))" "$raw" 2>/dev/null)
[[ -z "$path" ]] && exit 0

# Only auto-commit vault files (notes/, ops/, inbox/, self/)
[[ "$path" =~ /(notes|ops|inbox|self)/ ]] || exit 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Guard: only run in arscontexta vaults
[[ -f "$VAULT_ROOT/.arscontexta" ]] || exit 0

cd "$VAULT_ROOT"
git add "$path" 2>/dev/null
if ! git diff --cached --quiet 2>/dev/null; then
    basename=$(basename "$path")
    git commit -m "vault: update $basename" --no-verify 2>/dev/null
fi

exit 0
