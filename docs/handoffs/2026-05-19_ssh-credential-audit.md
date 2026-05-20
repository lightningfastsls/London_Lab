# SSH credential leakage audit — handoff

**Date:** 2026-05-19
**Goal:** Inventory every file on this machine that mentions the rig's SSH coordinates and classify each by git-tracking status, so we know what would be exposed if any of those files got pushed to a remote.

> **Privacy note (read first):** This handoff intentionally uses placeholders below
> (`<RIG_IP>`, `<RIG_USER>`, `<RIG_HOST>`). Substitute your actual values
> at runtime via environment variables — the handoff itself stays clean even
> if it gets committed.

## Why this exists

During the 2026-05-19 Path C session, the rig's Tailscale SSH coordinates
were used in many bash commands. We want to know which files persistently
hold those coordinates and whether each is git-protected (gitignored), git-
tracked (will be pushed), or outside any repo.

The Tailscale CGNAT IP itself is not directly attackable from the public
internet (Tailscale IPs aren't routable), and SSH still requires key auth.
The risk here is **reduced obscurity** if those files end up on GitHub or
similar.

## Search targets

The patterns to search for (extended regex syntax):

| Pattern | What it catches |
|---|---|
| `<RIG_IP>` | Direct IP literal |
| `<RIG_USER>@<RIG_IP>` | Full SSH coordinate form |
| `100\.113\.[0-9]+\.[0-9]+` | Broader Tailscale CGNAT range — catches the IP even after a rotation |
| `<RIG_HOST>` | Hostname (`hostname` output on the rig) |
| `ssh <RIG_USER>` | Generic ssh-as-user invocations |
| `scp [^ ]* <RIG_USER>@` | scp commands targeting the rig |

## Search roots (in order of risk)

1. **`/home/shachar/projects/mickey_london_lab/`** — main repo + every worktree. Highest priority (these can be pushed).
2. **`/home/shachar/.claude/projects/-home-shachar-projects-mickey-london-lab/memory/`** — durable memory store. Local-only by design, but worth confirming.
3. **`/home/shachar/.claude/jobs/`** — session logs / scratch. Local-only, but ephemeral state may surprise you.
4. **`~/.bash_history`**, **`~/.zsh_history`**, **`~/.ssh/known_hosts`** — system-level files. Local-only.

## Run script — copy/paste into a fresh bash session

```bash
#!/usr/bin/env bash
set -euo pipefail

# ──── Substitute the real values here at runtime ────
RIG_IP="${RIG_IP:?set RIG_IP env var before running}"
RIG_USER="${RIG_USER:?set RIG_USER env var before running}"
RIG_HOST="${RIG_HOST:?set RIG_HOST env var before running}"

# Combined extended-regex pattern
RIG_IP_ESC="${RIG_IP//./\\.}"  # escape dots
PATTERN="${RIG_IP_ESC}|${RIG_USER}@|${RIG_HOST}|100\\.113\\.[0-9]+\\.[0-9]+"

# Output report
OUT="/tmp/ssh_audit_$(date +%Y%m%d_%H%M%S).md"
echo "# SSH credential audit — $(date)" > "$OUT"
echo "" >> "$OUT"

# Find candidate files
declare -A SEEN
collect() {
    local root="$1"
    if [ ! -d "$root" ]; then return; fi
    grep -rlEI "$PATTERN" "$root" \
        --exclude-dir='.git' \
        --exclude-dir='node_modules' \
        --exclude-dir='__pycache__' \
        --exclude-dir='.venv' \
        2>/dev/null | while IFS= read -r f; do
            echo "$f"
        done
}

ROOTS=(
    "/home/shachar/projects/mickey_london_lab"
    "/home/shachar/.claude/projects/-home-shachar-projects-mickey-london-lab/memory"
    "/home/shachar/.claude/jobs"
)

# Optional system files (only the ones likely to exist)
SYS_FILES=(
    "$HOME/.bash_history"
    "$HOME/.zsh_history"
    "$HOME/.ssh/config"
    "$HOME/.ssh/known_hosts"
)

declare -A HITS_BY_CATEGORY
HITS_BY_CATEGORY[TRACKED]=""
HITS_BY_CATEGORY[IGNORED]=""
HITS_BY_CATEGORY[UNTRACKED]=""
HITS_BY_CATEGORY[OUTSIDE_REPO]=""

classify_file() {
    local f="$1"
    if [ ! -f "$f" ]; then return; fi
    local dir
    dir=$(dirname "$f")

    # Find the nearest git work-tree by walking up
    local repo_root=""
    if repo_root=$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null); then
        # Inside a repo. Tracked, ignored, or untracked?
        if git -C "$repo_root" ls-files --error-unmatch "$f" >/dev/null 2>&1; then
            echo "TRACKED|$f"
        elif git -C "$repo_root" check-ignore -q "$f" 2>/dev/null; then
            echo "IGNORED|$f"
        else
            echo "UNTRACKED|$f"
        fi
    else
        echo "OUTSIDE_REPO|$f"
    fi
}

# Walk each root
{
    for root in "${ROOTS[@]}"; do
        collect "$root"
    done
    for sf in "${SYS_FILES[@]}"; do
        if [ -f "$sf" ] && grep -EI -q "$PATTERN" "$sf" 2>/dev/null; then
            echo "$sf"
        fi
    done
} | sort -u | while IFS= read -r file; do
    classify_file "$file"
done > /tmp/_ssh_audit_classified.txt

# Group and write report
for cat in TRACKED UNTRACKED OUTSIDE_REPO IGNORED; do
    {
        echo ""
        echo "## $cat"
        echo ""
    } >> "$OUT"
    cat_files=$(grep "^$cat|" /tmp/_ssh_audit_classified.txt | cut -d'|' -f2- || true)
    if [ -z "$cat_files" ]; then
        echo "_(none)_" >> "$OUT"
    else
        while IFS= read -r f; do
            echo "- \`$f\`" >> "$OUT"
            # First matching line for context
            grep -nEI "$PATTERN" "$f" 2>/dev/null | head -1 \
                | sed -E "s/^/    Example: /" >> "$OUT" || true
        done <<< "$cat_files"
    fi
done

echo "" >> "$OUT"
echo "## Triage rubric" >> "$OUT"
cat >> "$OUT" <<'TRIAGE'

| Category | Risk if repo gets pushed | Recommended action |
|---|---|---|
| **TRACKED** | High — info is committed and will be pushed | Strip the SSH coordinates (replace with `<RIG>` placeholders) or move them to a local-only file, then commit the cleanup |
| **UNTRACKED** | Medium — a future `git add -A` could sweep them in | Add explicit `.gitignore` entry for the path or move the file outside the repo |
| **IGNORED** | Low — already protected from accidental commit | Nothing to do, but double-check the rule still applies after refactors |
| **OUTSIDE_REPO** | Local-only on this machine | Standard OS-level security; nothing repo-related to do |

TRIAGE

echo "Report written to: $OUT"
```

## How to invoke

```bash
RIG_IP=100.113.224.57 RIG_USER=shachar RIG_HOST=cloudyclaude bash /tmp/ssh_audit.sh
```

Then open the report at `/tmp/ssh_audit_<timestamp>.md`.

## What "done" looks like

- A markdown report with four sections (TRACKED, UNTRACKED, IGNORED, OUTSIDE_REPO).
- Every section either has a list of files or `_(none)_`.
- Every TRACKED entry has a follow-up note: strip, replace, or accept.
- Every UNTRACKED entry has either an added `.gitignore` rule or a justification.

## Known pre-existing leak (already identified at handoff-write time)

`docs/handoffs/transformer-training-deployment.md` (in this worktree, written before this session) contains the SSH command in plaintext. Audit will surface it under TRACKED/UNTRACKED depending on the worktree's git state. Action: replace literal SSH coords with `ssh <rig>` placeholder and document the substitution somewhere local.

## False-positive watch list

The Tailscale CGNAT range `100.64.0.0/10` is the public-not-routable space — patterns matching `100.113.*` may also catch:
- Public IPv4 addresses in third-party docs/config (almost never — this range is reserved)
- Test fixtures using fake Tailscale-style IPs
Most hits should be real.

## Cleanup script (optional — re-run after applying fixes)

The same audit script is idempotent. Re-run it after each fix; the report should shrink monotonically. Stop when TRACKED is empty (or has explicit accepted-risk justifications) and UNTRACKED is either empty or has corresponding `.gitignore` rules.
