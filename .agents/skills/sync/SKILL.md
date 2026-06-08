---
name: sync
description: Sync machine-local state before push or after pull. "/sync push" copies auto-memory into repo and stages it. "/sync pull" restores auto-memory from repo and rebuilds qmd index. "/sync status" shows what's out of sync.
version: "1.0"
user-invocable: true
context: fork
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# /sync — Machine State Synchronization

Ensures Codex's machine-local state stays in sync with the git repo when moving between computers.

## What gets synced

| Asset | Repo location | Machine location | Direction |
|-------|--------------|-----------------|-----------|
| Auto-memory (MEMORY.md) | `docs/SESSION_MEMORY.md` | Global Codex projects dir* | Both |
| qmd vector index | (rebuilt from notes/) | `~/.qmd-cache/` or XDG_CACHE_HOME | Pull only (rebuild) |

*Global path: Find it by reading the auto-memory path from the system prompt or checking common locations:
- Windows: `D:/.Codex/projects/D--mickey-london-lab/memory/MEMORY.md`
- General pattern: `<drive>/.Codex/projects/<project-hash>/memory/MEMORY.md`

**Already in git** (no sync needed): `.Codex/skills/`, `.Codex/hooks/`, `.Codex/agents/`, `.Codex/settings.local.json`, `.mcp.json`

**Never synced** (secrets): `.env` — must be recreated manually on new machines.

**Rebuildable**: `.venv/` (pip install), qmd Vulkan patch (see MEMORY.md for instructions).

## Arguments

- `push` — Before committing/pushing: copy auto-memory → repo, stage the file
- `pull` — After pulling on a new machine: restore auto-memory from repo, rebuild qmd
- `status` — Show diff between auto-memory and repo mirror (no changes made)

## Procedure

### /sync push

1. Read the current auto-memory from the global Codex projects directory:
   `D:/.Codex/projects/D--mickey-london-lab/memory/MEMORY.md`

2. Read the repo mirror: `docs/SESSION_MEMORY.md`

3. If they differ:
   - Copy auto-memory content → `docs/SESSION_MEMORY.md` (preserve the sync-rule header)
   - Run: `git add docs/SESSION_MEMORY.md`
   - Report: "SESSION_MEMORY.md updated and staged. Remember to include it in your commit."

4. If identical: Report "Auto-memory already in sync."

5. Also check: are there any unstaged changes in `.Codex/` that should be committed?
   - Run: `git status .Codex/ .mcp.json`
   - If changes found, report them so the user can decide whether to stage them.

### /sync pull

1. Read the repo mirror: `docs/SESSION_MEMORY.md`

2. Write it to the global Codex projects directory:
   `D:/.Codex/projects/D--mickey-london-lab/memory/MEMORY.md`
   (Create the `memory/` directory if it doesn't exist)

3. Report: "Auto-memory restored from repo."

4. Rebuild the qmd semantic search index:
   - Run: `qmd update` (scans notes/ for new/changed files)
   - Run: `qmd embed` (generates vector embeddings for semantic search)
   - If either command fails, report the error but continue — qmd may not be installed on this machine yet.
   - If qmd is not installed: report "qmd not found — install with `npm install -g @tobilu/qmd` then re-run `/sync pull`."

5. Remind about manual steps:
   - ".env file — recreate with NOTION_TOKEN, ANTHROPIC_API_KEY, NOTION_KB_DATABASE_ID, NOTION_NOTES_DATABASE_ID"
   - "qmd Vulkan patch — if using AMD GPU, patch llm.js:253 (see MEMORY.md)"
   - "Python venv — run `python -m venv .venv && pip install -r requirements.txt`"

### /sync status

1. Read both files (auto-memory and repo mirror)
2. Compare them (use diff or manual comparison)
3. Report:
   - Whether they match or differ
   - Last modified timestamps if available
   - Any untracked/modified files in `.Codex/`
