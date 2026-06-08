#!/usr/bin/env python3
"""Destructive-operation gate — PreToolUse(Bash).

Hard-blocks the high-blast-radius commands that the repo's history shows can
silently destroy data:
  * bulk git staging        git add -A | git add . | git add --all
  * working-tree resets     git reset --hard | git checkout . | git restore .
  * forced clean            git clean -f...
  * unsafe recursive delete rm -r / -rf  (unless the target is on the safe
                            allowlist: tmp dirs, .venv, build caches, etc.)

Why: commit 78d1c70 recorded an accidental `git add -A` that deleted 656 files
in USV_Detections/. CLAUDE.md "Git Data Safety" forbids bulk staging; this hook
makes that rule executable. See ops/health/repo-audit-2026-05-28.html.

Contract: reads one JSON event from stdin.
  * exit 2  -> BLOCK the tool call; stderr is shown to Claude.
  * exit 0  -> allow (also the fail-open path on any parse error, so a broken
               guard can never wedge the session).

Escape hatch: include the literal token  ALLOW_DESTRUCTIVE  anywhere in the
command (e.g. as a trailing `# ALLOW_DESTRUCTIVE`) to bypass after you have
reviewed it. Wire WITHOUT a trailing `|| true` in settings or exit 2 is lost.
"""

from __future__ import annotations

import json
import re
import sys

# --- bulk / working-tree git ops: always block (never routine) ---
GIT_BLOCK_PATTERNS = [
    (re.compile(r"\bgit\s+add\s+(?:[^|;&\n]*\s)?(?:-A\b|--all\b)"), "git add -A / --all"),
    (re.compile(r"\bgit\s+add\s+(?:[^|;&\n]*\s)?\.(?:\s|$)"), "git add ."),
    (re.compile(r"\bgit\s+reset\s+(?:[^|;&\n]*\s)?--hard\b"), "git reset --hard"),
    (re.compile(r"\bgit\s+clean\s+[^|;&\n]*-\w*f"), "git clean -f"),
    (re.compile(r"\bgit\s+checkout\s+(?:--\s+)?\.(?:\s|$)"), "git checkout ."),
    (re.compile(r"\bgit\s+restore\s+(?:--\s+)?\.(?:\s|$)"), "git restore ."),
]

# recursive rm: rm -r / -rf / -fr / -R (with or without f)
RM_RECURSIVE = re.compile(r"\brm\s+(?:-\w*r\w*|--recursive)\b", re.IGNORECASE)

# rm targets that are safe to nuke recursively (build/temp artifacts only)
SAFE_MARKERS = (
    "/tmp/", " /tmp", "CLAUDE_JOB_DIR", ".claude/jobs", "TMPDIR",
    ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".egg-info", "node_modules", "/dev/null", "build/", "dist/",
)


def block(reason: str, how: str) -> int:
    sys.stderr.write(
        f"[GUARD] Blocked a destructive command: {reason}.\n"
        f"{how}\n"
        f"Override (only after review): append  # ALLOW_DESTRUCTIVE  to the command.\n"
        f"Rationale: CLAUDE.md 'Git Data Safety' (commit 78d1c70 lost 656 files this way).\n"
    )
    return 2


def evaluate(command: str) -> int:
    if "ALLOW_DESTRUCTIVE" in command:
        return 0

    for pattern, label in GIT_BLOCK_PATTERNS:
        if pattern.search(command):
            return block(
                label,
                "Stage/restore by EXPLICIT path instead (e.g. `git add path/to/file`). "
                "Run `git status` first and review `git diff --cached --stat` for unexpected deletions.",
            )

    if RM_RECURSIVE.search(command):
        if any(marker in command for marker in SAFE_MARKERS):
            return 0  # recursive delete confined to a known-safe artifact path
        return block(
            "recursive rm outside the safe allowlist",
            "List the target first (`ls <dir>`), delete by explicit path, and confirm with the user "
            "if it holds data. Safe-prefix recursive deletes (tmp, .venv, __pycache__, build/) are allowed.",
        )

    return 0


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        event = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return 0  # fail-open

    if not isinstance(event, dict):
        return 0
    if event.get("hook_event_name") != "PreToolUse":
        return 0
    if event.get("tool_name") != "Bash":
        return 0

    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0
    command = str(tool_input.get("command", ""))
    if not command.strip():
        return 0

    try:
        return evaluate(command)
    except Exception:
        return 0  # never let a guard bug wedge the session


if __name__ == "__main__":
    sys.exit(main())
