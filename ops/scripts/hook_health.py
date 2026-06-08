#!/usr/bin/env python3
"""Hook & skill health beacon.

Answers the question the 2026-05-28 audit had to commission a whole
investigation to answer: *is my automation actually working?* Three of the
repo's PowerShell hooks had been silently dead for ~2 months because they ran on
the WSL->Windows bridge and could not read POSIX paths — nothing reported it.

This makes "let it bake" safe by making it observable.

Modes
-----
  --quick  (default): wiring + syntax only, NO hook execution. Cheap enough for
           SessionStart (a handful of py_compile/bash -n checks). Catches: a hook
           silently removed from settings, a script file missing, a syntax error.
           Warns; never blocks.
  --full :  everything in --quick, PLUS a synthetic-payload self-test that
           actually executes each side-effect-free hook and asserts it emits its
           expected signal (the deterministic "is it dead?" catch), PLUS a scan
           of recent transcripts for skills defined but unused in 30 days.

Exit code is always 0 (advisory). With --quiet, prints only when something
FAILs, so it is safe to wire into SessionStart.

Usage:
  python3 ops/scripts/hook_health.py            # quick, verbose
  python3 ops/scripts/hook_health.py --quick --quiet   # SessionStart beacon
  python3 ops/scripts/hook_health.py --full     # weekly cron / on demand
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR", str(Path(__file__).resolve().parents[2])))
HOOKS = ROOT / ".claude" / "hooks"
SETTINGS = ROOT / ".claude" / "settings.local.json"
SKILLS_DIR = ROOT / ".claude" / "skills"
TRANSCRIPT_GLOB = os.path.expanduser(
    "~/.claude/projects/-home-shachar-projects-mickey-london-lab/*.jsonl"
)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def load_wired_commands() -> str:
    """All hook command strings concatenated — used to detect wiring drift."""
    try:
        cfg = json.loads(SETTINGS.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return f"__SETTINGS_UNREADABLE__ {e}"
    cmds = []
    for blocks in (cfg.get("hooks") or {}).values():
        for b in blocks:
            for h in b.get("hooks", []):
                cmds.append(h.get("command", ""))
    return "\n".join(cmds)


def run(cmd: list[str], stdin: str = "", timeout: int = 15):
    """Run a hook; return (exit_code, stdout, stderr) or None on launch failure."""
    try:
        p = subprocess.run(
            cmd, input=stdin, capture_output=True, text=True, timeout=timeout
        )
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired):
        return None


# --- synthetic self-tests for the side-effect-free hooks -----------------------
def test_check_agents_tag():
    sh = HOOKS / "check_agents_tag.sh"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("some assistant text with no footer")
        notag = f.name
    try:
        r = run(["bash", str(sh)], json.dumps({"transcript_path": notag}))
        if r is None:
            return FAIL, "could not launch"
        code, out, _ = r
        return (PASS, "warns on missing footer") if "[HOOK]" in out else (FAIL, "no warning emitted")
    finally:
        os.unlink(notag)


def test_validate_note():
    sh = HOOKS / "validate-note.sh"
    d = tempfile.mkdtemp()
    note = Path(d) / "notes" / "bad.md"
    note.parent.mkdir(parents=True)
    note.write_text("---\ntype: finding\n---\nbody\n")
    r = run(["bash", str(sh)], json.dumps({"tool_input": {"file_path": str(note)}}))
    if r is None:
        return FAIL, "could not launch"
    _, out, _ = r
    return (PASS, "warns on bad frontmatter") if "WARN" in out else (FAIL, "no WARN emitted")


def test_corpus_canary_session():
    py = HOOKS / "corpus_canary.py"
    r = run(["python3", str(py)], json.dumps({"hook_event_name": "SessionStart"}))
    if r is None:
        return FAIL, "could not launch"
    _, out, _ = r
    return (PASS, "emits primer") if "CORPUS-INVARIANT" in out else (FAIL, "no primer")


def test_corpus_canary_edit():
    py = HOOKS / "corpus_canary.py"
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": "x.py", "content": "sample_rate = 300000"},
    }
    r = run(["python3", str(py)], json.dumps(payload))
    if r is None:
        return FAIL, "could not launch"
    _, out, _ = r
    return (PASS, "fires on canonical param") if "A/B/C/D" in out else (FAIL, "no edit warning")


def test_guard_destructive():
    py = HOOKS / "guard_destructive.py"
    base = {"hook_event_name": "PreToolUse", "tool_name": "Bash"}
    block = run(["python3", str(py)], json.dumps({**base, "tool_input": {"command": "git add -A"}}))
    allow = run(["python3", str(py)], json.dumps({**base, "tool_input": {"command": "git status"}}))
    if block is None or allow is None:
        return FAIL, "could not launch"
    if block[0] == 2 and allow[0] == 0:
        return PASS, "blocks bulk-git, allows safe"
    return FAIL, f"block exit={block[0]} (want 2), allow exit={allow[0]} (want 0)"


# Registry: name -> (settings substring to confirm wiring, script file,
#                    syntax kind, exec self-test fn or None)
REGISTRY = [
    ("corpus_canary (session)", "corpus_canary.py", "corpus_canary.py", "py", test_corpus_canary_session),
    ("corpus_canary (edit)", "corpus_canary.py", "corpus_canary.py", "py", test_corpus_canary_edit),
    ("guard_destructive", "guard_destructive.py", "guard_destructive.py", "py", test_guard_destructive),
    ("check_agents_tag", "check_agents_tag.sh", "check_agents_tag.sh", "sh", test_check_agents_tag),
    ("validate-note", "validate-note.sh", "validate-note.sh", "sh", test_validate_note),
    # side-effecting / heavy -> syntax-only, never executed by the beacon
    ("session-capture", "session-capture.sh", "session-capture.sh", "sh", None),
    ("session-orient", "session-orient.ps1", "session-orient.ps1", None, None),
    ("check_plan_mode", "check_plan_mode.ps1", "check_plan_mode.ps1", None, None),
]


def syntax_ok(kind: str, path: Path):
    if kind == "py":
        r = run([sys.executable, "-m", "py_compile", str(path)])
        return r is not None and r[0] == 0
    if kind == "sh":
        r = run(["bash", "-n", str(path)])
        return r is not None and r[0] == 0
    return None  # no checker (powershell) -> skip


def _genuine_invocations(transcript: str) -> set[str]:
    """Skills GENUINELY invoked in one transcript.

    Counts only structured invocation events — a `Skill` tool_use block, or a
    `<command-name>` slash marker — NOT bare name mentions. The session-orient
    banner and the skill-discovery reminder list every skill name in every
    transcript, so a substring match would mark everything 'used' (it did:
    reported 0 dormant when ~25 were). See ops/health/repo-audit-2026-05-28.html.
    """
    import re

    used: set[str] = set()
    for line in transcript.splitlines():
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        # (a) Skill tool_use blocks
        if obj.get("type") == "assistant":
            content = (obj.get("message") or {}).get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_use" and blk.get("name") == "Skill":
                        sk = (blk.get("input") or {}).get("skill") or (blk.get("input") or {}).get("command")
                        if sk:
                            used.add(str(sk).lstrip("/").strip())
        # (b) <command-name> slash markers in any text payload of the line
        for m in re.findall(r"<command-name>\s*/?([A-Za-z0-9_-]+)", line):
            used.add(m)
    return used


def scan_unused_skills(days: int = 30):
    cutoff = time.time() - days * 86400
    skills = sorted(p.name for p in SKILLS_DIR.glob("*/") if (p / "SKILL.md").exists())
    recent = [f for f in glob.glob(TRANSCRIPT_GLOB) if os.path.getmtime(f) >= cutoff]
    used: set[str] = set()
    for f in recent:
        try:
            used |= _genuine_invocations(Path(f).read_text(errors="ignore"))
        except OSError:
            continue
    return skills, sorted(set(skills) - used), len(recent)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="execute self-tests + scan skills")
    ap.add_argument("--quick", action="store_true", help="wiring + syntax only (default)")
    ap.add_argument("--quiet", action="store_true", help="print only on FAIL")
    args = ap.parse_args()
    full = args.full

    wired = load_wired_commands()
    lines, fails = [], 0

    for name, settings_sub, fname, kind, exec_fn in REGISTRY:
        path = HOOKS / fname
        # 1. wiring
        if settings_sub not in wired:
            lines.append(f"  {FAIL}  {name:24} not wired in settings")
            fails += 1
            continue
        # 2. existence
        if not path.exists():
            lines.append(f"  {FAIL}  {name:24} script file missing: {fname}")
            fails += 1
            continue
        # 3. syntax
        syn = syntax_ok(kind, path)
        if syn is False:
            lines.append(f"  {FAIL}  {name:24} syntax error in {fname}")
            fails += 1
            continue
        # 4. execution self-test (full mode, safe hooks only)
        if full and exec_fn is not None:
            status, detail = exec_fn()
            lines.append(f"  {status}  {name:24} {detail}")
            if status == FAIL:
                fails += 1
        else:
            why = "wired+syntax ok" if exec_fn or not full else "wired+syntax ok (not executed: side-effects)"
            lines.append(f"  {SKIP if full else PASS}  {name:24} {why}")

    report = [f"=== Hook Health ({'full' if full else 'quick'}) ===", *lines]

    if full:
        all_skills, unused, n = scan_unused_skills()
        report.append("")
        report.append(f"=== Skill activity (last 30d, {n} transcripts) ===")
        report.append(f"  {len(all_skills) - len(unused)}/{len(all_skills)} skills used; "
                      f"{len(unused)} dormant")
        if unused:
            report.append("  dormant: " + ", ".join(unused))

    report.append("")
    report.append(f"RESULT: {fails} failing" if fails else "RESULT: all hooks healthy")

    if fails or not args.quiet:
        print("\n".join(report))
    elif args.quiet and fails:
        print("\n".join(report))

    return 0  # advisory only


if __name__ == "__main__":
    sys.exit(main())
