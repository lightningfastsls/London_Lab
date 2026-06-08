#!/usr/bin/env python3
"""Vault pulse — local maintenance-staleness beacon.

The repo's KG-maintenance skills (/health, /reflect, /reweave, /stats) went
unrun for ~47 days because the only nudge was a passive line in ops/reminders.md
that nobody acted on. This is the active replacement: at SessionStart, if the
vault has gone stale (no ops/health/ activity in N days), print a LOUD, specific
worklist so the debt is impossible to ignore.

Deliberately conservative: it surfaces *pressure signals* it can compute
reliably (inbox backlog, orphan count, dormant-skill count, days overdue) and
then defers to `/health full` for the authoritative link/schema audit. It does
NOT reimplement dangling-link detection — many [[slug]] links are topic-map MOCs,
not notes, and a naive checker would cry wolf (the false-confidence failure mode
this whole effort is fighting).

Modes:
  --if-stale N : print ONLY if >N days since last ops/health/ activity, else exit
                 silent. Cheap staleness check happens BEFORE the note-graph parse.
  --mark-done  : touch ops/health/.last-maintenance to reset the clock for a week.
  (default)    : always print the pulse.

Always exits 0 (advisory). Note: 'age' uses filesystem mtime (a fresh git clone
resets mtimes, which makes this UNDER-report — safe: it never false-alarms on age).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR", str(Path(__file__).resolve().parents[2])))
NOTES = ROOT / "notes"
INBOX = ROOT / "inbox"
HEALTH = ROOT / "ops" / "health"
SKILLS_DIR = ROOT / ".claude" / "skills"
MARKER = HEALTH / ".last-maintenance"
WIKILINK = re.compile(r"\[\[([^\]|]+)")

# arscontexta maintenance-trigger thresholds (CLAUDE.md "Maintenance Triggers")
INBOX_THRESHOLD = 3
ORPHAN_AGE_DAYS = 7
STALE_AGE_DAYS = 30
STALE_MAX_INCOMING = 2


def days_since_last_maintenance() -> float | None:
    """Days since the newest ops/health/ activity (or the explicit marker)."""
    candidates = []
    if MARKER.exists():
        candidates.append(MARKER.stat().st_mtime)
    for pat in ("*.md", "*.html"):
        for f in HEALTH.glob(pat):
            candidates.append(f.stat().st_mtime)
    if not candidates:
        return None  # no history -> treat as "unknown", caller decides
    return (time.time() - max(candidates)) / 86400.0


def build_link_graph():
    """Return (titles, incoming_count) over notes/*.md."""
    files = list(NOTES.glob("*.md"))
    titles = {f.stem: f for f in files}
    incoming = {stem: 0 for stem in titles}
    for f in files:
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        seen = set()
        for tgt in WIKILINK.findall(text):
            tgt = tgt.strip()
            if tgt in titles and tgt != f.stem:
                seen.add(tgt)
        for tgt in seen:
            incoming[tgt] += 1
    return titles, incoming


def vault_signals():
    titles, incoming = build_link_graph()
    now = time.time()
    orphans, stale = [], []
    for stem, f in titles.items():
        inc = incoming[stem]
        age_days = (now - f.stat().st_mtime) / 86400.0
        if inc == 0 and age_days > ORPHAN_AGE_DAYS:
            orphans.append(stem)
        if age_days > STALE_AGE_DAYS and inc < STALE_MAX_INCOMING:
            stale.append(stem)
    inbox_n = len(list(INBOX.glob("*.md"))) if INBOX.exists() else 0
    return len(titles), orphans, stale, inbox_n


def dormant_skill_count(days: int = 30) -> int:
    """Reuse the genuine-invocation scan from hook_health (no duplication)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import hook_health  # noqa: E402
        _, unused, _ = hook_health.scan_unused_skills(days)
        return len(unused)
    except Exception:
        return -1  # unknown


def render(stale_days: float | None) -> str:
    n_notes, orphans, stale, inbox_n = vault_signals()
    dormant = dormant_skill_count()
    overdue = f"{stale_days:.0f} days" if stale_days is not None else "unknown"

    lines = [f"⚠ VAULT MAINTENANCE — {overdue} since last ops/health/ activity ({n_notes} notes)"]
    lines.append(f"  inbox:    {inbox_n} item(s)" + (f"  → ≥{INBOX_THRESHOLD}, run /reduce or /pipeline" if inbox_n >= INBOX_THRESHOLD else "  (ok)"))
    if orphans:
        shown = ", ".join(sorted(orphans)[:5])
        more = f" (+{len(orphans) - 5} more)" if len(orphans) > 5 else ""
        lines.append(f"  orphans:  {len(orphans)} note(s) with no incoming links  → /reflect")
        lines.append(f"            e.g. {shown}{more}")
    else:
        lines.append("  orphans:  0  (ok)")
    lines.append(f"  stale:    {len(stale)} note(s) >{STALE_AGE_DAYS}d old with <{STALE_MAX_INCOMING} incoming links" + ("  → /reweave" if stale else "  (ok)"))
    if dormant >= 0:
        lines.append(f"  skills:   {dormant} maintenance skill(s) dormant >30d")
    lines.append("  → Authoritative audit: /health full   |   then /reflect, /reweave, /reduce as flagged")
    lines.append("  → Silence for a week: python3 ops/scripts/vault_pulse.py --mark-done")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-stale", type=float, metavar="DAYS",
                    help="print only if >DAYS since last ops/health/ activity")
    ap.add_argument("--mark-done", action="store_true", help="reset the staleness clock")
    args = ap.parse_args()

    if args.mark_done:
        HEALTH.mkdir(parents=True, exist_ok=True)
        MARKER.touch()
        print(f"vault_pulse: marked maintenance done ({MARKER})")
        return 0

    stale_days = days_since_last_maintenance()

    if args.if_stale is not None:
        # cheap gate FIRST — skip the note-graph parse when the vault is fresh
        if stale_days is not None and stale_days <= args.if_stale:
            return 0
    print(render(stale_days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
