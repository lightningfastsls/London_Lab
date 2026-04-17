#!/usr/bin/env python3
"""Migrate 13 drifted confidence values in notes/ to canonical schema.

Migrations:
  confidence: reasoned  -> likely                 (9 notes)
  confidence: uncertain -> DELETE FIELD           (2 notes, both type: open-question)
  confidence: medium    -> likely                 (1 note, typo)
  confidence: confirmed -> proven                 (1 note, typo)

Preserves per-file line endings (CRLF notes stay CRLF; LF stays LF). Line-ending
normalization is deferred to a separate dedicated commit.

Usage:
  python scripts/migrate_schema_drift_2026_04_17.py           # dry-run
  python scripts/migrate_schema_drift_2026_04_17.py --apply   # write changes
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent / "notes"
APPLY = "--apply" in sys.argv

CONFIDENCE_MIGRATIONS = {
    "reasoned":  "likely",
    "medium":    "likely",
    "confirmed": "proven",
}

TYPE_RE       = re.compile(r"^type:\s*(\S+)\s*$", re.MULTILINE)
CONFIDENCE_RE = re.compile(r"^(confidence:\s*)(\S+)\s*$", re.MULTILINE)

def split_frontmatter(text: str, nl: str):
    opener = f"---{nl}"
    closer = f"{nl}---{nl}"
    if not text.startswith(opener):
        return None, None, None
    end = text.find(closer, len(opener))
    if end == -1:
        return None, None, None
    fm = text[len(opener):end]
    body = text[end + len(closer):]
    return fm, body, opener

def migrate_frontmatter(fm: str, nl: str) -> tuple[str, list[str]]:
    changes = []
    t_match = TYPE_RE.search(fm)
    note_type = t_match.group(1) if t_match else None

    lines = fm.split(nl)
    out_lines = []
    for line in lines:
        m = CONFIDENCE_RE.match(line)
        if not m:
            out_lines.append(line)
            continue
        val = m.group(2)
        if val in CONFIDENCE_MIGRATIONS:
            new_val = CONFIDENCE_MIGRATIONS[val]
            out_lines.append(f"{m.group(1)}{new_val}")
            changes.append(f"confidence: {val} -> {new_val}")
        elif val == "uncertain":
            if note_type == "open-question":
                changes.append("confidence: uncertain -> DELETED (type: open-question)")
                continue
            else:
                out_lines.append(f"{m.group(1)}speculative")
                changes.append(f"confidence: uncertain -> speculative (type: {note_type})")
        else:
            out_lines.append(line)
    return nl.join(out_lines), changes

def main():
    changed = 0
    for path in sorted(ROOT.glob("*.md")):
        raw_bytes = path.read_bytes()
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue
        nl = "\r\n" if "\r\n" in raw[:2000] else "\n"
        fm, body, opener = split_frontmatter(raw, nl)
        if fm is None:
            continue
        new_fm, changes = migrate_frontmatter(fm, nl)
        if not changes:
            continue
        new_text = f"{opener}{new_fm}{nl}---{nl}{body}"
        if new_text == raw:
            continue
        changed += 1
        print(f"[{'APPLY' if APPLY else 'DRY'}] {path.name}")
        for c in changes:
            print(f"    {c}")
        if APPLY:
            path.write_bytes(new_text.encode("utf-8"))

    verb = "updated" if APPLY else "would change"
    print(f"\n{changed} notes {verb}")

if __name__ == "__main__":
    main()
