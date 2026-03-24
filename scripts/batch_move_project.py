"""Batch move notes linked to 'Miki London lab' project to KB.

Skips notes already in KB (by title match).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notion_notes.commands.move import Mover
from notion_notes.config import load_config
from notion_notes.notion_client import NotionClientWrapper

PROJECTS_DB_ID = "305327422e0e4e8eb80be6f73d226712"
NOTES_DB_ID = "bee86f7bb4424fd787ab87b3932c3686"
KB_DB_ID = "30b2bc599f6e8032b337fbda2c975dda"
PROJECT_NAME = "Miki London lab"


def safe_print(msg: str) -> None:
    """Print with fallback for Windows console encoding issues."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def main() -> None:
    cfg = load_config(env_path=Path(".env"))
    notion = NotionClientWrapper(
        token=cfg.notion_token,
        rate_limit_rps=cfg.notion_rate_limit_rps,
    )

    # Find the project page ID
    projects = notion.query_database(PROJECTS_DB_ID)
    miki_project = None
    for p in projects:
        if p.title.strip().lower() == PROJECT_NAME.lower():
            miki_project = p
            break

    if not miki_project:
        print(f"ERROR: Could not find project '{PROJECT_NAME}'")
        return

    print(f"Project: {miki_project.title} ({miki_project.id})")

    # Query notes linked to this project
    notes = notion.query_database(
        NOTES_DB_ID,
        filter={
            "property": "\U0001f52c Projects",
            "relation": {"contains": miki_project.id},
        },
    )
    print(f"Found {len(notes)} notes linked to project")

    # Get KB titles for dedup
    kb_pages = notion.query_database(KB_DB_ID)
    kb_titles = {p.title.strip().lower() for p in kb_pages}

    # Filter out duplicates
    to_move = [n for n in notes if n.title.strip().lower() not in kb_titles]
    skipped = [n for n in notes if n.title.strip().lower() in kb_titles]

    if skipped:
        for n in skipped:
            safe_print(f"  [SKIP] {n.title} (already in KB)")

    if not to_move:
        print("Nothing to move!")
        return

    print(f"\nMoving {len(to_move)} notes...")

    # Move each note
    mover = Mover(notion=notion, target_db_id=KB_DB_ID)

    successes = 0
    failures = 0
    for note in to_move:
        result = mover.move_page(note.id)
        if result.error:
            safe_print(f"  [ERROR] {note.title}: {result.error}")
            failures += 1
        else:
            safe_print(f"  [OK]    {note.title} -> {result.new_page_id}")
            successes += 1

    safe_print(f"\nDone: {successes} moved, {failures} errors, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
