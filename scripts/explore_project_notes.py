"""Explore notes linked to the 'Miki London lab' project.

Queries the Notes DB for pages with the project relation,
then cross-references against KB to identify duplicates.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent to path so we can import notion_notes
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notion_notes.config import load_config
from notion_notes.notion_client import NotionClientWrapper

PROJECTS_DB_ID = "305327422e0e4e8eb80be6f73d226712"
NOTES_DB_ID = "bee86f7bb4424fd787ab87b3932c3686"
KB_DB_ID = "30b2bc599f6e8032b337fbda2c975dda"


def main() -> None:
    cfg = load_config(env_path=Path(".env"))
    notion = NotionClientWrapper(
        token=cfg.notion_token,
        rate_limit_rps=cfg.notion_rate_limit_rps,
    )

    # Step 1: Find the "Miki London lab" project page ID
    print("=== Querying Projects DB ===")
    projects = notion.query_database(PROJECTS_DB_ID)
    print(f"Found {len(projects)} projects total")

    miki_project = None
    for p in projects:
        print(f"  - {p.title!r} ({p.id})")
        if "miki london" in p.title.lower() or "london lab" in p.title.lower():
            miki_project = p

    if not miki_project:
        print("\nERROR: Could not find 'Miki London lab' project!")
        return

    print(f"\nFound project: {miki_project.title!r} ({miki_project.id})")
    project_id = miki_project.id

    # Step 2: Query Notes DB for pages with this project relation
    print("\n=== Querying Notes DB (filtered by project) ===")
    # Notion relation filter uses the related page ID
    notes = notion.query_database(
        NOTES_DB_ID,
        filter={
            "property": "\U0001f52c Projects",  # 🔬 Projects
            "relation": {"contains": project_id},
        },
    )
    print(f"Found {len(notes)} notes linked to '{miki_project.title}'")

    # Step 3: Get all KB page titles for dedup check
    print("\n=== Querying KB for dedup check ===")
    kb_pages = notion.query_database(KB_DB_ID)
    kb_titles = {p.title.strip().lower() for p in kb_pages}
    print(f"KB has {len(kb_pages)} pages total")

    # Step 4: Categorize notes
    already_in_kb = []
    to_move = []

    for note in notes:
        title_lower = note.title.strip().lower()
        if title_lower in kb_titles:
            already_in_kb.append(note)
        else:
            to_move.append(note)

    print(f"\n=== Results ===")
    print(f"Total project notes: {len(notes)}")
    print(f"Already in KB (by title match): {len(already_in_kb)}")
    print(f"New notes to move: {len(to_move)}")

    if already_in_kb:
        print(f"\n--- Already in KB ---")
        for n in already_in_kb:
            print(f"  [SKIP] {n.title}")

    if to_move:
        print(f"\n--- To Move ---")
        for n in to_move:
            # Check for tags and other properties
            tags = NotionClientWrapper.extract_property(n.properties, "Tags")
            print(f"  [MOVE] {n.title}  (tags: {tags or 'none'})")

    # Also show the property names available on notes for reference
    if notes:
        print(f"\n--- Notes DB Property Names ---")
        for key in sorted(notes[0].properties.keys()):
            prop_type = notes[0].properties[key].get("type", "?")
            print(f"  {key!r}: {prop_type}")


if __name__ == "__main__":
    main()
