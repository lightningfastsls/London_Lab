"""Upload PROJECT_STATE_REPORT.md to Notion Knowledge Base.

Handles:
- Markdown → Notion block conversion (headings, lists, code, paragraphs)
- Table → Notion table block conversion
- 100-block batch limit per API call
- 2000-char rich text limit
- Projects relation to 'Miki London lab'
"""

import re
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notion_notes.config import load_config
from notion_notes.notion_client import NotionClientWrapper


MIKI_LONDON_LAB_PAGE_ID = "2ad2bc59-9f6e-80f1-8ee0-e85e0ef3d8a8"
BATCH_SIZE = 100  # Notion API limit per request
MAX_TEXT_LEN = 2000  # Notion rich text limit


def _rich_text(content: str) -> list[dict]:
    """Create rich_text array, splitting if content exceeds 2000 chars."""
    chunks = []
    while content:
        chunk = content[:MAX_TEXT_LEN]
        chunks.append({"type": "text", "text": {"content": chunk}})
        content = content[MAX_TEXT_LEN:]
    return chunks


def _text_block(block_type: str, text: str) -> dict:
    """Create a text-based block (paragraph, heading, list item)."""
    return {
        "type": block_type,
        block_type: {"rich_text": _rich_text(text)},
    }


def _parse_table(lines: list[str]) -> dict:
    """Convert markdown table lines to a Notion table block."""
    # Parse header and data rows (skip separator row with dashes)
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Skip separator rows like |---|---|
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if cells and all(re.match(r'^[-:]+$', c) for c in cells):
            continue
        if cells:
            rows.append(cells)

    if not rows:
        return None

    n_cols = max(len(r) for r in rows)
    # Pad rows to equal column count
    for r in rows:
        while len(r) < n_cols:
            r.append("")

    table_rows = []
    for row in rows:
        table_rows.append({
            "type": "table_row",
            "table_row": {
                "cells": [[{"type": "text", "text": {"content": cell}}] for cell in row]
            }
        })

    return {
        "type": "table",
        "table": {
            "table_width": n_cols,
            "has_column_header": True,
            "has_row_header": False,
            "children": table_rows,
        }
    }


def markdown_to_notion_blocks(markdown: str) -> list[dict]:
    """Convert markdown to Notion blocks with table support."""
    blocks = []
    lines = markdown.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Fenced code block
        if line.startswith("```"):
            language = line[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # skip closing fence
            code_text = "\n".join(code_lines)
            # Split long code blocks
            if len(code_text) > MAX_TEXT_LEN:
                chunks = [code_text[j:j+MAX_TEXT_LEN] for j in range(0, len(code_text), MAX_TEXT_LEN)]
                for chunk in chunks:
                    blocks.append({
                        "type": "code",
                        "code": {
                            "rich_text": [{"type": "text", "text": {"content": chunk}}],
                            "language": language or "plain text",
                        }
                    })
            else:
                blocks.append({
                    "type": "code",
                    "code": {
                        "rich_text": [{"type": "text", "text": {"content": code_text}}],
                        "language": language or "plain text",
                    }
                })
            continue

        # Table detection: line starts with | and next line is separator
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r'^\s*\|[-\s|:]+\|', lines[i + 1]):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            table_block = _parse_table(table_lines)
            if table_block:
                blocks.append(table_block)
            continue

        # Divider
        if line.strip() == "---":
            blocks.append({"type": "divider", "divider": {}})
            i += 1
            continue

        # Skip empty lines
        if not line.strip():
            i += 1
            continue

        # Headings
        if line.startswith("### "):
            blocks.append(_text_block("heading_3", line[4:].strip()))
        elif line.startswith("## "):
            blocks.append(_text_block("heading_2", line[3:].strip()))
        elif line.startswith("# "):
            blocks.append(_text_block("heading_1", line[2:].strip()))
        # Bulleted list
        elif line.startswith("- "):
            blocks.append(_text_block("bulleted_list_item", line[2:].strip()))
        # Numbered list
        elif re.match(r'^\d+\.\s', line):
            text = line.split(". ", 1)[1] if ". " in line else line
            blocks.append(_text_block("numbered_list_item", text.strip()))
        # Bold lines (often used as labels)
        elif line.startswith("**") and line.endswith("**"):
            # Render as bold paragraph
            content = line.strip("* ")
            blocks.append({
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": content},
                        "annotations": {"bold": True},
                    }]
                }
            })
        # Blockquote
        elif line.startswith("> "):
            blocks.append({
                "type": "callout",
                "callout": {
                    "rich_text": _rich_text(line[2:].strip()),
                    "icon": {"type": "emoji", "emoji": "💡"},
                }
            })
        # Paragraph (everything else)
        else:
            text = line.strip()
            if text:
                blocks.append(_text_block("paragraph", text))

        i += 1

    return blocks


def upload_report(dry_run: bool = False):
    """Upload the project state report to Notion KB."""
    config = load_config(env_path=Path(".env"), dry_run=dry_run)
    client = NotionClientWrapper(token=config.notion_token, dry_run=dry_run)

    if not config.kb_database_id:
        print("ERROR: NOTION_KB_DATABASE_ID not set in .env")
        sys.exit(1)

    # Read the report
    report_path = Path("docs/PROJECT_STATE_REPORT.md")
    if not report_path.exists():
        print(f"ERROR: {report_path} not found")
        sys.exit(1)

    markdown = report_path.read_text(encoding="utf-8")
    print(f"Read report: {len(markdown)} chars")

    # Convert to Notion blocks
    blocks = markdown_to_notion_blocks(markdown)
    print(f"Converted to {len(blocks)} Notion blocks")

    # Create page with first batch of blocks
    first_batch = blocks[:BATCH_SIZE]
    remaining = blocks[BATCH_SIZE:]

    properties = {
        "Title": {"title": [{"text": {"content": "Mickey London Lab \u2014 Project State Report (2026-02-20)"}}]},
        "Status": {"select": {"name": "Raw"}},
        "Domain": {"select": {"name": "Project Management"}},
        "Tags": {"multi_select": [
            {"name": "project-report"},
            {"name": "USV"},
            {"name": "arscontexta"},
        ]},
        "Is Atomic": {"checkbox": False},
        "Projects": {"relation": [{"id": MIKI_LONDON_LAB_PAGE_ID}]},
    }

    print(f"Creating page with {len(first_batch)} blocks...")
    page_id = client.create_page(
        database_id=config.kb_database_id,
        properties=properties,
        children=first_batch,
    )

    if page_id is None and not dry_run:
        print("ERROR: Failed to create page")
        sys.exit(1)

    if dry_run:
        print(f"[DRY RUN] Would create page + {len(remaining)} more blocks in {len(remaining) // BATCH_SIZE + 1} batches")
        return

    print(f"Created page: {page_id}")

    # Append remaining blocks in batches
    batch_num = 1
    while remaining:
        batch = remaining[:BATCH_SIZE]
        remaining = remaining[BATCH_SIZE:]
        print(f"Appending batch {batch_num} ({len(batch)} blocks)...")
        time.sleep(0.5)  # Rate limiting courtesy
        client.append_blocks(page_id, batch)
        batch_num += 1

    print(f"\nDone! Page created with {len(blocks)} total blocks.")
    print(f"Page ID: {page_id}")
    print(f"Notion URL: https://notion.so/{page_id.replace('-', '')}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    upload_report(dry_run=dry_run)
