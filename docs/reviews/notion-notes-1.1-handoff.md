# Handoff: Notion Notes Phase 1.1 — Project Scaffold & API Client

## What Was Built

Foundation package for the Notion Atomic Notes automation (Phase 1.1 from `NOTION_NOTES_ROADMAP.md`). Every future command (tag, atomize, link, process, move) depends on these client wrappers and config loading.

## Files Created (7 new)

| File | Purpose | Lines |
|------|---------|-------|
| `notion_notes/__init__.py` | Package init, version string | 3 |
| `notion_notes/__main__.py` | `python -m notion_notes` entry point | 3 |
| `notion_notes/config.py` | `NotionNotesConfig` frozen dataclass + `load_config()` with manual .env parsing | ~100 |
| `notion_notes/claude_client.py` | `ClaudeClient` with `prompt()` and `prompt_json()` methods | ~100 |
| `notion_notes/notion_client.py` | `NotionPage` dataclass + `NotionClientWrapper` (read/write/helpers/rate limiting) | ~275 |
| `notion_notes/cli.py` | Click CLI group with `--dry-run`, `--env-file`, `--verbose`; `list-pages` command | ~55 |
| `tests/test_notion_client.py` | 41 tests with mocks covering all 10 plan test points + extras | ~300 |

## Files Modified (3)

| File | Change |
|------|--------|
| `.gitignore` | Added `.env`, `.cache/`, `taxonomy.json` |
| `requirements.txt` | Added `notion-client`, `anthropic` |
| `.env.example` | New template with required env vars |

## Key Design Decisions

1. **Manual .env parsing** — 15-line parser in `config.py`, avoids python-dotenv dependency. Uses `os.environ.setdefault()` so real env vars take precedence.
2. **Synchronous rate limiting** — `time.sleep()` based, matching the sync-only codebase. Exponential backoff on 429 responses (1s, 2s, 4s).
3. **8 block types in `blocks_to_markdown`** — paragraph, heading_1/2/3, bulleted_list_item, numbered_list_item, code, divider. Unknown types produce `[unsupported: <type>]`.
4. **`extract_title` scans by type** — finds the property with `type: "title"` rather than assuming key name "Name".
5. **Dry-run on all write ops** — `create_page`, `update_page_properties`, `append_blocks`, `add_relation` all check `self.dry_run` and log instead of calling the API.

## Validation Results

- py_compile: All 6 new .py files pass
- Tests: 41 new tests pass (273 total, 0 failures, 0 regressions)
- CLI: `python -m notion_notes --help` shows correct output with `list-pages` command

## Known Limitations / Future Work

- `list-pages` is the only command — Phase 2+ will add `tag`, `atomize`, `link`, `process`, `move`
- No `kb_database_id` / `notes_database_id` in config yet — will be added when commands need them
- `blocks_to_markdown` doesn't handle nested blocks (e.g., toggle lists, child pages)
- No retry on non-429 transient errors (e.g., 502, 503)

## How to Verify

```powershell
.\.venv\Scripts\python.exe -m py_compile notion_notes/config.py
.\.venv\Scripts\python.exe -m pytest tests/test_notion_client.py -v
.\.venv\Scripts\python.exe -m notion_notes --help
```
