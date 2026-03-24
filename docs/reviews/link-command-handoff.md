# Handoff: Link Command (Phase 4.1)

## What Was Built

The `link` command — discovers semantic connections between KB pages using Claude Sonnet. Uses two-stage filtering: fast pre-filtering by tag/domain overlap (in-memory from already-fetched pages), then Claude for semantic comparison of plausible candidates. A JSON cache prevents O(n^2) re-evaluation.

## Files Created (2 new)

| File | Purpose | Lines |
|------|---------|-------|
| `notion_notes/commands/link.py` | `Linker` class, `LinkConfig`, `Connection`, `LinkResult`, prompt templates, candidate filtering, JSON cache | ~310 |
| `tests/test_linker.py` | 17 tests across 12 test classes, all mocked | ~400 |

## Files Modified (3)

| File | Change |
|------|--------|
| `notion_notes/cli.py` | Added `link` subcommand with `--page`, `--all-unlinked`, `--recent`; imported `LinkConfig`, `Linker` |
| `NOTION_NOTES_ROADMAP.md` | Updated Phase 1-3 gate checkboxes to `[x]`, Phase 4.1 status to DONE |
| `IMPLEMENTATION_PROGRESS.md` | Added Phase 4.1 completion entry |

## Key Design Decisions

1. **In-memory candidate filtering** — All pages are fetched once via `query_database()`. Candidate filtering happens in-memory (tag set intersection, domain comparison), avoiding N additional API calls per target page.
2. **Two inclusion criteria** — A candidate is included if `tag_overlap >= min_tag_overlap` (default 1) OR same non-empty domain. This catches both topically related notes and domain neighbors.
3. **Canonical pair keys** — `_pair_key(id_a, id_b)` returns `tuple(sorted([id_a, id_b]))` so A-B == B-A. Prevents duplicate evaluation regardless of direction.
4. **JSON cache format** — `[[id_a, id_b], ...]` sorted list of pairs. Loaded as `set[tuple[str, str]]` for O(1) lookup. Saved after all pages processed (not per page).
5. **Bidirectional relations** — Both `add_relation(A, "Related Notes", [B])` and `add_relation(B, "Related Notes", [A])` are called. Notion's self-referencing relations don't auto-sync both directions.
6. **No Claude call for empty candidates** — If no candidates pass the pre-filter (all excluded by cache or no overlap), the Claude API is not called.
7. **Content truncation** — Target page content truncated to 6000 chars for the Sonnet prompt.

## Architecture

```
CLI (cli.py link command)
  -> loads config, builds NotionClientWrapper + ClaudeClient
  -> fetches all_pages from KB database
  -> determines target pages: --page | --all-unlinked | --recent N
  -> constructs Linker(notion, claude, link_config, dry_run)
  -> calls linker.link_pages(pages, all_pages) -> list[LinkResult]
  -> prints results

Linker._link_one_page(page, all_pages, cache):
  1. _find_candidates() — filter by tag overlap / domain match, exclude self + cached
  2. If no candidates -> return early (no Claude call)
  3. Fetch page blocks, convert to markdown
  4. Build prompt with candidate titles + metadata
  5. Call claude.prompt() -> raw text
  6. _parse_connections() strips fences, validates JSON, maps titles to IDs
  7. Create bidirectional relations (unless dry_run)
  8. Update Status -> "Linked" (unless dry_run)
  9. Add all evaluated pairs to cache
  10. Return LinkResult
```

## Test Coverage

17 tests, all mocked (no live API):

| # | Test Class | What It Verifies |
|---|-----------|-----------------|
| 1 | TestCandidateFiltering | Pages with shared tags found as candidates |
| 2-3 | TestCandidateExclusions | Self excluded, cached pairs excluded |
| 4-5 | TestPairKey | Sorted canonical keys, symmetry |
| 6-8 | TestParseConnections | Valid parsing, markdown fence stripping, unknown titles skipped |
| 9 | TestBidirectionalRelations | add_relation called both directions |
| 10-11 | TestCacheUpdate | Cache saved after evaluation, cached pairs not re-evaluated |
| 12 | TestTrivialConnections | Empty connections from Claude handled |
| 13 | TestMaxCandidates | Candidate list capped at max_candidates |
| 14 | TestDryRun | No relations or status updates in dry-run |
| 15 | TestEmptyCandidates | No Claude call when no candidates match |
| 16 | TestIncrementalLinking | Already-linked pages can be re-linked |
| 17 | TestDomainOnlyMatch | Same domain with no tag overlap still matches |

## Validation Results

- `py_compile`: All touched files pass
- `pytest tests/test_linker.py`: 17 passed
- `pytest tests/`: 375 passed, 2 failed (pre-existing failures in test_notion_client.py, unrelated)

## Pre-existing Test Failures (Not Introduced)

1. `TestNotionClientInit::test_init_creates_client` — Test expects `Client(auth=...)` but code passes `notion_version='2022-06-28'` too
2. `TestConfigLoad::test_env_vars_take_precedence_over_file` — Test comment says `setdefault` but `_parse_env_file` uses `os.environ[key] = value` which overwrites

## CLI Usage

```bash
python -m notion_notes link --page PAGE_ID           # link one page
python -m notion_notes link --all-unlinked            # link all non-linked pages
python -m notion_notes link --recent 10               # link 10 most recent pages
python -m notion_notes link --all-unlinked --dry-run  # preview connections
```

## Dependencies on Prior Work

- `NotionClientWrapper` — `get_page_blocks()`, `blocks_to_markdown()`, `query_database()`, `add_relation()`, `update_page_properties()`, `extract_property()`
- `ClaudeClient` — `prompt()` method
- `load_config()` — loads `KB_DATABASE_ID` from env

## What's Next

- Phase 5.1: `process` command (orchestrate tag -> atomize -> link) and `move` utility (inbox -> KB)
