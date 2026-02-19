# Handoff: Tag Command (Step 2)

## What Was Built

The `tag` command — classifies KB pages by domain and tags using Claude Haiku. Reads page content from Notion, prompts Claude to assign a domain + tags from a growing taxonomy, updates Notion page properties, and appends new tags to the taxonomy file.

## Files Created (4 new)

| File | Purpose | Lines |
|------|---------|-------|
| `notion_notes/commands/__init__.py` | Commands subpackage init | 1 |
| `notion_notes/commands/tag.py` | `Tagger` class, `TagConfig`, `TagResult`, prompt templates, JSON extraction, taxonomy I/O | ~280 |
| `taxonomy.json` | Seed taxonomy with 6 domains, 5 tags each | ~45 |
| `tests/test_tagger.py` | 19 tests across 12 test classes, all mocked | ~250 |

## Files Modified (2)

| File | Change |
|------|--------|
| `notion_notes/config.py` | Added `kb_database_id` field (from `KB_DATABASE_ID` env var) |
| `notion_notes/cli.py` | Added `tag` subcommand with `--page`, `--untagged`, `--retag-all`, `--taxonomy`; added `_load_cfg_and_clients` helper |

## Key Design Decisions

1. **Uses `prompt()` not `prompt_json()`** — Claude Haiku sometimes wraps JSON in markdown fences. Custom `_extract_json()` strips fences before parsing; `prompt_json()` in ClaudeClient doesn't handle this.
2. **Single retry on bad JSON** — If first response isn't valid JSON, sends a follow-up message asking for "ONLY valid JSON". Max 1 retry (configurable via `TagConfig.max_retries`).
3. **Taxonomy grows automatically** — New tags suggested by Claude are appended under the assigned domain. Saved once per batch (not per page) to reduce I/O.
4. **Seed taxonomy** — Both embedded in `_seed_taxonomy()` function and in repo-root `taxonomy.json`. The function handles the case where the file is missing.
5. **Content truncation** — Note content truncated to 3000 chars for the Haiku prompt to stay well within context limits.
6. **`kb_database_id` in shared config** — All KB-targeting commands (tag, atomize, link) will use this field. Loaded from `KB_DATABASE_ID` env var.

## Architecture

```
CLI (cli.py tag command)
  -> loads config, builds NotionClientWrapper + ClaudeClient
  -> constructs Tagger(notion, claude, tag_config, dry_run)
  -> determines pages: --page | --untagged | --retag-all
  -> calls tagger.tag_pages(pages) -> list[TagResult]
  -> prints results

Tagger._tag_one_page(page, taxonomy):
  1. Fetch blocks via notion.get_page_blocks()
  2. Convert to markdown via blocks_to_markdown()
  3. Build prompt with taxonomy
  4. Call claude.prompt() -> raw text
  5. _extract_json() strips fences, validates keys
  6. Validate domain against taxonomy
  7. Enforce max_tags limit
  8. Update Notion page (unless dry_run)
  9. Return TagResult
```

## Test Coverage

19 tests, all mocked (no live API):

| # | Test Class | What It Verifies |
|---|-----------|-----------------|
| 1-2 | TestBasicTagging | Valid response parsing, Notion property updates |
| 3 | TestExistingTags | Existing tags not marked as new |
| 4-5 | TestTaxonomyGrowth | New tags appended to file and in result |
| 6 | TestDomainValidation | Invalid domain records error |
| 7 | TestTagLimit | Excess tags trimmed to max_tags |
| 8 | TestPropertyFormat | Domain as select, Tags as multi_select |
| 9 | TestUntaggedFilter | Filter shape verification |
| 10 | TestDryRun | No Notion writes in dry-run |
| 11 | TestTaxonomySeed | Creates seed file if missing |
| 12-15 | TestMarkdownFenceStripping | json fences, plain fences, no fences, new_tags default |
| 16-17 | TestRetryOnBadJson | Retry succeeds, retry exhausted records error |
| 18-19 | TestEmptyContent | Empty blocks skipped, whitespace-only skipped |

## Validation Results

- `py_compile`: All 5 touched files pass
- `pytest tests/test_tagger.py`: 19 passed
- `pytest tests/`: 325 passed, 2 failed (pre-existing failures in test_notion_client.py, unrelated to this change)

## Pre-existing Test Failures (Not Introduced)

1. `TestNotionClientInit::test_init_creates_client` — Test expects `Client(auth=...)` but code now passes `notion_version='2022-06-28'` too
2. `TestConfigLoad::test_env_vars_take_precedence_over_file` — Test comment says `setdefault` but `_parse_env_file` uses `os.environ[key] = value` which overwrites

## CLI Usage

```bash
python -m notion_notes tag --page PAGE_ID           # tag one page
python -m notion_notes tag --untagged                # tag pages with empty Tags
python -m notion_notes tag --retag-all               # re-tag everything
python -m notion_notes tag --untagged --dry-run      # preview without writing
python -m notion_notes tag --taxonomy custom.json    # use different taxonomy file
```

## Dependencies on Prior Work

- `NotionClientWrapper` — `get_page_blocks()`, `blocks_to_markdown()`, `query_database()`, `update_page_properties()`
- `ClaudeClient` — `prompt()` method
- `load_config()` — loads `KB_DATABASE_ID` from env

## What's Next

- Step 3: `atomize` command (split long notes into atomic claims)
- Step 4: `link` command (find and create relations between notes using tags as fast-filter)
