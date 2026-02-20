# Notion Atomic Notes Automation — Implementation Roadmap

> This file is the master plan for building the Notion Atomic Notes CLI tool.
> It lives in the project root alongside CLAUDE.md.
> Claude Code: read this file when asked "what's next", "check the roadmap", or "what should I build".
> Human: use the `/implement` commands below by copy-pasting them into Claude Code sessions.

---

## How to Use This File

1. Work through modules **in order** within each phase (dependencies are noted)
2. Each module has:
   - **What**: brief description of what to build
   - **`/implement` command**: copy-paste into Claude Code (or type `/implement <module description>`)
   - **Test plan**: how Claude Code should verify the module works
   - **Exit criteria**: what "done" looks like
3. After each module: commit, run tests, fix issues, commit again
4. Phase gates must pass before starting the next phase

## Status Key

- **DONE** — Implemented and tested
- **READY** — Dependencies met, can start
- **BLOCKED** — Waiting on dependency or external input
- **FUTURE** — Not yet prioritized

---

## Project Overview

A CLI tool (`notion_notes.py`) that you run manually to process Notion notes. Three core commands:

1. **Atomize** — Split raw class notes into atomic concept pages (non-destructive)
2. **Link** — Discover semantic connections across courses/topics
3. **Tag** — Apply consistent categorization using a growing taxonomy

Plus utilities: `process` (run all three), `move` (inbox to Knowledge Base), `migrate` (one-time Notes DB to KB transfer).

### Architecture

```
You take notes in class -> Notes DB (inbox) or directly into Knowledge Base
        |
        v
Optionally: python notion_notes.py move --page PAGE_ID  (inbox -> KB)
        |
        v
When ready: python notion_notes.py process [command]
        |
        v
Script reads pages from Knowledge Base ONLY via Notion API
        |
        v
Claude API analyzes content (finds concepts, connections, tags)
        |
        v
Script writes back to Knowledge Base via API (new pages, relations, tags)

Note: The automation never sees the Notes database (privacy boundary).
```

### Privacy Boundary

```
+----------------------------------+     +-----------------------------------+
|   Notes (existing)               |     |   Knowledge Base (new)            |
|   -------------------            |     |   ----------------------          |
|   Personal notes    X NO API     |     |   Course notes        YES API    |
|   Random captures   ACCESS       |     |   Atomic notes        ACCESS     |
|   Siri captures                  |     |   Research notes                 |
|   Private stuff                  |     |   Cross-linked                   |
+----------------------------------+     +-----------------------------------+
```

### Dependencies

```
notion-client       # Official Notion SDK
anthropic           # Claude API SDK
click               # CLI framework
python-dotenv       # Environment variable management
```

---

## Phase 0: Manual Setup (No Code)

### 0.1 Create Knowledge Base Database in Notion

**What:** Manually create the new "Knowledge Base" database in Notion with the correct schema. Create and configure the Notion integration.
**Status:** READY
**Depends on:** Nothing

**Steps (all manual in Notion UI):**

1. Create a new full-page database called **"Knowledge Base"** with these properties:

| Property | Type | Purpose |
|----------|------|---------|
| `Title` | Title | Note name / concept name |
| `Tags` | Multi-select | Consistent categorization (auto-populated) |
| `Domain` | Select | High-level field (Neuroscience, Biology, Cognitive Science, CS, Math, etc.) |
| `Related Notes` | Relation (self-referencing) | Semantic connections to other KB pages |
| `Source Note` | Relation (self-referencing) | For atomic notes — points to the raw class note |
| `Is Atomic` | Checkbox | Distinguishes generated atomic notes from raw source notes |
| `UNI Course` | Relation -> Courses DB | Which course this came from |
| `Status` | Select | Options: Raw, Atomized, Linked, Reviewed |
| `Last Processed` | Date | When automation last touched this page |
| `Evergreen` | Checkbox | Marks mature/stable notes |
| `Created` | Created time | Auto-set by Notion |

2. Create a Notion integration at https://www.notion.so/my-integrations
3. Share ONLY the Knowledge Base database (and Courses database) with the integration
4. Do NOT share the Notes database with the integration
5. Store credentials:
   - `NOTION_TOKEN` — integration token
   - `ANTHROPIC_API_KEY` — Claude API key
   - `NOTION_KB_DATABASE_ID` — Knowledge Base database ID
   - `NOTION_NOTES_DATABASE_ID` — Notes database ID (only needed temporarily for migration)

**Exit criteria:**
- [ ] Knowledge Base database exists with all properties
- [ ] Integration created and shared with KB database only
- [ ] `.env` file created with all required credentials (gitignored)

---

## Phase 1: Notion API Utilities & Project Scaffold

### 1.1 Project Structure & Notion Client Wrapper

**What:** Set up the project structure and build a reusable Notion API client wrapper with core operations: read page, read blocks, create page, update properties, search/query database.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 0

/implement Notion Notes Project Scaffold & API Client

Set up the project structure and build a Notion API client wrapper. This is the foundation everything else builds on — every command needs to read from and write to Notion.

**Context:** Uses the official `notion-client` Python SDK. All API calls go through a single client class for consistent error handling, rate limiting, and logging. The Notion API has a rate limit of 3 requests/second — the client must handle this gracefully.

**Files to create:**

1. `notion_notes/__init__.py` (NEW) — Package init
2. `notion_notes/cli.py` (NEW) — Click CLI entry point

```python
import click

@click.group()
@click.option('--dry-run', is_flag=True, help='Show what would happen without writing to Notion')
@click.pass_context
def cli(ctx, dry_run):
    """Notion Atomic Notes — process, atomize, link, and tag your knowledge base."""
    ctx.ensure_object(dict)
    ctx.obj['dry_run'] = dry_run

# Subcommands added by each phase
```

3. `notion_notes/notion_client.py` (NEW) — Notion API wrapper

```python
from dataclasses import dataclass
from notion_client import Client

@dataclass
class NotionPage:
    """Lightweight representation of a Notion page."""
    id: str
    title: str
    properties: dict
    content_blocks: list[dict] | None = None  # Loaded lazily

class NotionClientWrapper:
    """Wrapper around official Notion SDK with rate limiting and helpers."""

    def __init__(self, token: str, dry_run: bool = False):
        self.client = Client(auth=token)
        self.dry_run = dry_run

    # --- Read operations ---
    def get_page(self, page_id: str) -> NotionPage: ...
    def get_page_blocks(self, page_id: str) -> list[dict]: ...
    def query_database(self, database_id: str, filter: dict = None,
                       sorts: list = None, page_size: int = 100) -> list[NotionPage]: ...
    def search_database(self, database_id: str, query: str) -> list[NotionPage]: ...

    # --- Write operations (respect dry_run) ---
    def create_page(self, database_id: str, properties: dict,
                    children: list[dict] = None) -> str: ...
    def update_page_properties(self, page_id: str, properties: dict) -> None: ...
    def append_blocks(self, page_id: str, children: list[dict]) -> None: ...
    def add_relation(self, page_id: str, property_name: str,
                     related_page_ids: list[str]) -> None: ...

    # --- Helpers ---
    def blocks_to_markdown(self, blocks: list[dict]) -> str:
        """Convert Notion blocks to plain text/markdown for Claude input."""
        ...
    def extract_title(self, page: dict) -> str: ...
    def extract_property(self, page: dict, property_name: str) -> any: ...

    # --- Rate limiting ---
    def _rate_limited_call(self, func, *args, **kwargs):
        """Wrapper that respects Notion's 3 req/s rate limit with exponential backoff."""
        ...
```

4. `notion_notes/config.py` (NEW) — Configuration loading

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class NotionNotesConfig:
    notion_token: str
    anthropic_api_key: str
    kb_database_id: str
    notes_database_id: str | None = None  # Only needed for migration
    taxonomy_path: Path = Path("taxonomy.json")
    cache_dir: Path = Path(".cache")
    dry_run: bool = False

def load_config(env_path: Path = Path(".env")) -> NotionNotesConfig:
    """Load configuration from .env file and environment variables."""
    ...
```

5. `notion_notes/claude_client.py` (NEW) — Claude API wrapper

```python
from anthropic import Anthropic

class ClaudeClient:
    """Thin wrapper around Anthropic SDK for structured prompting."""

    def __init__(self, api_key: str, default_model: str = "claude-haiku-4-5-20251001"):
        self.client = Anthropic(api_key=api_key)
        self.default_model = default_model

    def prompt(self, system: str, user: str, model: str = None,
              max_tokens: int = 4096) -> str:
        """Send a prompt and return the text response."""
        ...

    def prompt_json(self, system: str, user: str, model: str = None,
                   max_tokens: int = 4096) -> dict:
        """Send a prompt and parse the response as JSON."""
        ...
```

6. `tests/test_notion_client.py` (NEW) — Unit tests with mocks
7. `.env.example` (NEW) — Template for required env vars
8. `.gitignore` update — Ensure `.env`, `.cache/`, `taxonomy.json` are gitignored

**Test plan:**
```
1. NotionClientWrapper initializes with a token
2. blocks_to_markdown converts paragraph, heading, bulleted_list, numbered_list blocks correctly
3. extract_title handles title property correctly
4. extract_property handles multi-select, select, checkbox, date, relation types
5. dry_run mode prevents all write operations (create_page, update_page, append_blocks)
6. Rate limiter respects 3 req/s limit (mock timing test)
7. ClaudeClient.prompt returns text response (mock API)
8. ClaudeClient.prompt_json parses valid JSON from response
9. Config loads from .env file correctly
10. Config raises clear error when required vars are missing
```

**Exit criteria:**
- [ ] `notion_notes` package importable
- [ ] CLI entry point works: `python -m notion_notes --help` shows commands
- [ ] All Notion API operations tested with mocks
- [ ] Rate limiting verified
- [ ] Dry-run mode prevents writes
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

### 1.2 Migration Script

**What:** One-time migration of course/research notes from the existing Notes database to the new Knowledge Base database. Transfers content, maps properties, re-establishes relations.
**Status:** BLOCKED (depends on 1.1 + Phase 0)
**Review Tier:** 2
**Depends on:** Phase 1.1, Phase 0

/implement Notes to Knowledge Base Migration Script

Build and test the one-time migration script that moves course notes from the existing Notes DB to the new Knowledge Base DB. Must handle content transfer, property mapping, relation re-establishment, and produce a migration report.

**Context:** The existing Notes database contains personal + course notes. Only course/research notes should migrate. The migration script needs temporary access to both databases. After migration, revoke access to Notes DB.

**Files to create:**

1. `notion_notes/migration.py` (NEW) — Migration logic

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MigrationConfig:
    source_db_id: str           # Notes database
    target_db_id: str           # Knowledge Base database
    dry_run: bool = False

    # Filters: which pages to migrate
    require_course: bool = True           # Only pages with UNI Course set
    include_project_names: list[str] = None  # e.g., ["Miki London lab"]
    include_page_ids: list[str] = None    # Manual additions
    exclude_page_ids: list[str] = None    # Manual exclusions

@dataclass
class MigrationReport:
    total_found: int
    migrated: int
    skipped: int
    failed: int
    relation_remaps: int
    errors: list[tuple[str, str]]         # (page_title, error)
    id_mapping: dict[str, str]            # old_id -> new_id

class NoteMigrator:
    """Migrate pages from Notes DB to Knowledge Base DB."""

    def __init__(self, client: NotionClientWrapper, config: MigrationConfig): ...

    def migrate(self) -> MigrationReport:
        """
        Full migration pipeline:
        1. Query source DB with filters
        2. For each matching page:
           a. Read all content blocks
           b. Create new page in target DB with mapped properties
           c. Copy content blocks to new page
           d. Add "Migrated to KB on [date]" note to original
        3. Re-establish Related Notes relations using ID mapping
        4. Generate report
        """
        ...

    def _should_migrate(self, page: NotionPage) -> bool: ...
    def _map_properties(self, source_props: dict) -> dict:
        """Map Notes DB properties to Knowledge Base properties.
        Tags -> Tags, UNI Courses -> UNI Course, Evergreen -> Evergreen,
        Status: map existing values to Raw/Atomized/Linked/Reviewed."""
        ...
    def _remap_relations(self, id_mapping: dict) -> int: ...
```

2. Add `migrate` subcommand to `notion_notes/cli.py`

```bash
python -m notion_notes migrate --source NOTES_DB_ID --target KB_DB_ID [--dry-run]
```

Arguments:
- `--source` (required): Notes database ID
- `--target` (required): Knowledge Base database ID
- `--require-course` (default: True): Only migrate pages with UNI Course
- `--include-projects` (optional): Comma-separated project names
- `--include-pages` (optional): Comma-separated page IDs
- `--dry-run`: Show what would migrate without writing

3. `tests/test_migration.py` (NEW) — Tests with mocks

**Test plan:**
```
1. Migration filters correctly: pages with UNI Course are included, others excluded
2. Property mapping converts Tags, UNI Course, Evergreen, Status correctly
3. Status mapping: existing values map to Raw/Atomized/Linked/Reviewed
4. Content blocks are copied to new page
5. Original page gets "Migrated to KB" callout block
6. Relation remapping uses correct ID mapping (old_id -> new_id)
7. Dry-run mode produces report without creating any pages
8. Failed pages don't halt migration; errors collected in report
9. MigrationReport has accurate counts
```

**Exit criteria:**
- [ ] `python -m notion_notes migrate --dry-run` runs without error
- [ ] Dry-run report shows correct page count and property mappings
- [ ] Full migration transfers content and properties correctly (test on 2-3 pages)
- [ ] Relations re-established between migrated pages
- [ ] Migration report saved to disk
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 1 Gate

Before starting Phase 2:
- [x] Project structure set up with CLI entry point
- [x] Notion API wrapper tested with mocks for all operations
- [x] Claude API wrapper tested
- [ ] Migration script tested (at least dry-run on real data)
- [x] All tests pass

---

## Phase 2: Tag Command

### 2.1 Taxonomy-Based Tagging

**What:** CLI command that reads page content, sends it to Claude with the current taxonomy, applies `Domain` and `Tags` properties. Maintains a growing `taxonomy.json` file.
**Status:** BLOCKED (depends on Phase 1)
**Review Tier:** 2
**Depends on:** Phase 1.1

/implement Notion Notes Tag Command

Build the `tag` command — the simplest of the three core commands and the first to use Claude for content analysis. Tags become the fast-filter for the `link` command later.

**Context:** Start simple. Haiku is sufficient for tagging (~200 tokens/note, $0.05 per 100 notes). The taxonomy grows over time but prefers existing tags over new ones.

**Files to create:**

1. `notion_notes/commands/tag.py` (NEW) — Tag command logic

```python
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class TagConfig:
    taxonomy_path: Path = Path("taxonomy.json")
    model: str = "claude-haiku-4-5-20251001"  # Haiku sufficient for tagging
    max_tags: int = 4

class Tagger:
    """Apply consistent domain + tags to Knowledge Base pages."""

    def __init__(self, notion: NotionClientWrapper, claude: ClaudeClient,
                 config: TagConfig): ...

    def tag_pages(self, pages: list[NotionPage]) -> dict:
        """
        For each page:
        1. Read content (blocks -> markdown)
        2. Send content + current taxonomy to Claude
        3. Parse Claude response: {domain, tags, new_tags}
        4. Update page Domain and Tags properties
        5. If new tags suggested: add to taxonomy.json
        """
        ...

    def _build_tag_prompt(self, content: str, taxonomy: dict) -> str:
        """
        Build Claude prompt:
        'Categorize this note using the existing taxonomy where possible.
        If the note covers a concept not captured by existing tags, suggest
        a new tag following the naming convention (lowercase, hyphenated).
        Assign 1 domain and 1-4 tags. Prefer existing tags over new ones.'
        """
        ...

    def _load_taxonomy(self) -> dict: ...
    def _save_taxonomy(self, taxonomy: dict) -> None: ...
```

2. `taxonomy.json` (NEW) — Initial taxonomy seed

```json
{
  "domains": ["Neuroscience", "Biology", "Cognitive Science", "Computer Science",
              "Mathematics", "Statistics", "Psychology"],
  "tags": {
    "Neuroscience": ["synaptic-plasticity", "neural-circuits", "sensory-processing",
                     "motor-systems", "neurotransmitters", "electrophysiology"],
    "Biology": ["cell-biology", "genetics", "immunology", "evolution",
                "molecular-biology", "developmental-biology"],
    "Cognitive Science": ["perception", "memory", "decision-making", "language",
                          "attention", "consciousness"],
    "Computer Science": ["algorithms", "machine-learning", "signal-processing",
                         "data-structures"],
    "Mathematics": ["linear-algebra", "calculus", "probability", "statistics"],
    "Statistics": ["hypothesis-testing", "regression", "bayesian-methods"],
    "Psychology": ["behavioral", "clinical", "social", "developmental"]
  }
}
```

3. Add `tag` subcommand to CLI:

```bash
python -m notion_notes tag [--page PAGE_ID | --untagged | --retag-all]
```

Arguments:
- `--page` (optional): Tag a specific page
- `--untagged` (flag): Tag all pages without Tags set
- `--retag-all` (flag): Re-tag all pages (overwrites existing tags)
- (inherited) `--dry-run`: Show proposed tags without writing

4. `tests/test_tagger.py` (NEW)

**Test plan:**
```
1. Tagger correctly parses Claude JSON response with domain + tags
2. Existing taxonomy tags are preferred (mock Claude to return existing tags)
3. New tags are added to taxonomy.json when suggested
4. Domain is a single value from the allowed list
5. Tags limited to max_tags (default 4)
6. Page properties updated correctly (Domain as select, Tags as multi-select)
7. --untagged filter only returns pages with empty Tags property
8. Dry-run shows proposed tags without writing to Notion
9. Taxonomy file created if missing (with seed data)
10. Invalid Claude response (bad JSON) handled gracefully with retry
```

**Exit criteria:**
- [ ] `python -m notion_notes tag --untagged --dry-run` runs and shows proposed tags
- [ ] Tags applied to 10-20 real notes and verified manually in Notion
- [ ] Taxonomy file grows with new tags as needed
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 2 Gate

Before starting Phase 3:
- [x] Tag command works end-to-end on real Knowledge Base pages
- [x] Taxonomy file populated with initial domains and tags
- [ ] Tags verified as sensible by manual review
- [x] All tests pass

---

## Phase 3: Atomize Command

### 3.1 Non-Destructive Concept Splitting

**What:** CLI command that reads raw class notes, sends content to Claude to identify distinct concepts, creates new atomic Notion pages linked back to the source note. The original note is never modified (except for a callout summarizing what was created).
**Status:** BLOCKED (depends on Phase 2)
**Review Tier:** 2
**Depends on:** Phase 1.1 (API client), Phase 2.1 (tagging — atomic notes get auto-tagged)

/implement Notion Notes Atomize Command

Build the `atomize` command — takes raw class notes and generates atomic concept pages from them. This is the most valuable command: it does the time-consuming structuring work that killed previous Zettelkasten attempts.

**Context:** Sonnet recommended here — quality of atomic note generation matters more than cost ($1.50 vs $0.50 per 100 notes). Each atomic note must be self-contained, accurate, and faithful to what was taught. Short/shorthand notes should be fleshed out. Over-splitting must be avoided.

**Key principle:** The original note is NEVER rewritten. Atomic pages are derived views that link back to the source.

**Files to create:**

1. `notion_notes/commands/atomize.py` (NEW) — Atomize command logic

```python
from dataclasses import dataclass

@dataclass
class AtomizeConfig:
    model: str = "claude-sonnet-4-6"  # Sonnet for quality
    min_concepts: int = 1
    max_concepts: int = 15            # Safety limit
    auto_tag: bool = True             # Tag atomic notes after creation

@dataclass
class AtomicNote:
    title: str                        # Concept name (not a sentence)
    content: str                      # Self-contained explanation
    domain: str                       # Suggested domain
    tags: list[str]                   # Suggested tags
    connections: list[str]            # Suggested connections to other concepts

@dataclass
class AtomizeResult:
    source_page_id: str
    source_title: str
    atomic_notes_created: list[str]   # Page IDs of created atomic notes
    skipped_reason: str | None        # If skipped (already atomized, too short, etc.)

class Atomizer:
    """Split raw class notes into atomic concept pages."""

    def __init__(self, notion: NotionClientWrapper, claude: ClaudeClient,
                 config: AtomizeConfig, tagger: Tagger = None): ...

    def atomize_pages(self, pages: list[NotionPage]) -> list[AtomizeResult]:
        """
        For each page:
        1. Skip if already atomized (Status == "Atomized" or has atomic children)
        2. Read full content (blocks -> markdown)
        3. Send to Claude for concept identification
        4. For each concept, create a new Notion page:
           - Title: concept name
           - Content: extracted content (self-contained)
           - Source Note relation -> original page
           - Is Atomic = True
           - Domain + Tags auto-applied
           - Status = "Atomized"
        5. Update original page:
           - Add callout block at top: "Atomized into N concept notes: [links]"
           - Set Status -> "Atomized"
           - Do NOT modify original content
        """
        ...

    def _build_atomize_prompt(self, content: str, title: str) -> str:
        """
        Build Claude prompt:
        'You are processing a student's class notes into atomic notes.
        For each distinct concept, claim, mechanism, or idea in the source:
        - Give it a concise title (the concept name, not a sentence)
        - Write a clear, self-contained atomic note that captures the idea fully.
          Use the student's own phrasing as the foundation but ensure the note
          makes sense on its own without the surrounding lecture context.
        - If the student's notes are brief or shorthand on a point, flesh it out
          into a complete explanation.
        - Suggest which domain and tags apply
        - Note connections to other concepts mentioned in this note

        Important: only create separate atomic notes when there are genuinely
        distinct concepts. Don't over-split.'

        Response format: JSON array of {title, content, domain, tags, connections}
        """
        ...

    def _create_atomic_page(self, source_page_id: str, note: AtomicNote,
                            database_id: str) -> str: ...
    def _add_summary_callout(self, page_id: str, atomic_page_ids: list[str],
                             atomic_titles: list[str]) -> None: ...
```

2. Add `atomize` subcommand to CLI:

```bash
python -m notion_notes atomize [--page PAGE_ID | --unprocessed | --recent N]
```

Arguments:
- `--page` (optional): Atomize a specific page
- `--unprocessed` (flag): Atomize all pages with Status = "Raw"
- `--recent N` (optional): Atomize N most recently created pages
- (inherited) `--dry-run`: Show proposed atomic notes without creating them

3. `tests/test_atomizer.py` (NEW)

**Test plan:**
```
1. Atomizer correctly parses Claude JSON response into AtomicNote objects
2. Each atomic page created with: correct title, Is Atomic = True, Source Note relation
3. Original page gets callout block with links to atomic notes
4. Original page Status updated to "Atomized"
5. Original page content is NOT modified (only callout added)
6. Already-atomized pages are skipped
7. Single-concept notes produce exactly 1 atomic note (no over-splitting)
8. max_concepts limit prevents runaway generation
9. Dry-run shows proposed atomic notes without creating any pages
10. Auto-tagging applies domain + tags to created atomic notes
11. Claude response with bad JSON retried gracefully
```

**Exit criteria:**
- [ ] `python -m notion_notes atomize --page PAGE_ID --dry-run` shows proposed splits
- [ ] Atomized 3-5 real lecture notes and verified quality in Notion
- [ ] Atomic notes are self-contained, accurate, and link back to source
- [ ] Original notes have summary callout and Status = "Atomized"
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 3 Gate

Before starting Phase 4:
- [x] Atomize command works end-to-end on real class notes
- [x] Atomic notes are high quality (self-contained, accurate, properly linked)
- [x] No over-splitting (distinct concepts only)
- [x] All tests pass

---

## Phase 4: Link Command

### 4.1 Semantic Connection Discovery

**What:** CLI command that scans atomic (and raw) notes, discovers semantic connections across courses/topics, and creates `Related Notes` relations between connected pages. Uses tag/domain overlap as a fast pre-filter, then Claude for semantic comparison.
**Status:** DONE
**Review Tier:** 2
**Depends on:** Phase 2.1 (tags for fast filtering), Phase 3.1 (atomic notes to link)

/implement Notion Notes Link Command

Build the `link` command — discovers and creates semantic connections between knowledge base pages. This is the most expensive command (each note compared against 10-20 candidates) but also where the compound value lives: cross-domain connections you didn't explicitly make.

**Context:** The key optimization challenge is avoiding O(n^2) API calls as the database grows. Solution: fast pre-filtering by tag/domain overlap via Notion API, then semantic comparison via Claude only for plausible candidates. A local cache tracks already-evaluated pairs.

**Files to create:**

1. `notion_notes/commands/link.py` (NEW) — Link command logic

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class LinkConfig:
    model: str = "claude-sonnet-4-6"    # Sonnet for nuanced connections
    max_candidates: int = 20            # Max candidates to compare per page
    min_tag_overlap: int = 1            # Minimum shared tags for fast filter
    cache_path: Path = Path(".cache/link_pairs.json")
    connection_types: tuple[str, ...] = (
        "same-concept-different-context",
        "prerequisite-builds-on",
        "contradicts-alternative-view",
        "shares-mechanism-principle",
        "part-of-same-system",
    )

@dataclass
class Connection:
    page_a_id: str
    page_b_id: str
    connection_type: str
    explanation: str                     # One-sentence explanation

@dataclass
class LinkResult:
    page_id: str
    page_title: str
    connections_found: list[Connection]
    candidates_evaluated: int
    skipped_cached: int

class Linker:
    """Discover and create semantic connections between KB pages."""

    def __init__(self, notion: NotionClientWrapper, claude: ClaudeClient,
                 config: LinkConfig): ...

    def link_pages(self, pages: list[NotionPage]) -> list[LinkResult]:
        """
        For each page:
        1. Find candidate pages via fast filters:
           a. Tag overlap (Notion API query: pages sharing >= 1 tag)
           b. Same domain (Notion API query)
           c. Exclude already-evaluated pairs (from cache)
        2. For top candidates, send page + candidates to Claude:
           'Given this note and these candidates, identify which are genuinely
           related. Only flag connections useful for cross-domain understanding.
           Skip trivial connections.'
        3. Create Related Notes relations for confirmed connections
        4. Update cache with evaluated pairs
        5. Update page Status -> "Linked"
        """
        ...

    def _find_candidates(self, page: NotionPage,
                         all_pages: list[NotionPage]) -> list[NotionPage]:
        """
        Fast pre-filtering:
        1. Tag overlap >= min_tag_overlap
        2. Same domain (broadens search)
        3. Exclude self
        4. Exclude already-evaluated pairs (from cache)
        5. Limit to max_candidates
        """
        ...

    def _build_link_prompt(self, page_content: str, page_title: str,
                           candidates: list[tuple[str, str]]) -> str:
        """
        Build Claude prompt for semantic comparison.
        Response format: JSON array of {candidate_title, connection_type, explanation}
        """
        ...

    def _load_cache(self) -> set[tuple[str, str]]: ...
    def _save_cache(self, cache: set[tuple[str, str]]) -> None: ...
    def _pair_key(self, id_a: str, id_b: str) -> tuple[str, str]:
        """Canonical pair key (sorted) to avoid duplicate evaluation."""
        return tuple(sorted([id_a, id_b]))
```

2. Add `link` subcommand to CLI:

```bash
python -m notion_notes link [--page PAGE_ID | --all-unlinked | --recent N]
```

Arguments:
- `--page` (optional): Find connections for a specific page
- `--all-unlinked` (flag): Link all pages with Status != "Linked"
- `--recent N` (optional): Link N most recently created/atomized pages
- (inherited) `--dry-run`: Show proposed connections without creating relations

3. `tests/test_linker.py` (NEW)

**Test plan:**
```
1. Candidate filtering finds pages with shared tags
2. Candidate filtering excludes self and cached pairs
3. Pair key is canonical (sorted) so A-B == B-A
4. Claude response parsed correctly into Connection objects
5. Related Notes relations created bidirectionally
6. Cache updated after evaluation (pair not re-evaluated)
7. Trivial connections (same course, adjacent lecture) filtered out
8. Max candidates limit respected
9. Dry-run shows proposed connections without creating relations
10. Empty candidates list handled gracefully (no Claude call)
11. Already-linked pages can be re-linked to find new connections (incremental)
```

**Exit criteria:**
- [ ] `python -m notion_notes link --page PAGE_ID --dry-run` shows proposed connections
- [ ] Linked 10+ pages and verified connections are genuine and useful
- [ ] Cross-domain connections discovered (e.g., Neuroscience <-> Cognitive Science)
- [ ] Cache prevents re-evaluation of already-checked pairs
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 4 Gate

Before starting Phase 5:
- [x] Link command discovers genuine cross-domain connections
- [x] Cache system prevents redundant API calls
- [x] Relations created bidirectionally in Notion
- [x] All tests pass

---

## Phase 5: Orchestration & Utilities

### 5.1 Process Command & Move Utility

**What:** `process` command that runs tag -> atomize -> link in sequence on unprocessed notes. `move` command that transfers a page from Notes inbox to Knowledge Base. Both are thin wrappers around existing functionality.
**Status:** DONE
**Review Tier:** 1
**Depends on:** Phases 2, 3, 4

/implement Notion Notes Process & Move Commands

Build the `process` orchestrator (tag -> atomize -> link in sequence) and the `move` utility (transfer page from Notes inbox to Knowledge Base). These are convenience wrappers.

**Context:** `process` is the "I took a bunch of notes this week, go sort them out" command. `move` is for when you use Notes as an inbox and want to promote a page to KB when ready.

**Files to create:**

1. `notion_notes/commands/process.py` (NEW) — Orchestrator

```python
from dataclasses import dataclass

@dataclass
class ProcessConfig:
    tag_config: TagConfig
    atomize_config: AtomizeConfig
    link_config: LinkConfig

@dataclass
class ProcessReport:
    pages_processed: int
    tags_applied: int
    atomic_notes_created: int
    connections_found: int
    errors: list[str]
    cost_estimate: dict           # {tag: $X, atomize: $Y, link: $Z, total: $T}

class Processor:
    """Run full processing pipeline: tag -> atomize -> link."""

    def __init__(self, notion: NotionClientWrapper, claude: ClaudeClient,
                 config: ProcessConfig): ...

    def process(self, pages: list[NotionPage]) -> ProcessReport:
        """
        Pipeline:
        1. Tag all pages (fast, cheap — Haiku)
        2. Atomize pages that aren't yet atomized (Sonnet)
        3. Link all pages including newly created atomic notes (Sonnet)
        4. Compute cost estimate from token usage
        5. Return summary report
        """
        ...
```

2. `notion_notes/commands/move.py` (NEW) — Inbox to KB mover

```python
class Mover:
    """Transfer a page from Notes inbox to Knowledge Base."""

    def __init__(self, notion: NotionClientWrapper): ...

    def move_page(self, page_id: str, target_db_id: str) -> str:
        """
        1. Read page from Notes DB (content + properties)
        2. Create new page in Knowledge Base with mapped properties
        3. Set Status = "Raw" in KB
        4. Add "Moved to KB" note on original page
        5. Return new page ID
        """
        ...
```

3. Add `process` and `move` subcommands to CLI:

```bash
python -m notion_notes process [--recent N | --unprocessed]
python -m notion_notes move --page PAGE_ID
```

Arguments for `process`:
- `--recent N`: Process N most recently created pages
- `--unprocessed` (flag): Process all pages with Status = "Raw"
- (inherited) `--dry-run`

Arguments for `move`:
- `--page` (required): Page ID to move from Notes to KB

4. Add basic logging to all commands:

```python
# notion_notes/logging_config.py
import logging

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
```

5. `tests/test_processor.py` (NEW)
6. `tests/test_mover.py` (NEW)

**Test plan:**
```
1. Process runs tag -> atomize -> link in correct order
2. Process skips already-processed pages
3. ProcessReport counts are accurate (tags, atomics, connections)
4. Process continues if one page fails (error collected, others processed)
5. Move creates page in target DB with correct properties
6. Move sets Status = "Raw" in new page
7. Move adds note to original page
8. Move in dry-run mode creates nothing
9. Cost estimate calculation uses correct token counts per model
```

**Exit criteria:**
- [ ] `python -m notion_notes process --recent 5` runs full pipeline
- [ ] ProcessReport shows accurate counts and cost estimate
- [ ] Move transfers a page correctly from Notes to KB
- [ ] Logging output is clear and informative
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 5 Gate (Project Complete)

Full system verification (validated 2026-02-20):
- [x] `python -m notion_notes tag --page PAGE_ID` works (live: tagged "Rapid integration paper questions" -> Neuroscience)
- [x] `python -m notion_notes atomize --page PAGE_ID` works (live dry-run: 5 concepts identified)
- [x] `python -m notion_notes link --page PAGE_ID` works (live dry-run: evaluated candidates)
- [x] `python -m notion_notes process --recent 1` runs all three in sequence (live dry-run: 1 tagged, 8 concepts, 3 connections)
- [x] `python -m notion_notes move --page PAGE_ID` transfers a page (live: 6 Miki London lab pages moved)
- [x] `python -m notion_notes migrate` works (live: 216 pages migrated in prior session)
- [x] All commands respect `--dry-run`
- [x] All 424 tests pass
- [x] Bug fix: `--recent N` now correctly limits to N pages (was fetching all due to Notion API `page_size` semantics)

---

## Dependency Graph

```
Phase 0 (Manual: DB + Integration setup)
    |
    v
Phase 1.1 (API Client + Project Scaffold)
    |        \
    v         v
Phase 1.2   Phase 2.1
(Migration)  (Tag)
              |
              v
            Phase 3.1
            (Atomize)
              |
              v
            Phase 4.1
            (Link)
              |
              v
            Phase 5.1
            (Process + Move)
```

---

## Recommended Execution Order

| Priority | Module | Why |
|----------|--------|-----|
| **1** | Phase 0 (Manual setup) | Everything else depends on the database existing |
| **2** | Phase 1.1 (API client) | Foundation for all commands |
| **3** | Phase 2.1 (Tag) | Simplest command, immediately useful, enables link filtering |
| **4** | Phase 3.1 (Atomize) | Core value — the note structuring you can't do manually |
| **5** | Phase 4.1 (Link) | Compound value — cross-domain connections |
| **6** | Phase 5.1 (Process + Move) | Convenience wrappers, quick to build |
| **7** | Phase 1.2 (Migration) | Can defer — start fresh in KB and migrate later |

---

## Cost & Performance Estimates

| Command | Tokens/note | Cost per 100 notes | Model |
|---------|-------------|---------------------|-------|
| Tag | ~200 | ~$0.05 | Haiku |
| Atomize | ~2000 | ~$1.50 | Sonnet |
| Link | ~3000 (page + candidates) | ~$1-2 | Sonnet |

**Full semester processing** (~200-400 pages): ~$5-15 per run depending on model mix.

**Runtime:** 5-15 minutes for a few hundred notes (Notion API rate limit is the bottleneck at 3 req/s, not Claude).

---

## Model Selection Guide

| Task | Model | Rationale |
|------|-------|-----------|
| Tagging | Haiku | Simple classification, cost-efficient |
| Atomizing | Sonnet | Quality matters — notes must be accurate and self-contained |
| Linking | Sonnet | Nuanced semantic comparison across domains |
| Migration | N/A | No Claude needed — just API data transfer |

---

## Project Structure

```
notion_notes/
+-- __init__.py
+-- cli.py                      # Click CLI entry point
+-- config.py                   # Configuration loading (.env)
+-- notion_client.py            # Notion API wrapper with rate limiting
+-- claude_client.py            # Claude API wrapper
+-- migration.py                # One-time Notes -> KB migration
+-- logging_config.py           # Logging setup
+-- commands/
|   +-- __init__.py
|   +-- tag.py                  # Tag command
|   +-- atomize.py              # Atomize command
|   +-- link.py                 # Link command
|   +-- process.py              # Process orchestrator
|   +-- move.py                 # Inbox -> KB mover
tests/
+-- test_notion_client.py
+-- test_claude_client.py
+-- test_migration.py
+-- test_tagger.py
+-- test_atomizer.py
+-- test_linker.py
+-- test_processor.py
+-- test_mover.py
.env                            # Credentials (gitignored)
.env.example                    # Template
taxonomy.json                   # Growing tag taxonomy
.cache/
+-- link_pairs.json             # Evaluated pair cache
```

---

## What This Does NOT Do

- Does not replace your note-taking or learning — you take the raw notes, the automation structures them
- Does not run automatically — you decide when to process
- Does not require switching away from Notion
- Does not try to be a study tool — it's a librarian, not a tutor
- Does not touch the private Notes database (privacy boundary enforced)

---

## Borrowed Principles (from arscontexta)

1. **Reduce (-> `atomize`)**: Raw lecture notes distilled into atomic, self-contained claims
2. **Reflect (-> `link`)**: Periodic scanning for connections you didn't explicitly make
3. **Reweave (-> incremental `link`)**: New notes automatically connected to existing knowledge

---

## Knowledge Base Properties Reference

| Property | Type | Values/Notes |
|----------|------|--------------|
| `Title` | Title | Note name / concept name |
| `Tags` | Multi-select | Grows via taxonomy.json |
| `Domain` | Select | Neuroscience, Biology, Cognitive Science, CS, Math, Statistics, Psychology |
| `Related Notes` | Relation (self) | Bidirectional semantic connections |
| `Source Note` | Relation (self) | For atomic notes -> raw source |
| `Is Atomic` | Checkbox | True for generated atomic notes |
| `UNI Course` | Relation | Which course (linked to Courses DB) |
| `Status` | Select | Raw, Atomized, Linked, Reviewed |
| `Last Processed` | Date | Automation timestamp |
| `Evergreen` | Checkbox | Mature/stable notes |
| `Created` | Created time | Auto-set |
