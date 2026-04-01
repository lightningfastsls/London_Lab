# Notion Atomic Notes Automation — Implementation Plan

## What This Is

A CLI tool (Python script) that you run manually whenever you want to process your Notion notes. It does three things:

1. **Atomize** — Takes your raw class notes and generates atomic concept pages from them. Your original note stays untouched as a source record. New atomic pages are created with full explanations and linked back to the source.
2. **Link** — Scans atomic notes and discovers semantic connections across courses/topics. Adds Notion relation properties between related pages.
3. **Tag** — Applies consistent categorization using a taxonomy that grows with your knowledge base.

You run it when you feel like it. No schedules, no hooks, no background processes.

---

## Architecture

```
You take notes in class → Notes DB (inbox) or directly into Knowledge Base
        |
        v
Optionally: python notion_notes.py move --page PAGE_ID  (inbox → KB)
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

Note: The automation never sees the Notes database.
```

### Dependencies
- Python 3.10+
- `notion-client` (official Notion SDK)
- `anthropic` (Claude API SDK)
- A Notion integration token with read/write access to your notes database
- Anthropic API key

---

## Database Architecture

### The Problem

The current "Notes" database is a general-purpose inbox that contains personal notes, research notes, course notes, and everything else. The Claude API integration should NOT have access to personal content. We need a clean separation.

### The Solution: A New Dedicated Database

Create a new database called **"Knowledge Base"** (or whatever name you prefer). This is the only database the automation touches. The existing Notes database stays exactly as it is — your personal inbox, untouched.

#### Privacy Boundary

```
┌─────────────────────────────────┐     ┌──────────────────────────────────┐
│   Notes (existing)              │     │   Knowledge Base (new)           │
│   ─────────────────             │     │   ──────────────────             │
│   Personal notes    ✗ NO API    │     │   Course notes        ✓ API     │
│   Random captures   ACCESS      │     │   Atomic notes        ACCESS    │
│   Siri captures                 │     │   Research notes                │
│   Private stuff                 │     │   Cross-linked                  │
└─────────────────────────────────┘     └──────────────────────────────────┘
```

The Notion integration token is ONLY shared with the Knowledge Base database (and any related databases it needs, like Courses). Your Notes database is never shared with the integration.

### Knowledge Base — Properties

| Property | Type | Purpose |
|----------|------|---------|
| `Title` | Title | Note name / concept name |
| `Tags` | Multi-select | Consistent categorization (auto-populated) |
| `Domain` | Select | High-level field (Neuroscience, Biology, Cognitive Science, CS, Math, etc.) |
| `Related Notes` | Relation (self-referencing) | Semantic connections to other knowledge base pages |
| `Source Note` | Relation (self-referencing) | For atomic notes — points to the raw class note they came from |
| `Is Atomic` | Checkbox | Distinguishes generated atomic notes from raw source notes |
| `UNI Course` | Relation → Courses DB | Which course this came from |
| `Status` | Select | Options: Raw, Atomized, Linked, Reviewed |
| `Last Processed` | Date | When automation last touched this page |
| `Evergreen` | Checkbox | Carried over from your current system — marks mature/stable notes |
| `Created` | Created time | Auto-set by Notion |

### Migration Plan: Move Course Notes to Knowledge Base

A one-time migration script that:

1. **Queries the existing Notes database** filtering by:
   - `UNI Courses` is not empty, OR
   - `Projects` contains a research project (e.g., "Miki London lab"), OR
   - Manual list of page IDs you want to migrate
2. **For each note to migrate:**
   - Reads all content (blocks) from the original page
   - Creates a new page in Knowledge Base with the same title and content
   - Maps existing properties where they carry over:
     - `Tags` → `Tags`
     - `UNI Courses` → `UNI Course`
     - `Status` → `Status` (map "Leaf" to "Raw" or whatever makes sense)
     - `Evergreen` → `Evergreen`
     - `Related Notes` → will need to be re-established after all pages are migrated
   - Adds a note to the original page: "Migrated to Knowledge Base on [date]"
   - Does NOT delete the original (you can clean up manually later)
3. **After all pages are migrated:**
   - Re-establish `Related Notes` relations between migrated pages in the new database
   - Generate a migration report: what moved, what was skipped, any issues

```bash
python notion_notes.py migrate --source NOTES_DB_ID --target KB_DB_ID [--dry-run]
```

**Important:** The migration script needs temporary access to both databases. After migration is complete, you can revoke the integration's access to the Notes database so the automation only ever sees the Knowledge Base going forward.

### Updating Course Pages

Your course pages currently likely embed or link to the Notes database (filtered by course). After migration:

1. Each course page should get a new linked view of the Knowledge Base, filtered by `UNI Course = [this course]`
2. You can keep the old Notes database view alongside it during transition
3. Once you've confirmed everything migrated correctly, remove the old view

The script can generate the filter configs, but you'll need to manually add the linked database views to each course page (Notion API can't insert linked database views into page content — this is a Notion API limitation).

### Going Forward: New Notes Workflow

After migration, your workflow changes slightly:

**Before (current):**
1. Take notes in class → they go into Notes database
2. Manually decide what to do with them

**After:**
1. Take notes in class → put them directly into Knowledge Base (or into Notes as inbox, then move to KB when ready)
2. Run `python notion_notes.py process --recent 10` when you want
3. Automation tags, atomizes, and links them

You can also keep using Notes as an inbox and have a simple `move` command that transfers a page from Notes to Knowledge Base when you've decided it's worth keeping:

```bash
python notion_notes.py move --page PAGE_ID
```

---

## Commands

### `atomize` — Non-Destructive Concept Splitting

```bash
python notion_notes.py atomize [--page PAGE_ID | --unprocessed | --recent N]
```

**What it does:**
1. Reads the content of a note (or batch of unprocessed notes)
2. Sends content to Claude to identify distinct concepts and generate proper atomic notes from your raw class notes:

```
You are processing a student's class notes into atomic notes. 
For each distinct concept, claim, mechanism, or idea in the source:

- Give it a concise title (the concept name, not a sentence)
- Write a clear, self-contained atomic note that captures the idea fully. 
  Use the student's own phrasing as the foundation but ensure the note 
  makes sense on its own without the surrounding lecture context.
- If the student's notes are brief or shorthand on a point, flesh it out 
  into a complete explanation — the student was in the lecture and 
  understands it, but their future self (or a search) needs the full picture.
- Suggest which domain and tags apply
- Note connections to other concepts mentioned in this note or likely 
  connections to concepts in other fields

Important: only create separate atomic notes when there are genuinely 
distinct concepts. Don't over-split — a single mechanism described in 
detail is one note, not five.
```

3. For each identified concept, creates a new Notion page:
   - Title: concept name
   - Content: extracted content (preserving original phrasing)
   - `Source Note` relation → points to original page
   - `Is Atomic` → checked
   - `Tags` and `Domain` → auto-applied
   - `Status` → "Atomized"
4. Updates the original note:
   - Adds a callout block at the top: "Atomized into N concept notes: [links]"
   - Sets `Status` → "Atomized"
   - Does NOT modify the original content

**Key principle:** The original note is never rewritten. The atomic pages are *derived* views that link back. You can always find the full original context.

### `link` — Discover and Create Connections

```bash
python notion_notes.py link [--page PAGE_ID | --all-unlinked | --recent N]
```

**What it does:**
1. For a given page (or batch), reads its content and current tags
2. Searches existing database for candidate related pages using:
   - Tag overlap (fast filter via Notion API)
   - Same domain (fast filter)
   - Then semantic comparison via Claude for top candidates
3. Sends the page + candidates to Claude:

```
Given this note and these candidate notes, identify which are genuinely 
related and explain the connection in one sentence. Types of connections:
- Same concept in different contexts
- Prerequisite / builds-on relationship  
- Contradicts or offers alternative view
- Shares underlying mechanism or principle
- Part of the same system or pathway

Only flag connections that would be useful for a student trying to 
build cross-domain understanding. Skip trivial connections.
```

4. Creates `Related Notes` relations for confirmed connections
5. Optionally adds a brief connection annotation as a comment on the relation (Notion supports this in databases)

**Optimization:** To avoid quadratic API calls as the database grows, the script:
- First filters by tag/domain overlap (cheap, via Notion API)
- Only sends semantically plausible candidates to Claude
- Tracks which pairs have already been evaluated (via a local cache file)
- Processes incrementally — new notes get linked to existing ones, not the entire database re-evaluated

### `tag` — Consistent Categorization

```bash
python notion_notes.py tag [--page PAGE_ID | --untagged | --retag-all]
```

**What it does:**
1. Maintains a taxonomy file (`taxonomy.json`) that grows over time:

```json
{
  "domains": ["Neuroscience", "Biology", "Cognitive Science", "Computer Science", "Mathematics", "Statistics"],
  "tags": {
    "Neuroscience": ["synaptic-plasticity", "neural-circuits", "sensory-processing", "motor-systems", ...],
    "Biology": ["cell-biology", "genetics", "immunology", "evolution", ...],
    "Cognitive Science": ["perception", "memory", "decision-making", "language", ...],
    ...
  }
}
```

2. For untagged pages, sends content + current taxonomy to Claude:

```
Categorize this note using the existing taxonomy where possible. 
If the note covers a concept not captured by existing tags, suggest 
a new tag following the naming convention (lowercase, hyphenated).
Assign 1 domain and 1-4 tags. Prefer existing tags over new ones.
```

3. Updates the page's `Domain` and `Tags` properties
4. If new tags were created, adds them to `taxonomy.json`

### `process` — Run Everything

```bash
python notion_notes.py process [--recent N]
```

Runs `tag` → `atomize` → `link` in sequence on unprocessed notes. This is your "I took a bunch of notes this week, go sort them out" command.

---

## Implementation Order

### Step 0: Create Knowledge Base Database in Notion
- Manually create the new "Knowledge Base" database in Notion with the properties listed above
- Create a Notion integration at https://www.notion.so/my-integrations
- Share ONLY the Knowledge Base database (and Courses database) with the integration
- Do NOT share the Notes database with the integration
- Store the token securely (env variable or .env file)

### Step 1: Build Notion API Utilities + Migration Script
- Write utility functions: read page, read page content (blocks), create page, update properties, search database
- Build and test the migration script with `--dry-run` first
- Run migration of course/research notes from Notes → Knowledge Base
- Re-establish relations between migrated pages
- Verify everything looks correct in the new database
- Update course pages with new linked database views (manual step)
- Revoke integration access to the Notes database

### Step 2: Implement `tag`
Start here because it's the simplest and immediately useful. Also, tags become the fast-filter for the `link` command later.
- Build the taxonomy file with initial domains based on your current courses
- Implement the Claude prompt for categorization
- Test on 10-20 migrated notes, verify tags make sense
- Iterate on taxonomy

### Step 3: Implement `atomize`
- Build the content extraction (Notion blocks → text for Claude)
- Build the page creation (Claude output → new Notion pages with relations back to source)
- Handle edge cases: short notes that are already about one concept, notes with shorthand that need fleshing out
- Test on a few long lecture notes, verify the atomic notes are accurate, self-contained, and faithful to what was taught

### Step 4: Implement `link`
- Build the candidate search (tag overlap + domain matching)
- Build the semantic comparison prompt
- Build the local cache for already-evaluated pairs
- Implement the relation creation
- Test incrementally — start with notes from one course, verify connections, then expand

### Step 5: Implement `process` and `move` wrappers
- `process`: orchestrate tag → atomize → link in sequence
- `move`: transfer a page from Notes inbox to Knowledge Base
- Add a `--dry-run` flag that shows what would happen without writing to Notion
- Add basic logging so you can review what the script did

---

## Cost and Performance Estimates

- **Tagging**: ~200 tokens per note (input + output). 100 notes ≈ $0.05 with Haiku.
- **Atomizing**: ~2000 tokens per note (full content in + generated atomic notes out). 100 notes ≈ $0.50 with Haiku, ~$1.50 with Sonnet. Sonnet recommended here — quality of atomic note generation matters.
- **Linking**: Most expensive — each note compared against ~10-20 candidates. 100 notes ≈ $1-2 with Haiku.
- **Total for a semester of notes** (~200-400 pages): roughly $5-15 per full processing run depending on model mix. Use Haiku for tag, Sonnet for atomize and link.

Running time: expect ~5-15 minutes for a full run on a few hundred notes (Notion API rate limits are the bottleneck, not Claude).

---

## What This Does NOT Do

- Does not replace your note-taking or your learning — you're in the lecture, you're engaged, you take the raw notes. The automation handles the structuring and connecting afterward.
- Does not run automatically — you decide when to process
- Does not require switching away from Notion
- Does not try to be a study tool — it's a librarian, not a tutor

---

## What Arscontexta Taught Us (Borrowed Principles)

Three ideas from arscontexta are embedded in this design:

1. **Reduce (→ `atomize`)**: Taking raw captured material and distilling it into atomic, self-contained claims. Your class notes are the raw capture; the automation does the reduction into atomic pieces — the time-consuming structuring work that killed your previous Zettelkasten attempt.

2. **Reflect (→ `link`)**: Periodically scanning for connections you didn't explicitly make. The cross-domain linking is where the real compound value lives, especially across fields in a master's program.

3. **Reweave (→ incremental `link`)**: When new notes arrive, going back to find connections to older material. The incremental processing model means every new concept automatically gets connected to the existing knowledge base.

Everything else (MOCs, subagent spawning, hooks, session management) is agent-context infrastructure that doesn't apply to a personal knowledge base in Notion.
