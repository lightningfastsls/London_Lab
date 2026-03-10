# Knowledge Activation Architecture — Implementation Roadmap

> **Phase 15** of the USV Detection & Analysis project.
> Closes the knowledge activation gap: vault notes (505+) are high-quality but only surfaced
> when the agent voluntarily searches. These four modules implement activation at session start,
> mid-session editing, source file reading, and cross-agent handoff.
>
> **Research basis:** Self-RAG, FLARE, CRAG, and Adaptive RAG patterns applied to agent-operated
> knowledge systems. See `knowledge-activation-plan.md` for full research context.

---

## How to Use This File

1. Work through modules **in order** (15.1 → 15.4)
2. Each module has a self-contained `/implement` block
3. After each module: commit, verify, update this file's status
4. Phase gate must pass before considering extensions (PreToolUse hooks, skill-level gates)

## Status Key

- **DONE** — Implemented and tested
- **READY** — Dependencies met, can start
- **BLOCKED** — Waiting on dependency or external input

---

### 15.1 /kcheck Micro-Skill

**What:** FLARE-inspired mid-session knowledge retrieval skill that searches the vault based on the agent's stated intent before modifying constrained systems.
**Status:** READY
**Review Tier:** 1 (simple skill wrapper around qmd)
**Depends on:** None

/implement /kcheck Micro-Skill

Create a lightweight `/kcheck` skill that takes a brief description of intended work as input, runs qmd searches against the vault, and shows relevant note titles + descriptions so the agent can decide what to load fully. This is the FLARE pattern (Forward-Looking Active REtrieval) adapted for Claude Code: the agent describes what it's about to do, and that description drives a vault search.

**Context:** Inspired by FLARE (Jiang et al., EMNLP 2023) — retrieval triggered by generation intent rather than post-hoc. The relevance threshold follows CRAG's evaluator pattern: only surface results above a cutoff to prevent noise-induced gate fatigue. Follows existing skill patterns in `.claude/skills/`.

**Files to create:**

1. `.claude/skills/kcheck/SKILL.md` (NEW) — Skill definition

    The skill procedure:
    ```
    /kcheck <what I'm about to do>

    1. Extract key nouns/concepts from the input description
    2. Run qmd vector_search with the full description (semantic match, limit 5)
    3. Run qmd search with extracted keywords (exact match, limit 5)
    4. Deduplicate results by note title
    5. Apply relevance threshold — only show results with score >= 0.3 for vector, >= 0.1 for keyword
    6. Display: note title + first sentence of description for top 5-8 results
    7. If any results reference files the agent is about to modify → flag as CRITICAL
    8. Agent decides which notes to load fully with qmd get
    ```

    Skill metadata:
    ```yaml
    name: kcheck
    description: "Knowledge check — search vault for constraints relevant to planned work. Use before modifying detection, export, labeling, or other constrained systems. Triggers on /kcheck, 'knowledge check', 'check vault'."
    ```

    Output format:
    ```
    ## Knowledge Check: <summarized intent>

    ### Relevant Notes
    - **<note-title>** — <first sentence of description> [score: X.XX]
    - **<note-title>** — <first sentence of description> [score: X.XX]

    ### CRITICAL (references files you're modifying)  [only if matches found]
    - **<note-title>** — <constraint summary>

    ### No strong matches  [only if all results below threshold]
    No vault notes found above relevance threshold for this intent.
    ```

    Token cost: ~1,000-1,500 tokens per invocation (8 descriptions × ~150 tokens + overhead).

2. `CLAUDE.md` (EDIT) — Add mid-session knowledge check procedure

    Add to the task procedures / coding conventions area:
    ```markdown
    ## Mid-Session Knowledge Checks
    Before modifying files in high-risk directories (detection app, export adapters,
    labeling pipeline), run `/kcheck "<brief description of planned changes>"`.
    This is mandatory for HIGH-risk canary files and recommended for any non-trivial
    modification to existing systems.

    Skip /kcheck for: new standalone files, test files, documentation-only changes.
    ```

**Test plan:**
    ```
    1. Run /kcheck "modify detection overlay rendering" — verify it returns detection-related vault notes
    2. Run /kcheck "add a new utility function" — verify it returns few/no results (low-risk intent)
    3. Run /kcheck with a description matching a known vault note — verify that note appears in results
    4. Verify CRITICAL flagging works when results reference files mentioned in the intent
    5. Verify results below relevance threshold are filtered out (not shown as noise)
    ```

**Exit criteria:**
- [ ] `/kcheck` skill file exists at `.claude/skills/kcheck/SKILL.md`
- [ ] Skill runs successfully against at least 3 test descriptions
- [ ] CRITICAL flagging works when vault notes reference files in the intent
- [ ] Results respect relevance thresholds (no noise entries)
- [ ] CLAUDE.md updated with `/kcheck` usage procedure
- [ ] py_compile not applicable (markdown/skill only)

---

### 15.2 Goal-Aware Orient Hook

**What:** Enhance the session-orient PowerShell hook to automatically search the vault for notes relevant to active goal threads, writing results to `ops/session-relevance.md` for agent consumption at session start.
**Status:** READY
**Review Tier:** 2 (PowerShell scripting + qmd CLI integration + goals.md parsing)
**Depends on:** None (but shares relevance threshold design with 15.1)

/implement Goal-Aware Orient Hook

Enhance `.claude/hooks/session-orient.ps1` to parse active threads from `ops/goals.md`, run qmd searches for each thread, and write a relevance brief to `ops/session-relevance.md`. This implements baseline RAG at session start — the Self-RAG pattern (Asai et al., ICLR 2024) adapted as a procedural gate since we can't emit reflection tokens.

**Context:** The orient hook already reads goals.md and outputs session state. This enhancement adds vault search after the existing logic. The CRAG-inspired relevance threshold is critical: surfacing noise teaches the agent to ignore the brief entirely. Use qmd CLI (`qmd search "query"`, `qmd vsearch "query"`) since the hook runs in PowerShell outside MCP context. Read the current hook first: `.claude/hooks/session-orient.ps1`. Read `ops/goals.md` to understand active thread format (markdown headings/lists with status markers).

**Files to create:**

1. `.claude/hooks/session-orient.ps1` (EDIT) — Add knowledge activation section

    After the existing goals.md read and session state output, add:

    ```powershell
    # --- Knowledge Activation ---
    # 1. Parse ops/goals.md to extract active thread titles + descriptions
    #    - Active = threads NOT marked "DONE", "COMPLETED", "PAUSED"
    #    - Extract: { title: string, description: string }
    #    - Cap at 5 active threads
    #
    # 2. For each active thread:
    #    a. Run: qmd search "<thread_title>" --limit 2
    #    b. Run: qmd vsearch "<thread_description_or_title>" --limit 2
    #    c. Parse JSON output to get note titles + descriptions
    #    d. Deduplicate by note title
    #    e. Filter by relevance score (keyword >= 0.1, vector >= 0.3) [ASSUMED thresholds]
    #
    # 3. Write results to ops/session-relevance.md
    ```

    Goals.md parsing logic:
    - Scan for `## ` or `### ` headings that represent thread titles
    - Check subsequent lines for status markers (Done, Completed, Paused)
    - Use the first 1-2 sentences after the heading as the description
    - Simple regex-based parsing, not a full markdown parser

    Edge cases:
    - No active threads → write brief noting "no active threads, skipping relevance search"
    - qmd unavailable or errors → write brief noting failure, don't block the hook
    - Thread title very short (< 3 words) → use description for vector search, skip keyword search
    - Zero results above threshold for a thread → note "no strong matches" (don't show weak results)

2. `ops/session-relevance.md` (NEW, auto-generated) — Relevance brief template

    ```markdown
    # Session Relevance Brief (auto-generated)
    <!-- Generated at: <timestamp> -->
    <!-- Source threads: <count> active threads from goals.md -->

    ## Thread: <thread_title>
    - [<note-title>] — <note description, first sentence only>
    - [<note-title>] — <note description, first sentence only>
    > See also: notes/<relevant-topic-map>.md

    ## Thread: <thread_title>
    - No strong matches above relevance threshold.
    ```

    Constraints:
    - Total file must stay under 3,500 tokens (~200 tokens per thread × 5 threads + overhead)
    - Include one topic map pointer per thread if a relevant map exists
    - Use first sentence only of note descriptions (not full description)

3. `CLAUDE.md` (EDIT) — Add session-relevance.md to orient procedure

    In the orient/session-start section, after the goals.md read step, add:
    ```
    - Read ops/session-relevance.md (auto-generated vault relevance brief)
    - If any listed notes are directly relevant to the current task, load them with qmd get
    ```

**Test plan:**
    ```
    1. Run hook manually: powershell .claude/hooks/session-orient.ps1
    2. Verify ops/session-relevance.md is generated with at least one thread section
    3. Verify token count of generated file stays under 3,500 (~14KB text)
    4. Verify qmd queries return meaningful results (check note titles make sense for thread)
    5. Test with goals.md having no active threads — verify graceful "no active threads" message
    6. Test with qmd unavailable (rename binary temporarily) — verify hook completes without error
    7. Verify deduplication: same note found by both keyword and vector search appears only once
    ```

**Exit criteria:**
- [ ] Hook generates `ops/session-relevance.md` with relevant results
- [ ] Relevance threshold filters out weak matches (no noise entries)
- [ ] Token count stays under 3,500
- [ ] Edge cases handled (no threads, qmd errors, short titles)
- [ ] CLAUDE.md orient procedure includes session-relevance.md read step
- [ ] Hook still completes in under 10 seconds (qmd queries are fast)

---

### 15.3 Canary Comments in Source Files

**What:** Add standardized `# VAULT:` comments to high-risk source files that reference vault notes containing architectural constraints. When the agent opens these files to edit, it sees note references inline.
**Status:** READY
**Review Tier:** 1 (comment insertion + mapping document)
**Depends on:** 15.1 (canary comments reference /kcheck for HIGH-risk files)

/implement Canary Comments in Source Files

Add `# VAULT:` comments to high-risk source files that reference vault notes with architectural constraints or regression-causing decisions. This is static activation — when the agent reads a file to edit it, the vault references are right there. Follows the Adaptive RAG principle: route retrieval depth by predicted risk. HIGH-risk files get canaries + mandatory `/kcheck`. MEDIUM-risk files get canaries only. LOW-risk files (standalone utilities, tests, configs) get nothing.

**Context:** The two known high-risk areas from recent bug hunts: (1) detection app's ghost/saved/current detection state tiers, (2) DeepSqueak import's prefix-match vs exact-match behavior. Search the vault to discover additional constraint notes tied to source files. Use `qmd search "architectural invariant"`, `qmd search "design decision"`, `qmd search "bug fix constraint"`, and cross-reference with actual source paths.

**Files to create:**

1. Multiple source files (EDIT) — Insert canary comments

    Comment format (Python):
    ```python
    # VAULT: <note-title-1>, <note-title-2>
    # Run `qmd get "<note-title>"` or `/kcheck` before modifying this file.
    ```

    Placement: immediately after the module docstring or file header, before imports.

    For non-Python files:
    - PowerShell: `# VAULT: ...`
    - Markdown: `<!-- VAULT: ... -->`
    - JSON/config: skip (no comment syntax or use `"_vault_refs"` key)

    Known candidates [ASSUMED — implementor should verify by vault search]:
    - `src/usv_spectrogram/app/core/saved_detection_tracker.py` → `saved-previous ghost detections...three aligned detection state tiers`
    - `src/usv_spectrogram/app/core/detection_logic.py` → ghost detection state tiers
    - `src/usv_spectrogram/app/widgets/spectrogram_view.py` → detection overlay rendering constraints
    - `src/usv_spectrogram/classification/deepsqueak_import.py` → `DeepSqueak import...exact subdirectory name matches`
    - `src/usv_spectrogram/classification/raven_export.py` → export format constraints
    - `src/usv_spectrogram/detection/energy_detector.py` → STFT parameter constraints (ADR-001, ADR-002)

    Risk classification:
    - **HIGH** (canary + mandatory `/kcheck`): Files that caused regressions — detection app state, export adapters
    - **MEDIUM** (canary only): Files with non-obvious design decisions — energy detector, config files
    - **LOW** (no canary): Standalone utilities, tests, new files

    **Critical constraint: err on the side of FEWER canaries.** Five well-placed canaries that always get read beat fifty that get ignored.

2. `ops/vault-canary-map.md` (NEW) — Registry of all canary-annotated files

    ```markdown
    # Vault Canary Map
    <!-- Last updated: YYYY-MM-DD -->
    Files with VAULT comments and their referenced notes.
    Audit periodically: are canaries pointing to current notes?

    | File | Risk | Referenced Notes |
    |------|------|-----------------|
    | src/usv_spectrogram/app/core/saved_detection_tracker.py | HIGH | ghost-detection-state-tiers |
    | src/usv_spectrogram/classification/deepsqueak_import.py | HIGH | deepsqueak-subdirectory-naming |
    | src/usv_spectrogram/detection/energy_detector.py | MEDIUM | STFT-parameter-constraints |
    ```

3. `CLAUDE.md` (EDIT) — Add vault canary comment convention

    Add to coding conventions section:
    ```markdown
    ## Vault Canary Comments
    Source files with `# VAULT:` comments reference knowledge vault notes that contain
    constraints or architectural decisions relevant to that file. Before making non-trivial
    modifications to these files, run `qmd get "<note-title>"` for each referenced note.
    For HIGH-risk files, also run `/kcheck` with your intended changes.
    Do not remove or modify VAULT comments without updating the corresponding vault notes.
    See `ops/vault-canary-map.md` for the full registry.
    ```

**Test plan:**
    ```
    1. Verify each annotated file still parses correctly (py_compile for .py files)
    2. Verify canary note titles match actual vault note titles (qmd get succeeds for each)
    3. Verify ops/vault-canary-map.md matches actual canary comments in source files
    4. Verify at least 3 HIGH-risk and 2 MEDIUM-risk files are annotated
    5. Verify CLAUDE.md includes the canary convention section
    ```

**Exit criteria:**
- [ ] At least 3 high-risk source files have `# VAULT:` canary comments
- [ ] All referenced note titles resolve (qmd get succeeds)
- [ ] `ops/vault-canary-map.md` exists with risk levels and is accurate
- [ ] Annotated Python files pass py_compile
- [ ] CLAUDE.md includes vault canary comment convention
- [ ] No more than ~10 total canary files (signal > noise)

---

### 15.4 Knowledge-Enriched Codex Handoffs

**What:** Integrate vault search into the Codex task specification workflow so that handoff documents include relevant architectural constraints from the knowledge graph.
**Status:** READY
**Review Tier:** 1 (procedure/template changes)
**Depends on:** None (but benefits from 15.1 /kcheck pattern being established)

/implement Knowledge-Enriched Codex Handoffs

When generating a Codex task spec, automatically search the vault for constraints relevant to the task and include them in the spec. Codex runs without vault access, so constraints must be flattened into plain text in the handoff document. This ensures Codex respects architectural invariants it can't discover on its own.

**Context:** Codex handoff protocol is documented in `docs/handoffs/README.md` and `CLAUDE.md`. Codex task specs go to `docs/handoffs/YYYY-MM-DD_task-name.md`. Currently, vault knowledge is not included in these specs — Codex operates without knowledge of architectural constraints unless Claude Code manually includes them. The Codex ownership boundary is defined in `AGENTS.md`. [ASSUMED: No existing `/handoff` or `/codex` skill exists — this is implemented as a CLAUDE.md procedure enhancement and a template addition.]

**Files to create:**

1. `templates/codex-handoff.md` (NEW) — Codex task spec template with vault section

    ```markdown
    # Codex Task: <task-name>
    <!-- From: Claude Code | To: Codex | Date: YYYY-MM-DD -->

    ## Task
    <imperative description of what to build/fix>

    ## Files to Modify
    - `path/to/file.py` — <what changes>

    ## Relevant Constraints (from vault)
    <!-- Auto-populated by vault search before writing handoff. -->
    <!-- Codex: treat these as hard constraints. -->
    - <constraint description from note>
      Source: <note title> (verified YYYY-MM-DD)
    - <constraint description from note>
      Source: <note title> (verified YYYY-MM-DD)

    ## Test Plan
    - <specific verification steps>

    ## Acceptance Criteria
    - [ ] <verifiable criterion>
    ```

    Cap at 5 constraints per handoff. Flatten vault knowledge into plain text — Codex doesn't need to search, it just needs the constraint statements. Include verification date to signal freshness.

2. `CLAUDE.md` (EDIT) — Add vault search step to Codex handoff procedure

    Find the section about Codex task routing / handoff protocol. Add:
    ```markdown
    ## Codex Handoff Vault Search
    Before writing a Codex task spec, run:
    1. `qmd deep_search "<task description summary>"` or `/kcheck "<task description>"`
    2. From results, extract notes that describe constraints on files/systems the task will modify
    3. Include up to 5 constraints in the "Relevant Constraints" section of the handoff
    4. Use the template at `templates/codex-handoff.md`
    ```

3. `docs/handoffs/README.md` (EDIT) — Reference the new template

    Add a note about the template and the vault search step to the existing handoff documentation.

**Test plan:**
    ```
    1. Create a test handoff for a known constrained system (e.g., detection app modification)
    2. Verify the vault search returns relevant constraint notes
    3. Verify the handoff template renders correctly with 2-3 constraints populated
    4. Verify constraint descriptions are self-contained (Codex can understand without vault access)
    5. Verify CLAUDE.md includes the Codex handoff vault search procedure
    ```

**Exit criteria:**
- [ ] `templates/codex-handoff.md` exists with "Relevant Constraints" section
- [ ] CLAUDE.md includes vault search step for Codex handoffs
- [ ] `docs/handoffs/README.md` references the template
- [ ] A sample handoff demonstrates the constraint format works

---

## Phase 15 Gate

- [ ] `/kcheck` skill exists and runs successfully against test descriptions
- [ ] Orient hook generates `ops/session-relevance.md` with filtered, relevant results
- [ ] At least 3 high-risk source files have VAULT canary comments
- [ ] Codex handoff template includes "Relevant Constraints" section
- [ ] All CLAUDE.md changes committed (orient, canaries, /kcheck, Codex handoffs)
- [ ] `ops/vault-canary-map.md` exists and is accurate
- [ ] Token budgets verified (orient < 3,500 tokens, /kcheck < 1,500 per invocation)

## Post-Gate: 1-Week Evaluation

After running for ~10 sessions, assess activation effectiveness:

**Activation metrics:**
- Did the agent load notes from session-relevance that it wouldn't have found otherwise?
- Did any canary comment prevent a regression?
- How many times was `/kcheck` invoked? Was it skipped when it should have been used?
- Did Codex task specs with constraints produce fewer regressions than without?

**Relevance quality (CRAG-inspired):**
- What % of surfaced notes did the agent actually load fully? (Low % = threshold too loose)
- Which notes were surfaced but never loaded? (Candidates for better descriptions)
- Which regressions occurred that vault notes *could* have prevented? (Gap analysis)

**Token budget:**
- Is the session-relevance.md budget holding at < 3,500 tokens?
- Average /kcheck cost per session?
- Total activation overhead as % of context window?

Based on results, decide whether to extend with:
- PreToolUse hook warnings for high-risk directories
- Skill-level pre-retrieval gates on specific workflows
- Usage tracking log for retrieval quality improvement
