# Plan: Knowledge Activation Architecture — Phase 1

## Objective

Close the knowledge activation gap: vault notes (505) are high-quality but only surfaced when the agent voluntarily searches. This plan implements three changes that require no new infrastructure and address activation at session start, mid-session editing, and cross-agent handoff.

## Research Context (RAG Literature)

This architecture is informed by four established patterns from the RAG research literature:

- **Self-RAG** (Asai et al., 2023 — ICLR 2024): Trains models to emit "reflection tokens" that decide whether retrieval is needed before generating. Our procedural gates (Workstreams 2 & 4) are the Claude Code equivalent — since we can't fine-tune the model, we implement the retrieve/no-retrieve decision as explicit skill steps.
- **FLARE** (Jiang et al., 2023 — EMNLP): Uses the model's *draft output* as a retrieval query, retrieving only when confidence is low. Inspires Workstream 4 (`/kcheck`): the agent describes what it's about to modify, and that intent becomes the search query.
- **CRAG** (Yan et al., 2024): Adds a lightweight retrieval evaluator that classifies results as Correct/Incorrect/Ambiguous before passing them to generation. Inspires the relevance thresholds in Workstreams 1 and 2 — don't surface noise, or the agent learns to ignore gates.
- **Adaptive RAG** (Jeong et al., 2024): Routes queries to different retrieval depths based on predicted complexity. Inspires the risk classification: modifications to existing constrained systems get deep search, new standalone code gets shallow or no search.

The key lesson across all four: **the "when to retrieve" decision matters as much as retrieval quality itself.** Our vault has good retrieval infrastructure (qmd, topic maps, dense links). The gap is entirely in activation triggers.

## Scope

Four workstreams, implementable across 1-2 sessions:

1. **Goal-Aware Orient Hook** — automatic qmd search at session start (baseline RAG)
2. **Canary Comments** — vault note references embedded in high-risk source files (static activation)
3. **Knowledge-Enriched Codex Handoff** — vault search integrated into task spec generation
4. **`/kcheck` Micro-Skill** — FLARE-inspired intent-based mid-session retrieval

---

## Workstream 1: Goal-Aware Orient Hook

### What Changes

Enhance `.claude/hooks/session-orient.ps1` to:
1. Parse `ops/goals.md` and extract active thread titles + descriptions
2. Run qmd searches against each active thread
3. Write results to `ops/session-relevance.md`
4. Add instruction in `CLAUDE.md` orient procedure to read that file

### Implementation Steps

#### Step 1: Read current orient hook

```
Read .claude/hooks/session-orient.ps1
Read ops/goals.md (to understand the format of active threads)
Read CLAUDE.md (to find the orient procedure section)
```

#### Step 2: Add qmd search logic to orient hook

In `session-orient.ps1`, after the existing goals.md read:

```powershell
# --- Knowledge Activation ---
# Extract active thread titles from goals.md
# For each thread:
#   1. Run: qmd search "<thread_title>" --limit 4
#   2. Run: qmd vector_search "<thread_description>" --limit 4
#   3. Collect note titles + descriptions from results
#   4. Deduplicate by note title

# Write results to ops/session-relevance.md
```

**Format for `ops/session-relevance.md`:**

```markdown
# Session Relevance Brief (auto-generated)
<!-- Generated at: <timestamp> -->
<!-- Source threads: <count> active threads from goals.md -->

## Thread: <thread_title>
- [<note-title>] — <note description, first sentence only>
- [<note-title>] — <note description, first sentence only>
> See also: topic-maps/<relevant-map>.md

## Thread: <thread_title>
...
```

**Constraints:**
- Cap at 5 active threads (skip threads marked done/paused)
- Cap at 4 results per thread (2 keyword + 2 vector, deduplicated)
- **Relevance threshold (CRAG-inspired):** Only include results above qmd's relevance score cutoff. If a thread returns zero results above threshold, note "no strong matches" rather than showing weak results. Noisy briefs teach the agent to skip reading the file.
- Total file should stay under 3,500 tokens (~200 tokens per thread × 5 threads + overhead)
- Use CLI (`qmd search "query"`) not MCP — hook is PowerShell, no conversation context needed
- Include one topic map pointer per thread if a relevant map exists

#### Step 3: Parse goals.md thread extraction

Inspect `ops/goals.md` to determine:
- How are active threads delimited? (headings? list items? YAML frontmatter?)
- What constitutes "active" vs "done/paused"?
- Where is the title vs description boundary?

Write a parsing function that extracts `{ title: string, description: string }[]` from goals.md. Keep parsing simple — regex on markdown structure, not a full parser.

#### Step 4: Wire into CLAUDE.md orient procedure

Find the orient/session-start section in CLAUDE.md. Add:

```
- Read ops/session-relevance.md (auto-generated vault relevance brief)
- If any listed notes are directly relevant to the current task, load them with qmd read
```

Place this AFTER the goals.md read (so the agent has task context) and BEFORE starting work.

#### Step 5: Test

- Run the hook manually: `powershell .claude/hooks/session-orient.ps1`
- Verify `ops/session-relevance.md` is generated with reasonable results
- Verify token count stays under 3,500
- Check that qmd queries return meaningful results (not noise)

### Edge Cases

- If goals.md has no active threads → write a brief noting "no active threads, skipping relevance search"
- If qmd is unavailable or errors → write a brief noting the failure, don't block the hook
- If a thread title is very short (e.g., "cleanup") → use description for vector search, skip keyword search for that thread

---

## Workstream 2: Canary Comments in High-Risk Source Files

### What Changes

Add standardized `# VAULT:` comments at the top of source files that have known vault constraints. When the agent opens these files to edit, it sees the note titles inline and can choose to `qmd read` them.

### Implementation Steps

#### Step 1: Identify high-risk files

Search the vault for notes that describe:
- Architectural invariants (`type: decision` or similar)
- Bug fixes tied to specific files
- Design constraints that caused regressions

Run:
```
qmd search "architectural invariant"
qmd search "bug fix constraint"
qmd search "design decision"
```

Cross-reference results with actual source file paths to build the mapping.

**Known candidates from the briefing:**
- Detection app files → `saved-previous ghost detections form three aligned detection state tiers`
- Export/import adapter files → `DeepSqueak import required exact subdirectory name matches`

#### Step 2: Define the canary comment format

```python
# VAULT: <note-title-1>, <note-title-2>
# Run `qmd read "<note-title>"` before modifying this file.
```

Place immediately after the module docstring or file header, before imports.

For non-Python files, adapt the comment syntax:
- PowerShell: `# VAULT: ...`
- JSON/config: add a `"_vault_refs"` key if the format supports it, otherwise skip
- Markdown: `<!-- VAULT: ... -->`

#### Step 3: Insert canary comments

For each identified file:
1. Open the file
2. Add the canary comment block after the header
3. Verify the file still runs/parses correctly

#### Step 4: Add convention to CLAUDE.md

In the coding conventions section of CLAUDE.md, add:

```
## Vault Canary Comments
Source files with `# VAULT:` comments reference knowledge vault notes that contain
constraints or architectural decisions relevant to that file. Before making non-trivial
modifications to these files, run `qmd read "<note-title>"` for each referenced note.
Do not remove or modify VAULT comments without updating the corresponding vault notes.
```

#### Step 5: Document the mapping

Create `ops/vault-canary-map.md`:

```markdown
# Vault Canary Map
Files with VAULT comments and their referenced notes.
Risk level: HIGH = architectural invariants, BUG = caused regressions, CONSTRAINT = design decisions.

| File | Risk | Referenced Notes |
|------|------|-----------------|
| src/detection/... | HIGH | ghost-detection-state-tiers, ... |
| src/export/... | BUG | deepsqueak-subdirectory-naming, ... |
```

This allows periodic auditing: are the canaries still pointing to current notes?

### Risk Classification (Adaptive RAG Pattern)

Not all files need canaries. Apply the Adaptive RAG principle — route retrieval depth by predicted risk:
- **HIGH RISK (always canary + always `/kcheck`):** Files that have caused regressions before, or that implement architectural invariants. These are your detection app and export adapters.
- **MEDIUM RISK (canary only):** Files with design decisions that aren't fragile but are non-obvious. The canary comment is a sufficient nudge.
- **LOW RISK (no canary):** Standalone utilities, tests, config files. Adding canaries here creates noise that dilutes the signal from high-risk files.

Err on the side of fewer canaries. Five well-placed canaries that always get read beat fifty that get ignored.

---

## Workstream 4: `/kcheck` Micro-Skill (FLARE-Inspired)

### Research Basis

FLARE (Forward-Looking Active REtrieval) retrieves information mid-generation by using the model's *intended next output* as a search query. In Claude Code's context, the equivalent is: before modifying a system, the agent describes what it's about to do, and that description drives a vault search. This is the key mid-session activation mechanism missing from the original plan.

### What Changes

Create a lightweight `/kcheck` skill that:
1. Takes a brief description of intended work as input
2. Runs qmd search against the vault
3. Shows relevant note titles + descriptions
4. Lets the agent decide what to load fully

### Skill Definition

```yaml
name: kcheck
description: "Knowledge check — search vault for constraints relevant to planned work"
trigger: "Before modifying detection, export, labeling, or other constrained systems"
```

### Skill Procedure

```
/kcheck <what I'm about to do>

1. Extract key nouns/concepts from the input description
2. Run qmd vector_search with the full description (semantic match)
3. Run qmd search with extracted keywords (exact match)
4. Deduplicate results
5. Apply relevance threshold — only show results with score above cutoff
6. Display: note title + first sentence of description for top 5-8 results
7. If any results reference files the agent is about to modify → flag as CRITICAL
8. Agent decides which notes to load fully with qmd read
```

### CLAUDE.md Integration

Add to the task procedures section:

```
## Mid-Session Knowledge Checks
Before modifying files in high-risk directories (detection app, export adapters,
labeling pipeline), run `/kcheck "<brief description of planned changes>"`.
This is mandatory for HIGH-risk canary files and recommended for any non-trivial
modification to existing systems.

Skip /kcheck for: new standalone files, test files, documentation-only changes.
```

### Why This Works Better Than Canaries Alone

Canary comments activate when the agent is *already looking at the file*. `/kcheck` activates earlier — when the agent is *planning* the modification. This catches constraints that live in notes about *related* systems, not just the specific file being edited. For example, modifying the detection app might require knowledge from a note about the labeling pipeline's assumptions, which wouldn't be in the detection file's canary.

### Token Cost

~1,000-1,500 tokens per invocation (8 note descriptions × ~150 tokens each). Comparable to reading a short file.

### Implementation Notes

- The skill should be a simple wrapper around qmd — no complex logic
- Consider making it a shared utility function that Workstream 2 canary gates can also call
- The relevance threshold should match Workstream 1's threshold for consistency
- Log which notes `/kcheck` surfaces vs. which the agent actually loads — this is the usage tracking data that helps identify poorly-described notes (per CRAG's evaluator pattern)

---

## Workstream 3: Knowledge-Enriched Codex Handoffs

### What Changes

When generating a Codex task spec, automatically search the vault for constraints relevant to the task and include them in the spec.

### Implementation Steps

#### Step 1: Find the Codex handoff template/skill

Locate where Codex task specs are generated. Check:
- Any `/handoff` or `/codex` skill definitions
- CLAUDE.md sections about Codex task routing
- `docs/codex_index.md` structure

#### Step 2: Add vault search to handoff generation

Before writing the task spec, run:
```
qmd deep_search "<task description summary>"
```

From results, extract notes that describe constraints on the files/systems the task will modify.

#### Step 3: Add "Relevant Constraints" section to task spec template

```markdown
## Relevant Constraints (from vault)
<!-- Auto-populated by vault search. Codex: treat these as hard constraints. -->
- <constraint description from note>
  Source: <note title> (verified <date>)
- <constraint description from note>
  Source: <note title> (verified <date>)
```

Cap at 5 constraints. Flatten vault knowledge into plain text — Codex doesn't need to search, it just needs the constraint statements.

#### Step 4: Consider periodic codex_index.md refresh

If time permits, create a script that regenerates `docs/codex_index.md` from topic maps filtered to implementation-relevant notes. Run manually or weekly. This gives Codex a static knowledge snapshot beyond per-task constraints.

---

## CLAUDE.md Changes Summary

All CLAUDE.md modifications in one place for review:

1. **Orient procedure**: Add `Read ops/session-relevance.md` step
2. **Coding conventions**: Add Vault Canary Comments section
3. **Codex handoff procedure**: Add vault search step before spec generation
4. **Task procedures**: Add `/kcheck` mandatory use for high-risk modifications
5. **Skills directory**: Add `/kcheck` skill definition

---

## Validation Criteria

After implementation, verify:

- [ ] `session-orient.ps1` generates `ops/session-relevance.md` with relevant results
- [ ] Relevance threshold filters out weak matches (no "noise" entries in brief)
- [ ] Token count of session-relevance.md stays under 3,500
- [ ] At least 3 high-risk source files have VAULT canary comments with risk classification
- [ ] CLAUDE.md orient procedure includes session-relevance.md read
- [ ] CLAUDE.md includes vault canary comment convention
- [ ] `/kcheck` skill exists and runs successfully against a test description
- [ ] `/kcheck` correctly flags CRITICAL when results reference files being modified
- [ ] Codex task spec template includes "Relevant Constraints" section
- [ ] `ops/vault-canary-map.md` exists with risk levels and is accurate

## Post-Implementation Evaluation (1 week)

After running for ~10 sessions, assess:

**Activation effectiveness:**
- Did the agent load any notes from the session-relevance brief that it wouldn't have found otherwise?
- Did any canary comment prevent a regression?
- How many times was `/kcheck` invoked? Was it skipped when it should have been used?
- Are the qmd queries returning useful results or noise?

**Relevance quality (CRAG-inspired diagnostic):**
- What % of surfaced notes did the agent actually load fully? (Low % = threshold too loose)
- Which notes were surfaced but never loaded? (Candidates for better descriptions)
- Which regressions occurred that vault notes *could* have prevented? (Gap analysis)

**Token budget:**
- Is the session-relevance.md token budget holding?
- What's the average `/kcheck` cost per session?
- Total activation overhead as % of context window?

Based on results, decide whether to add:
- PreToolUse hook warnings for high-risk directories (Option B from discussion)
- Skill-level pre-retrieval gates (Layer 2 from briefing) on specific workflows
- Usage tracking log for retrieval quality improvement


## Research References

Papers that informed this architecture:

- **Self-RAG**: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (ICLR 2024). → Adaptive on-demand retrieval via reflection tokens. Our procedural gates are the prompt-engineering equivalent.
- **FLARE**: Jiang et al., "Active Retrieval Augmented Generation" (EMNLP 2023). → Forward-looking retrieval triggered by generation uncertainty. Inspires the `/kcheck` intent-based search pattern.
- **CRAG**: Yan et al., "Corrective Retrieval Augmented Generation" (2024). → Lightweight retrieval evaluator with Correct/Incorrect/Ambiguous classification. Inspires relevance thresholds and the post-implementation quality diagnostics.
- **Adaptive RAG**: Jeong et al., "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity" (2024). → Query routing by complexity. Inspires risk-based classification of which files/operations need retrieval.
- **Agentic RAG Survey**: Singh et al., "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG" (arXiv 2501.09136, 2025). → Comprehensive taxonomy of agent-in-the-loop RAG patterns.
