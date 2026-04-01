# Plan: arscontexta Health Restoration & Operational Fixes

> **Context:** Two arscontexta audits (2026-03-20) revealed that the system architecture is sound but operational health has degraded, especially under throughput pressure in mickey-lab. This plan addresses the actual problems found, prioritized by impact.  
> **Repos:** mickey-lab (`/mnt/d/mickey_london_lab`) and Cloudy-Claude (`/mnt/d/we_do_this/Cloudy-Claude`)  
> **Approach:** Fix what's broken before adding new tools. The architecture works — the operations need attention.  
> **Note:** Dangling links in mickey-lab were confirmed resolved (0) on 2026-03-20 health check. That fix and related cross-space link resolution work have been removed from this plan.

---

## Priority 1: mickey-lab — Degradation Under Load

mickey-lab's vault grew 4.5× (117 → 525 notes) and quality metrics degraded during bulk ingestion. The link graph is healthy (confirmed 0 dangling links), but schema, descriptions, filenames, and reference docs need repair.

### 1.1 Restore Schema Compliance (91.8% → 100%)

**Problem:** 39 notes missing `topics:` field in YAML frontmatter, caused by bulk ingestion.

**Plan:**
1. List all non-compliant notes: scan `notes/` for files where YAML frontmatter lacks `topics:` field
2. For each note, determine the appropriate topic(s) from the note's content and its wiki-link neighbors
3. Add `topics:` field to each note's frontmatter
4. Run `/validate` across all notes to confirm 100% compliance

**Acceptance:** Schema compliance back to 100%.

**Process improvement:** Investigate why `/reduce` didn't enforce the `topics:` field during bulk ingestion. The `validate-note.ps1` hook fires PostToolUse on Write — did it fire and warn? Was the warning ignored? If the hook warned but the agent continued, consider making the hook blocking (fail the write) rather than advisory. (See Priority 3.2.)

### 1.2 Windows Filename Sanitization

**Problem:** Colons in note titles create zero-byte files on Windows, causing unresolvable references. 2 known instances.

**Plan:**
1. Find the 2 zero-byte files in `notes/`
2. Recreate them with sanitized filenames (replace colons with em-dashes or hyphens)
3. Update all inbound wiki links that reference the old filenames
4. Add a sanitization step to `/reduce` (and any other skill that creates notes) that strips or replaces characters invalid in Windows paths: `: * ? " < > |`

**Acceptance:** No zero-byte files in `notes/`. `/reduce` rejects or sanitizes filenames with invalid characters going forward.

### 1.3 Fix Description Quality Drift

**Problem:** `/reduce` produced weak descriptions (title restatements) during bulk ingestion. A self-check gate was added 2026-03-07 but similar gaps may exist in other skills.

**Plan:**
1. Sample 20 notes from the bulk ingestion batch (those with lowest `confidence` or newest creation dates)
2. Evaluate each description: does it add information beyond the title? Is it a genuine atomic claim summary?
3. For notes with weak descriptions, rewrite the `description:` field
4. Audit other skills that produce description fields (`/reflect`, `/reweave`) — do they have equivalent quality gates?
5. If gaps found, add description quality checks to those skills matching the gate added to `/reduce` on 2026-03-07

**Acceptance:** Sampled descriptions meet quality standard. Quality gates documented for all description-producing skills.

### 1.4 Reference Document Compliance (28.6% → 100%)

**Problem:** Only 4 of 14 non-exempt reference files meet the PRD template (missing Purpose, Derivation Questions, Curated Claims sections).

**Plan:**
1. List the 10 non-compliant reference files
2. For each, add the missing sections (Purpose, Derivation Questions, Curated Claims) based on the file's content and role in the system
3. This is a content authoring task, not automation — each reference doc serves a specific purpose in the skill graph and needs human judgment about what its derivation questions and curated claims should be

**Acceptance:** All 14 non-exempt reference files pass PRD template compliance.

---

## Priority 2: Cloudy-Claude — Activate Unused Features

Cloudy-Claude's problems are the opposite of mickey-lab's: the system has features that exist but have never been used. The vault is small (143 notes) and structurally healthy, but the operational learning loop is dead.

### 2.1 Mine the 185 Session Logs

**Problem:** 185 session JSON files in `ops/sessions/`, never analyzed. The `/remember --mine-sessions` threshold has been exceeded 37x.

**Plan:**
1. Run `/remember --mine-sessions` (or the equivalent command) to process the accumulated session logs
2. Review what gets extracted — are there patterns about frequently needed context, repeated failures, or recurring decision points?
3. If the extraction is low quality, manually review the 10 most recent sessions and extract key observations/decisions as notes

**Acceptance:** Session backlog processed. Any extracted knowledge integrated into `ops/observations/` or converted to notes.

### 2.2 Bootstrap the Observation/Tension System

**Problem:** 0 observations, 0 tensions ever recorded. The operational learning loop exists in CLAUDE.md but has never been exercised.

**Plan:**
1. Seed the system with 3-5 observations from what we already know:
   - "Context relevance is goal-driven — misses task-specific context when user's question diverges from goal threads"
   - "Session logs are write-only — valuable patterns are never extracted"
   - "Maintenance triggers fire but agent rarely acts autonomously"
   - "Priority/SAP notes are stale but still discoverable, creating confusion"
   - (add more based on what session mining in 2.1 reveals)
2. Run `/rethink` on these observations to see if the system can generate useful insights
3. If the system works, add a reminder to invoke `/remember` at the end of sessions where friction occurred

**Acceptance:** `ops/observations/` has ≥3 entries. `/rethink` has been invoked at least once.

### 2.3 Fix Stale Notes

**Problem:** `notes/priority-likely-erp-vendor.md` and `notes/priority-meeting-key-integration-opportunity.md` reference Priority/SAP, which is explicitly "1+ year away."

**Plan:**
1. Mark both notes with `meta_state: outdated` in their YAML frontmatter
2. Add a note to each explaining current state: Madaf is the current ERP, Priority integration is deferred
3. Search for any other notes referencing Priority/SAP that should also be marked

**Acceptance:** All Priority/SAP notes marked `meta_state: outdated`. No confusion possible.

### 2.4 Fix Inbox Counting Bug

**Problem:** Session-orient hook reported "Inbox has 5 items" but actual count is 1 active file + 1 archive directory. Hook counts `.md` files recursively including `inbox/archive/`.

**Plan:**
1. Edit `session-orient.ps1` to exclude `archive/` subdirectory from inbox count
2. Use non-recursive file listing or add an explicit `-Exclude` filter for the `archive` directory

**Acceptance:** Hook reports accurate inbox count.

---

## Priority 3: Shared Improvements (Both Repos)

These improvements apply to the shared arscontexta infrastructure and benefit both repos.

### 3.1 Inbox Archive Step in /reduce

**Problem (mickey-lab):** `/reduce` has no archive step — processed files linger in inbox.

**Plan:**
1. Edit the `/reduce` skill to add an archive step after successful processing: move processed inbox file to `inbox/archive/`
2. Update queue.json entry to reflect the archive
3. Ensure any wiki links created during reduction point to the notes, not the inbox source

**Acceptance:** After `/reduce` completes, processed inbox files are in `inbox/archive/`, not lingering in `inbox/`.

### 3.2 Make validate-note Blocking (Not Advisory)

**Problem (mickey-lab):** Schema compliance degraded during bulk ingestion because `validate-note.ps1` warns but doesn't block writes. This directly enabled Priority 1.1's problem.

**Plan:**
1. Evaluate whether making the hook blocking (exit code 1 → Claude Code rejects the write) would break any legitimate workflows
2. If safe: update `validate-note.ps1` to return exit code 1 on validation failure for required fields (`description`, `type`, `topics`), forcing the agent to fix the note before it's accepted
3. If not safe (some skills need to write partial notes as intermediate state): add a severity level — block on missing required fields, warn on missing optional fields

**Acceptance:** Writing a note without required YAML fields to `notes/` fails at the hook level. This prevents future schema drift during bulk ingestion.

### 3.3 Lower the /rethink Observation Threshold

**Problem (mickey-lab):** Threshold for `/rethink` trigger is 10 observations. Patterns can linger below threshold for weeks before action.

**Plan:**
1. Change threshold from 10 to 7 (as already recommended by Phase 5.2 review but not yet implemented)
2. Update the threshold in `ops/config.yaml` or wherever it's defined
3. Consider adding a time-based trigger: if any observation is older than 14 days without review, trigger `/rethink` regardless of count

**Acceptance:** Threshold updated. Time-based trigger added if feasible.

---

## Priority 4: Investigate (Don't Implement Yet) — COMPLETE (2026-03-22)

Full investigation reports: `docs/investigations/P4-investigation-handoff.md` and individual 4.x files.

### 4.1 Task-Aware Context Surfacing — COMPLETE

**Root cause found:** qmd's npm shim is broken on Windows (`/bin/sh.exe` not found). Knowledge activation has **never worked**. The "No strong matches" output was a silent execution failure, not a retrieval quality issue. Secondary: BM25 query dilution from status words in goal-thread prose.

**Fix proposed (4 changes):** Use `node` directly instead of broken shim, strip status words from vsearch queries, preserve periods in title extraction, handle stderr in JSON parsing. See `docs/investigations/4.1-task-aware-context-surfacing.md`.

### 4.2 Session Pattern Mining — DEFERRED

**Finding:** Only 13 of 347 session transcripts survive (~3-day retention). Batch mining is architecturally wrong — data is GC'd before it can be mined. Capture-time extraction (Option C hybrid) is the right architecture but premature given that knowledge activation never worked. See `docs/investigations/4.2-session-pattern-mining.md`.

### 4.3 qmd Retrieval Precision — COMPLETE

**9 findings** from 20+ query-mode combinations:
- Lex P@1 = 90% for domain queries (3-5 keyword terms), 0% for diluted goal-thread queries
- Vec search has two reliability issues: MCP disconnects, no CLI fallback (model download required)
- Uniform 0.93 scores make threshold filtering useless
- Self-retrieval fails for 40% of notes (abstract descriptions)
- BM25 dilution confirmed: full prose → 0 results, condensed → 5 hits

**7 ranked recommendations** in `docs/investigations/4.3-qmd-retrieval-precision-benchmark.md`.

### 4.4 OpenViking Re-evaluation — CLOSED (deferred)

No triggers fired. qmd retrieval is sufficient with query construction fixes. See `docs/investigations/4.4-openviking-decision.md`.

---

## Execution Order

```
Week 1: 1.1 Schema compliance (mickey-lab, 39 notes)
         1.2 Filename sanitization (mickey-lab, 2 files + /reduce fix)
         2.4 Inbox count bug (Cloudy-Claude, 5 min fix)
         3.1 /reduce archive step (shared)

Week 2: 1.3 Description quality audit + fixes (mickey-lab)
         2.1 Mine 185 session logs (Cloudy-Claude)
         2.2 Bootstrap observations (Cloudy-Claude)
         3.2 Blocking validation hook (shared)
         3.3 Lower /rethink threshold (shared)

Week 3: 1.4 Reference doc compliance (mickey-lab)
         2.3 Mark stale notes (Cloudy-Claude)
         4.1-4.3 Investigations (both repos)
         4.4 OpenViking re-evaluation decision
```

---

## Success Metrics

| Metric | Before | Target | Repo |
|--------|--------|--------|------|
| Schema compliance | 91.8% | 100% | mickey-lab |
| Zero-byte files | 2 | 0 | mickey-lab |
| Reference doc compliance | 28.6% | 100% | mickey-lab |
| Description quality (sample) | Unknown | All add info beyond title | mickey-lab |
| Session logs processed | 0 of 185 | 185 of 185 | Cloudy-Claude |
| Observations recorded | 0 | ≥3 | Cloudy-Claude |
| Stale notes marked | 0 | All Priority/SAP notes | Cloudy-Claude |
| Inbox count accuracy | Overcounts | Accurate | Cloudy-Claude |
| /reduce auto-archives | No | Yes | Both |
| validate-note blocking | Advisory | Blocking on required fields | Both |
| /rethink threshold | 10 | 7 + time-based | Both |

---

## Notes for Claude Code

- **This is a maintenance plan, not a feature plan.** The goal is to restore both vaults to their design-level quality, not to add new capabilities.
- **mickey-lab is the higher priority** because it has active degradation (schema drift, zero-byte files, weak descriptions). Cloudy-Claude is healthy but underutilized.
- **The shared improvements (Priority 3) should be implemented once and propagated to both repos.** Since both repos share the same arscontexta version, changes to skills and hooks should apply to both.
- **Track the success metrics table.** Run `/health` or `/stats` before starting and after each priority block to measure progress.
- **The CRLF issue (from Cloudy-Claude audit) affects scripting.** If any of these fixes involve bash scripts processing vault files, use `dos2unix` or handle `\r\n` explicitly. PowerShell scripts handle CRLF natively.
- **Don't rewrite CLAUDE.md or AGENTS.md.** The audits confirmed these are deeply integrated with arscontexta. The fixes in this plan work within the existing system, not around it.
- **Priority 4 items are investigations, not implementations.** Write analysis documents with recommendations. Do not implement changes without review.
