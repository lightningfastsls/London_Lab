# Infrastructure Improvements — Implementation Roadmap

> **Source:** `PLAN_skills-integration.md` (Feb 27 2026), refined through architectural review.
> **Scope:** 2 modules — command disambiguation + tracking document consolidation.
> **Context:** The original plan had 4 improvements. Playwright visual verification was rejected
> (heavy dependency, fragile selectors, false confidence for a dev tool). MCP builder patterns
> were deferred to the Cloudy Claude project. The staleness-audit step was superseded by the
> deeper fix: consolidating tracking documents so status lives in ONE canonical place.

---

## Phase 1: Command Disambiguation

### 1.1 Rename /verify to /verify-code

**What:** Resolve the command name collision between code verification (`/verify` in `.claude/commands/`) and note quality verification (`/verify` in `.claude/skills/`). Rename the code command to `/verify-code`.
**Status:** READY
**Review Tier:** 1 (file rename + reference updates)
**Depends on:** None

/implement Rename /verify to /verify-code

Rename the code verification command from `/verify` to `/verify-code` to resolve the dispatch
collision with the arscontexta `/verify` skill (note quality verification).

**Context:** Two `/verify` commands exist — `.claude/commands/verify.md` (py_compile + pytest + flake8)
and `.claude/skills/verify/SKILL.md` (recite + validate + review for notes). Context-dependent
dispatch usually works but creates ambiguity. The arscontexta skill is harder to rename (plugin-managed),
so we rename the code command instead.

**Files to modify:**

1. `.claude/commands/verify.md` (RENAME to `.claude/commands/verify-code.md`)
   - Rename the file. No content changes needed — the file's contents are already correct.
   - After rename, the command becomes `/verify-code`.

2. `.claude/commands/implement.md` (EDIT)
   - Search for any `/verify` references in the context of code verification
   - Replace with `/verify-code`
   - Do NOT change references to the arscontexta `/verify` skill

3. `.claude/commands/verify-quick.md` (REVIEW)
   - Check if it references `/verify` as the "full version"
   - If so, update to `/verify-code`

4. `CLAUDE.md` (EDIT)
   - Search for `/verify` references in Quick Commands or workflow sections
   - Update code-context references to `/verify-code`
   - Do NOT change the KG pipeline reference: `/seed -> /reduce -> /reflect -> /reweave -> /verify`
     (that's the arscontexta skill, which keeps its name)

5. **Full sweep**: Grep all `.claude/` files for bare `/verify` and update code-context ones.

**Naming after change:**

| Command | Purpose | Source |
|---------|---------|--------|
| `/verify-code` | py_compile + pytest + flake8 | `.claude/commands/verify-code.md` |
| `/verify-quick` | py_compile on modified files + pytest | `.claude/commands/verify-quick.md` |
| `/verify` | Recite + validate + review (notes) | `.claude/skills/verify/SKILL.md` (arscontexta) |
| `/validate` | Schema validation only (notes) | `.claude/skills/validate/SKILL.md` (arscontexta) |

**Test plan:**
```
1. After rename, confirm `/verify-code` dispatches to the code verification command
2. Confirm `/verify` dispatches to the arscontexta note quality skill (not the old code command)
3. Grep all .claude/ files — no remaining `/verify` references that mean code verification
4. Grep CLAUDE.md — the KG pipeline reference still says `/verify` (arscontexta, correct)
```

**Exit criteria:**
- [ ] `.claude/commands/verify.md` no longer exists
- [ ] `.claude/commands/verify-code.md` exists with correct content
- [ ] No stale `/verify` references in `.claude/commands/` or CLAUDE.md (code context)
- [ ] KG pipeline reference in CLAUDE.md unchanged (`/verify` = arscontexta)

---

## Phase 2: Tracking Document Consolidation

### 2.1 Establish Authority Hierarchy and Convert to Append-Only Archive

**What:** Fix the root cause of doc-staleness trust violations by establishing a clear authority
hierarchy across the 5 tracking documents, converting IMPLEMENTATION_PROGRESS.md to an append-only
archive, updating the CLAUDE.md reference table, and cleaning up dead files.
**Status:** READY
**Review Tier:** 1 (documentation restructuring, no code logic)
**Depends on:** None (independent of 1.1)

/implement Tracking Document Consolidation

Consolidate the 5 overlapping tracking documents into a clear 3-tier authority hierarchy.
The root problem: `IMPLEMENTATION_PROGRESS.md`, `PROJECTS.md`, and `ops/goals.md` all track
"what phase is each project on?" independently, leading to drift. When they disagree, agents
trust the stale doc over the user (the PROJECTS.md incident where Project 4 said "not started"
when it was fully built).

**Context:**
- The incident is documented in `ops/observations/stale-docs-caused-agent-to-distrust-user-about-pipeline.md`
- The State Update Rule in MEMORY.md already requires updating `ops/goals.md` — but CLAUDE.md
  tells agents to read IMPLEMENTATION_PROGRESS.md (49K tokens, unreadable) as ground truth
- `ops/tasks.md` is empty and unused — dead weight

**The 3-tier model:**

| Tier | File | Purpose | Update frequency | Read when? |
|------|------|---------|-----------------|------------|
| Session state | `ops/goals.md` | "What am I resuming RIGHT NOW?" | Every session | Session start (orient hook) |
| Project dashboard | `PROJECTS.md` | "Big picture across all projects" | When projects change | Starting new project/milestone |
| Implementation archive | `IMPLEMENTATION_PROGRESS.md` | "How was X built?" | Append-only when phases complete | Debugging, reviewing past work |

**Authority rule:** For current status, `ops/goals.md` > `PROJECTS.md` > `IMPLEMENTATION_PROGRESS.md`.

**Files to modify:**

1. `CLAUDE.md` — Key Reference Documents table (EDIT, lines ~124-131)

   Replace the current reference table entries for tracking documents:

   **Current:**
   ```
   | `IMPLEMENTATION_PROGRESS.md` | **Start of every session**; **update after implementation** |
   ```

   **New:**
   ```
   | `ops/goals.md`               | **Start of every session** (canonical current state — orient hook reads this) |
   | `PROJECTS.md`                | When starting a new project or checking cross-project status (periodic dashboard) |
   | `IMPLEMENTATION_PROGRESS.md` | When reviewing how a module was built (append-only archive, NOT current state) |
   ```

   This is the single highest-impact change — it redirects agents from the 49K unreadable file
   to the 31-line concise one.

2. `PROJECTS.md` — Add authority hierarchy header (EDIT, near top)

   After the existing `> **Last updated:**` line, add:
   ```
   > **Canonical session state:** For what's active RIGHT NOW, see `ops/goals.md`.
   > This dashboard is updated periodically (not every session). If status here
   > conflicts with `ops/goals.md`, trust `ops/goals.md`.
   ```

3. `IMPLEMENTATION_PROGRESS.md` — Convert to append-only archive (EDIT)

   a. Replace the "Current Status" section at the top (lines 9-10 and the "Latest Update" block)
      with an archive header:
      ```
      ## About This File

      This is an **append-only implementation archive**. Each entry records what was built,
      what files were created/modified, key decisions, and test results.

      - **For current project status:** see `ops/goals.md` (canonical) or `PROJECTS.md` (dashboard)
      - **For what to build next:** see `ROADMAP.md`
      - **This file:** reference for "how was module X built?" — consulted on demand, not every session

      ---
      ```

   b. Keep ALL existing session log entries below — they are the valuable history.
      Do NOT delete any session entries.

   c. Rename the existing "Latest Update" / "Previous Update" headers to dated entries
      for consistency with the older session log format. Example:
      `**Latest Update (2026-02-25):**` stays as-is (it's already dated).

4. `.claude/commands/implement.md` — Update Phase 3 and add staleness audit (EDIT)

   a. In Phase 3 (DOCUMENT), change step that says "update IMPLEMENTATION_PROGRESS.md" to:
      ```
      Append a dated entry to `IMPLEMENTATION_PROGRESS.md` with:
      - Module name and status
      - Files created/modified
      - Key decisions
      - Test count and results
      - Review reference
      Do NOT edit existing entries — this file is append-only.
      ```

   b. Add a new Phase 3.5 between DOCUMENT and REVIEW:
      ```
      ## Phase 3.5: STALENESS AUDIT
      After creating/updating module documentation, check whether the implementation
      invalidated any existing documentation claims:

      1. Grep these files for references to the implemented module name:
         - PROJECTS.md
         - ops/goals.md
         - docs/modules/*.md (all module docs, not just the current one)
      2. For each reference found, check if it's still accurate.
         Common drift: status fields saying "Not started" for things that now exist.
      3. If stale references found: fix them and list fixes in the handoff under "## Staleness Fixes"
      4. If clean: note "Staleness audit: clean" in the handoff.
      ```

5. `ops/tasks.md` — DELETE this file
   - It is empty (all sections say "none") and has never been used
   - Task tracking is handled by the Claude Code TaskCreate/TaskUpdate tools during implementation
   - `ops/goals.md` covers active threads

**Test plan:**
```
1. Read ops/goals.md — confirm it still has Active Threads, Waiting, Completed sections
2. Read PROJECTS.md — confirm authority header is present near the top
3. Read IMPLEMENTATION_PROGRESS.md lines 1-20 — confirm "Current Status" replaced with archive header
4. Read IMPLEMENTATION_PROGRESS.md lines 100+ — confirm historical entries are PRESERVED
5. Read CLAUDE.md reference table — confirm 3 separate entries with correct "When to Read" guidance
6. Read .claude/commands/implement.md — confirm "append to" language and staleness audit step
7. Confirm ops/tasks.md no longer exists
8. Grep CLAUDE.md for "IMPLEMENTATION_PROGRESS" — no "start of every session" instruction remains
```

**Exit criteria:**
- [ ] CLAUDE.md reference table has 3 entries (goals.md, PROJECTS.md, IMPLEMENTATION_PROGRESS.md) with distinct read-triggers
- [ ] CLAUDE.md no longer tells agents to read IMPLEMENTATION_PROGRESS.md at session start
- [ ] PROJECTS.md has authority hierarchy header pointing to ops/goals.md
- [ ] IMPLEMENTATION_PROGRESS.md top section is an archive header (no "Current Status")
- [ ] ALL historical entries in IMPLEMENTATION_PROGRESS.md are preserved (no data loss)
- [ ] /implement skill says "append to" not "update" for IMPLEMENTATION_PROGRESS.md
- [ ] /implement skill includes staleness audit step (Phase 3.5)
- [ ] ops/tasks.md deleted

---

## Phase 1-2 Gate

- [ ] `/verify` dispatches to arscontexta note quality (not code verification)
- [ ] `/verify-code` dispatches to py_compile + pytest + flake8
- [ ] Authority hierarchy established: ops/goals.md > PROJECTS.md > IMPLEMENTATION_PROGRESS.md
- [ ] No agent will be told to read a 49K-token file at session start
- [ ] IMPLEMENTATION_PROGRESS.md is append-only (no status section to maintain)
- [ ] /implement skill has staleness audit step
- [ ] No dead files (ops/tasks.md removed)
