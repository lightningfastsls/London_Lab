# Plan: CLAUDE.md Vault Search Integration & Slim-Down

## Context

The agent doesn't search the vault before domain reasoning tasks (explanations, architecture analysis, design rationale questions). It only searches before file modifications of high-risk canary files. This means vault knowledge (design decisions, constraints, prior findings) never enters the context window for non-modification tasks.

Research backing: activation timing is the bottleneck, not retrieval quality. The agent's metacognitive confidence diverges from its actual knowledge — it doesn't know what it doesn't know.

## Goals

1. Add a Knowledge Activation rule to Core Rules (Behavioral Contract) so vault search fires before domain reasoning, not just file modifications
2. Delete the now-superseded "Mid-Session Knowledge Checks" section
3. Move Project Structure block to a reference doc (slim CLAUDE.md by ~65 lines)
4. Embed KG convention triggers into skill definitions (safest path for moved content)

---

## Phase 1: Pre-Flight Checks

Before making any changes, verify what already exists.

### Task 1.1: Check for existing reference docs

The user suspects some of the content targeted for extraction may already live in reference docs. Check:

```bash
# Check if project structure / file index already exists somewhere
cat docs/architecture/patterns.md   # mentioned in CLAUDE.md line 176
cat docs/scripts-index.md           # mentioned in CLAUDE.md line 163
ls docs/architecture/
ls docs/reference/

# Check the KG reference doc — does it already cover conventions?
cat docs/workflow/knowledge-graph-reference.md  # mentioned in CLAUDE.md line 180

# Check notes/index.md — is this already a structure overview?
head -80 notes/index.md
```

**Decision gate**: If `knowledge-graph-reference.md` already contains Atomic Notes / Wiki Links / Topic Maps / Processing Pipeline / Schema conventions, then Phase 3's KG extraction is already done — just verify the content is current and skip that extraction. If `docs/architecture/` already has a structure index, same logic for Phase 2.

---

## Phase 2: Surgical Edit — Knowledge Activation Rule

This is the high-priority behavioral change. Do this first, independently of any structural refactoring.

### Task 2.1: Add Knowledge Activation to Core Rules

Insert after the "Epistemic Honesty" block (after line 62, before line 64 "### Approval Request Format"):

```markdown
#### Knowledge Activation
- **Search before reasoning**: Before explaining, analyzing, or modifying domain-specific systems,
  search the vault (qmd + topic maps). Filter: "Would the vault plausibly change my answer?"
- **Modifications**: /kcheck mandatory for HIGH-risk canary files, recommended for constrained systems.
- **Skip for**: pure code mechanics, general knowledge, test files, documentation-only changes.
```

### Task 2.2: Delete Mid-Session Knowledge Checks

Delete lines 250-256 (the "## Mid-Session Knowledge Checks" section). This is fully superseded by the new Core Rule. The Vault Canary Comments section (lines 237-242) stays — it's about file-level canary markers, not the search trigger.

### Task 2.3: Add cross-reference in Session Rhythm

Change line 287 from:
```
- **Work**: Do the task. Surface connections. Write down discoveries immediately.
```
to:
```
- **Work**: Do the task. **Vault search before domain reasoning (Core Rules → Knowledge Activation).** Surface connections. Write down discoveries immediately.
```

### Net effect: +4 lines, -7 lines = **-3 lines net**

---

## Phase 3: Extract Project Structure to Reference Doc

Move lines 100-166 (Project Structure block) out of CLAUDE.md into a reference doc.

### Task 3.1: Check what exists (from Phase 1 results)

If a structure index already exists in `docs/architecture/` or similar, update it rather than creating a new file.

### Task 3.2: Create or update reference doc

Target location: `docs/architecture/project-structure.md` (or wherever the existing index lives).

Move the full project structure tree and the "All `src/` paths above are relative to..." note.

### Task 3.3: Replace in CLAUDE.md with pointer

Replace the entire Project Structure block with:

```markdown
## Project Structure

See `docs/architecture/project-structure.md` for the full directory tree.
Key entry points are listed in the Task Routing table below.
```

### Net effect: ~-60 lines

---

## Phase 4: KG Conventions — Embed Triggers in Skills

**⚠️ IMPORTANT: Before implementing this phase, confer with `arscontexta-expert` agent.** The expert should validate:
- Whether embedding KG conventions into skill definitions is methodologically sound
- Which skills need which conventions (not all skills need all rules)
- Whether this changes the relationship between CLAUDE.md and the skill layer in ways that create maintenance burden or authority conflicts

### Rationale

Lines 315-440 of CLAUDE.md contain KG conventions (Atomic Notes, Wiki Links, Topic Maps, Processing Pipeline, Schema, Maintenance, etc.). These are important but they're reference material, not behavioral rules. Moving them behind a pointer risks the agent never reading them. The safest extraction path is embedding the relevant subset into each skill that needs them:

- `/reduce` needs: Atomic Notes conventions, Schema, Processing Pipeline, Wiki Links
- `/reflect` needs: Wiki Links, Topic Maps, link density targets
- `/reweave` needs: Topic Maps, Maintenance thresholds, Wiki Links
- `/seed` needs: Processing Pipeline, inbox routing
- `/verify` needs: Schema validation, note quality criteria

### Task 4.1: Consult arscontexta-expert

Ask the expert to review this plan and confirm or modify the skill-convention mapping above.

### Task 4.2: Audit skill definitions

Check each skill's current definition to see if it already loads KG conventions or if there's an injection point:

```bash
# Find skill definition files
ls -la skills/
# or wherever skill definitions live — check CLAUDE.md or ops/ for skill registry
```

### Task 4.3: Add convention loading to skills

For each skill identified in 4.1, add an instruction to load the relevant KG conventions at skill invocation time. The exact mechanism depends on how skills are defined (if they have a preamble section, a "read before executing" block, etc.).

### Task 4.4: Slim CLAUDE.md KG section

Once conventions are embedded in skills, reduce the KG section in CLAUDE.md to:
- Philosophy (keep — it's behavioral framing)
- Discovery-First Design (keep — quality gate)
- Brief pointers to the full conventions doc
- Session Rhythm, Where Things Go, Operational Space (keep — these are behavioral, not reference)

Move the detailed convention content (Atomic Notes, Wiki Links, Topic Maps, Schema, Processing Pipeline, Maintenance, Graph Analysis, etc.) to `docs/workflow/knowledge-graph-reference.md` (which may already exist per line 180).

### Net effect: ~-80 lines (estimate, depends on what stays)

---

## Execution Order

1. **Phase 1** first — understand what already exists
2. **Phase 2** immediately after — this is the critical behavioral fix, zero dependencies
3. **Phase 3** next — low risk, clear extraction
4. **Phase 4** last — requires arscontexta-expert consultation, higher complexity

Phases 2 and 3 can be done in one session. Phase 4 is a separate session.

---

## Validation

After Phase 2, test the behavioral change:
- Ask the agent to explain the VQ-VAE architecture (the original failure case)
- Ask "why did we choose K=64 codebook entries?"
- Ask about detection pipeline design decisions
- In each case, verify the agent searches the vault BEFORE answering

After Phase 4, verify:
- Running `/reduce` on a test note → agent follows Atomic Notes conventions without CLAUDE.md containing them
- Running `/reflect` → agent checks link density, topic map membership
- Running a skill that DOESN'T need KG conventions (e.g., pure code task) → no unnecessary vault loading
