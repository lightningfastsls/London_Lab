---
name: roadmap-from-plan
description: Convert a web Claude implementation plan into a standalone ROADMAP file with /implement blocks, then extract theoretical knowledge to the KG.
---

Convert the following implementation plan into structured ROADMAP.md format: $ARGUMENTS

## Your Task

You are converting a high-level implementation plan (typically produced by web Claude) into the structured ROADMAP.md format used by this project. Each step in the plan becomes a module entry with an `/implement` block that the `/implement` skill can execute.

## Step 1: Read Current State

1. Read `ROADMAP.md` — find the last phase number so you can continue numbering
2. Read `DECISIONS.md` — understand existing ADRs that new modules must reference
3. Read `docs/architecture/patterns.md` — understand established patterns new modules should follow
4. Skim the plan provided by the user to understand its scope

## Step 2: Analyze the Plan

Parse the user's plan and identify:
- **Phases/steps**: What are the distinct implementation units?
- **Dependencies**: Which steps depend on others?
- **Files**: What files need to be created or modified in each step?
- **Data structures**: What configs, dataclasses, or models are defined?
- **Algorithms/logic**: What does each module actually do?
- **Test requirements**: What should be tested?
- **Exit criteria**: What does "done" look like for each step?

If the plan is vague on any of these, use your judgment based on the project's patterns (frozen dataclasses, CLI scripts, pytest tests, etc.) to fill in reasonable defaults. Flag assumptions.

## Step 3: Generate ROADMAP Entries

For EACH step in the plan, generate a ROADMAP module entry following this exact format:

```markdown
### N.M Module Name

**What:** [1-2 sentence description of what to build]
**Status:** READY (or BLOCKED if depends on unbuilt module)
**Review Tier:** [1 = trivial config/glue, 2 = standard module, 3 = DSP/ML/complex logic]
**Depends on:** [Phase N.M or "None"]

/implement [Module Name]

[Brief imperative description of what to build — 1-2 sentences]

**Context:** [Reference relevant docs, ADRs, design documents. Mention key constraints.]

**Files to create:**

1. `path/to/file.py` (NEW) — [Description]

```python
@dataclass
class ConfigName:
    field: type = default    # comment explaining the field
```

[Architecture/logic description with enough detail for implementation]

2. `path/to/another_file.py` (NEW) — [Description]

[Same pattern — show data structures, describe logic]

**Test plan:**
```
1. [Specific test case]
2. [Another test case]
...
```

**Exit criteria:**
- [ ] [Specific, verifiable criterion]
- [ ] [Another criterion]
- [ ] All tests pass
- [ ] py_compile passes on all new files

---
```

## Formatting Rules

1. **Phase numbering**: Continue from the last phase in ROADMAP.md (currently Phase 8). If the plan is a sub-phase, use N.M numbering.

2. **Review tiers**:
   - Tier 1: Config files, simple glue code, exports
   - Tier 2: Standard modules with clear logic, data pipelines, CLI tools
   - Tier 3: DSP/signal processing, ML models, complex algorithms

3. **`/implement` blocks must be self-contained**: A fresh Claude Code session should be able to execute the `/implement` block without needing the original web Claude plan. Include:
   - All dataclass definitions with field types and defaults
   - Architecture descriptions (for models: layer sizes, activations, shapes)
   - Algorithm descriptions (for processing: step-by-step logic)
   - Key design decisions and constraints
   - References to existing code that should be followed or reused

4. **Code snippets**: Include Python dataclass/config definitions inline. Show forward pass signatures for models. Show CLI argument structure for scripts.

5. **Dependencies**: Mark modules as BLOCKED if they depend on something not yet built. Mark as READY if all dependencies exist.

6. **Test plans**: Be specific. "Tests pass" is not a test plan. Each test should describe WHAT is being verified.

7. **Exit criteria**: Must be objectively verifiable. Include shape checks, loss thresholds, metric targets where applicable.

8. **DSP parameters**: If any module touches audio/spectrograms, reference ADR-001 (sr=300000) and ADR-002 (n_fft=512, hop=128). Never leave sample rate implicit.

## Step 4: Assemble Phase Gate

If the plan spans multiple modules, add a phase gate after the last module:

```markdown
## Phase N Gate

- [ ] [Summary criterion for each module]
- [ ] All tests passing
- [ ] All module docs written
```

## Step 5: Present to User

Show the generated ROADMAP section and ask the user:
1. Does the phase numbering make sense?
2. Are the review tiers correct?
3. Are there any missing dependencies?
4. Should any steps be split further or merged?
5. Any context from the web Claude conversation that should be added to the `/implement` blocks?

**Do NOT write to ROADMAP.md yet.** Present the formatted output and wait for approval. After approval, write to a **new standalone file** named `ROADMAP_<PLAN_NAME>.md` (e.g., `ROADMAP_VACATION_DRAFT.md`). Do NOT append to the main ROADMAP.md — standalone roadmap files work with `/implement` just as well, and keeping them separate avoids bloating the main ROADMAP.

## Step 6: Extract Theoretical Knowledge to KG

Web Claude plans frequently contain rich theoretical content — scientific rationale, methodology choices, statistical methods, domain insights — that goes beyond implementation instructions. This knowledge must be captured in the knowledge graph, not just preserved in `/implement` Context sections.

**After the ROADMAP file is written**, do the following:

1. **Scan the original plan** for theoretical content: scientific methods, domain insights, design rationale, statistical approaches, literature references, methodology decisions
2. **Ask the user**: "This plan contains theoretical knowledge about [topics]. Should I run `/reduce` on the source to extract it into the knowledge graph?" — present a brief summary of what would be extracted
3. **If approved**, run `/reduce` on the source plan file (or the inbox copy if it was `/seed`'d). This extracts atomic notes, enriches existing notes, and identifies tensions/open questions
4. **If the plan is not yet in inbox/**, suggest `/seed` first to establish provenance before `/reduce`

**Why this step is mandatory:** Implementation plans from web Claude are dual-purpose documents — they contain both task specifications AND domain knowledge. The ROADMAP captures the "what to build" but loses the "why this method" and "how this connects to the literature." Without this step, theoretical knowledge evaporates after the session ends.

## Important Notes

- The `/implement` blocks are the most critical part. They serve as the specification that Claude Code will follow during implementation. Err on the side of MORE detail, not less.
- If the plan from web Claude includes scientific rationale or theoretical motivation, preserve it in the **Context** section — it helps Claude Code make better design decisions. Additionally, this theoretical content should be extracted to the knowledge graph via Step 6.
- If the plan references external papers or techniques, mention them so Claude Code can search for implementation details.
- Preserve any code snippets, architecture diagrams, or formulas from the original plan.
