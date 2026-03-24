---
name: roadmap-from-plan
description: Convert a plan into a standalone ROADMAP file with /implement blocks, then extract theoretical knowledge to the KG.
---

Convert the following implementation plan into structured ROADMAP format: $ARGUMENTS

## What This Skill Does

You are converting a high-level implementation plan into the structured ROADMAP format used by this project. Each step becomes a module entry with a self-contained `/implement` block — meaning a fresh Claude Code session can execute it without needing the original plan.

This is a two-part job:
1. **Format conversion** — plan steps become implementable ROADMAP modules
2. **Knowledge capture** — theoretical insights in the plan get routed to the knowledge graph

Both parts matter. The ROADMAP captures *what to build*. The KG captures *why this approach* — methodology choices, scientific rationale, and domain insights that would otherwise evaporate after the session ends.

## Step 1: Resolve the Input

The user's input might be:
- **A file path** (e.g., `PLAN_foo.md`, `docs/plans/bar.md`) — read the file
- **Pasted text** — use it directly
- **A URL or reference** — fetch or ask the user to paste the content

If the argument looks like a file path (contains `/`, `\`, or ends in `.md`/`.txt`/`.pdf`), try reading it first. If it doesn't exist, ask the user.

## Step 2: Read Project State

1. Read `ROADMAP.md` — find the **last phase number** so you can continue numbering. Count the actual phases — never hardcode a phase number.
2. Scan for other `ROADMAP_*.md` files — check for naming conflicts
3. Read `DECISIONS.md` — understand existing ADRs that new modules must reference
4. Read `docs/architecture/patterns.md` — established patterns new modules should follow

## Step 3: Analyze the Plan (and Flag Gaps)

Parse the plan and identify for each step:
- **What it builds** (files, modules, features)
- **Dependencies** (which steps need others done first)
- **Data structures** (configs, dataclasses, models)
- **Algorithms/logic** (what each module actually does)
- **Test requirements** and **exit criteria**

### Gap detection

If any step has fewer than 2 sentences of implementation detail, flag it. Present a summary to the user before generating:

> "Steps 1, 2, and 4 have enough detail for self-contained /implement blocks. Step 3 ('Add caching layer') is under-specified — I don't know what to cache, where, or what eviction strategy to use. Can you clarify, or should I make reasonable assumptions and flag them?"

Fill gaps with reasonable defaults based on project patterns (frozen dataclasses, CLI scripts, pytest tests), but **mark every assumption** with `[ASSUMED]` so the user can spot-check during review.

### Theoretical content scan

While analyzing, note any scientific rationale, methodology choices, statistical methods, literature references, or domain insights. You'll use these in Step 7 (KG extraction). Jot them down now so you don't lose them.

## Step 4: Generate ROADMAP Entries

For EACH step in the plan, generate a module entry. See `references/example-module.md` for a complete worked example.

The format:

```
### N.M Module Name

**What:** [1-2 sentence description]
**Status:** READY | BLOCKED
**Review Tier:** [1 | 2 | 3]
**Depends on:** [Phase N.M or "None"]

/implement [Module Name]

[Brief imperative description — 1-2 sentences]

**Context:** [References to docs, ADRs, constraints. Preserve scientific rationale here.]

**Files to create:**

1. `path/to/file.py` (NEW | EDIT) — [Description]

    ```python
    @dataclass(frozen=True)
    class ConfigName:
        field: type = default    # comment explaining the field
    ```

    [Architecture/logic description with enough detail that a fresh session can build it]

**Test plan:**
    ```
    1. [Specific test case — WHAT is being verified]
    2. [Another test case]
    ```

**Exit criteria:**
- [ ] [Specific, verifiable criterion]
- [ ] All tests pass
- [ ] py_compile passes on all new files

---
```

### What makes a good /implement block

The `/implement` block is the most critical output — it's the spec a fresh Claude Code session will follow. It must be **self-contained**: a reader who has never seen the original plan should be able to implement the module from the block alone.

A good block answers these questions:
- **What files** to create or modify (exact paths)
- **What data structures** to define (show the dataclass with field types and defaults)
- **What logic** to implement (algorithm description, not just "process the data")
- **What constraints** apply (sample rates, parameter ranges, size limits)
- **What patterns** to follow (reference existing code as examples)

A bad block says "implement caching." A good block says "Add an LRU cache to `src/api/client.py` using `functools.lru_cache` with a 1000-entry limit on the `fetch_vehicle()` method. Cache key is `(plate_number,)`. Add a `clear_cache()` method exposed via the CLI."

### Review tier assignment

| Tier | Criteria | Example |
|------|----------|---------|
| 1 | Config files, renames, simple glue, exports | Rename a command, add a re-export |
| 2 | Standard modules with clear logic, data pipelines, CLI tools | Database layer, API client, data parser |
| 3 | DSP/signal processing, ML models, statistical algorithms, complex math | STFT computation, VQ-VAE, information theory |

When in doubt, tier up. A Tier 3 review that finds nothing wrong is fine; a Tier 2 review that misses a DSP bug is not.

## Step 5: Phase Gate

If the plan spans multiple modules, add a phase gate after the last module:

```
## Phase N Gate

- [ ] [Summary criterion for each module]
- [ ] All tests passing
- [ ] All module docs written
```

## Step 6: Present and Write

**Present the complete output to the user** and ask:
1. Does the phase numbering make sense?
2. Are the review tiers correct?
3. Any missing dependencies?
4. Should any steps be split or merged?
5. Any context from the original plan missing from the /implement blocks?

**Do NOT write to ROADMAP.md.** After approval, write to a standalone `ROADMAP_<PLAN_NAME>.md` file. Standalone roadmaps work with `/implement` just as well, and keeping them separate prevents bloating the main ROADMAP.

## Step 7: Extract Theoretical Knowledge to KG

Implementation plans — especially from web Claude — are dual-purpose documents. They contain both *task specifications* AND *domain knowledge*. The ROADMAP captures the "what to build" but loses the "why this method" and "how this connects to the literature."

**After the ROADMAP file is written:**

1. Review the theoretical content you flagged during Step 3
2. Present a summary: "This plan contains theoretical knowledge about [topics]. Should I run `/reduce` on the source to extract it into the knowledge graph?"
3. If approved and the source isn't in `inbox/` yet, suggest `/seed` first for provenance
4. Run `/reduce` on the source file

This step matters because six months from now, someone will look at the `/implement` block and know *what* to build, but not *why* this approach was chosen over alternatives. The KG preserves that reasoning.

## Formatting Rules

1. **Phase numbering**: Always dynamic — read ROADMAP.md and count. Never hardcode.
2. **DSP parameters**: If ANY module touches audio/spectrograms, reference ADR-001 (sr=300000) and ADR-002 (n_fft=512, hop=128). Never leave sample rate implicit.
3. **Code snippets**: Include Python dataclass/config definitions inline. Show forward pass signatures for models. Show CLI argument structure for scripts.
4. **Dependencies**: BLOCKED if depending on an unbuilt module. READY if all dependencies exist.
5. **Test plans**: Be specific. "Tests pass" is not a test plan. Each test describes WHAT is being verified.
6. **Exit criteria**: Objectively verifiable. Include shape checks, loss thresholds, metric targets.
7. **Assumptions**: Mark with `[ASSUMED]` — every assumption visible, not hidden.
