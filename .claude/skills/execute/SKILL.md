---
name: execute
description: |
  Continue project work from a handoff or plan document — the canonical
  session-resume pattern in this repo. Auto-triggers on:
    - "/execute <path>"
    - "execute docs/handoffs/X.md"
    - "read and execute X.md"
    - "orchestrate the execution of X.md"
    - "<path>.md execute this plan/handoff"

  Enforces the continuation-session contract:
    - Full read of the handoff (chunked if >500 LOC, per Large File Protocol).
    - Follow pointer chains (handoffs that delegate to a canonical spec).
    - Detect orchestrator-mode signals → propose subagent dispatch graph.
    - Mine binding constraints (Files NOT to touch, HIGH-risk canaries,
      corpus constants, pre-existing tests-as-spec).
    - Run /kcheck if any HIGH-risk canary file is in scope.
    - Apply /implement Step 0 (test-architect) if the handoff invokes it.
    - Present a CLAUDE.md-shaped approval request BEFORE any code.
    - Auto-trigger /wrap-session before `result:` if substantial.

  SKIP for: handoff inspection-only requests ("read X.md and tell me what
  it says" — no execution), or follow-up questions about a handoff already
  in active execution this session.
version: "1.0"
generated_from: "manual"
user-invocable: true
trigger: /execute
argument-hint: "<path-to-handoff-or-plan-document>"
---

# /execute

**Target document: $ARGUMENTS**

If no path provided, ask which handoff/plan to execute. Do NOT guess.

You are the **main session in continuation mode**. The user has handed you a
durable spec that was written by a prior session you don't share context with.
Your job is to make that spec actionable in the current session without losing
the constraints that made it executable in the first place.

## Phase 1 — Resolve and read

1. **Resolve the path.** Accepted forms:
   - `docs/handoffs/<file>.md` (most common — 80%+ of executions)
   - Root `HANDOFF_*.md` or `PLAN_*.md`
   - `.claude/worktrees/<name>/...` (worktree-local handoff)
   - Absolute Windows path under `/mnt/c/users/...` (translate as-is on WSL)
   - Inline handoff text after the word "execute" (paste, not a path)

   If the path doesn't exist, do NOT silently search — print the resolution
   attempts you made and ask the user.

2. **Measure first.** Run `wc -l <path>` before reading.
   - ≤500 LOC: single Read call.
   - >500 LOC: chunked reads in 500-line segments. State the total line
     count after the first chunk. NEVER assume one read captured everything.

3. **Follow pointer chains.** Many handoffs are "thin pointers" that delegate
   to a canonical spec elsewhere. Watch for phrases like:
   - "Read the canonical spec at ..."
   - "Predecessor: ..." / "Read first: ..."
   - "This handoff is a thin pointer; the original ... has the full ..."
   - References to a `PLAN_*.md` at the repo root
   Follow the chain to its leaves. Read each target with the same chunking
   rule. The dispatch graph lives at the deepest spec, not the entry point.

## Phase 2 — Classify the work

Read the handoff(s) and answer:

| Question | Where to look | Why it matters |
|---|---|---|
| Single-stream or multi-stream? | "Stream M/V/X", "Phase 1...N", "Module 18.2a/b" | Determines orchestrator mode |
| Code-modifying or analysis-only? | Verbs: "implement", "refactor", "add" vs. "analyze", "compare", "render" | Determines worktree need |
| Touches a HIGH-risk canary? | "Files NOT to touch", paths in `ops/vault-canary-map.md` | Triggers mandatory /kcheck |
| Invokes /implement? | Literal "`/implement`" or "ROADMAP §X exit criteria" | Triggers test-architect Step 0 |
| Has pre-existing tests as spec? | "test files...do NOT modify their expectations" | Lock test expectations |
| Has a "Done means" / exit criteria? | Section titled exit criteria or completion | Defines `result:` line |

## Phase 3 — Orchestrator-mode detection

**Switch to dispatcher mode** if ANY of the following is true:

- Title or body contains "orchestrator" / "orchestrate".
- Multiple parallel streams (Stream M + Stream V, Phase 1 + Phase 3 + Phase 4).
- Length >150 LOC after pointer-chain expansion.
- Explicit mention of `claude -p` / "spawn an agent" / subagent_type.
- Has a "Files NOT to touch" block (implies multiple concurrent edits).
- Plan has ≥3 phases with at least one I/O-bound step.

**When in dispatcher mode:**
- You are NOT the implementer. You decompose → dispatch → verify → synthesize.
- Use `Agent` tool for in-process subagents; mention `claude -p` only if the
  handoff explicitly calls for it.
- Never accept a subagent's "done" blindly — verify with a fresh agent or a
  direct re-read of the artifact.
- Parallelize I/O-bound steps (data downloads, batch detection runs) with
  cognitive steps (coding, review). Background long jobs before coding.
- Cite the rule once in your proposal: `feedback_orchestrator_mode`.

## Phase 4 — Constraint extraction

Mine the handoff(s) and produce a flat constraint list in your proposal:

1. **Files NOT to touch** — list every path verbatim. HIGH-risk canaries get
   a 🔒 marker.
2. **Production model paths** — e.g., `models/hard_neg_retrain/best_model.pt`
   is the current CNN; do not silently swap to a different model.
3. **Corpus constants** — if the work involves sample rate, USV band, STFT
   params, ICI, bout threshold: load `data/corpus_facts/{dataset}.json`
   (Layer 2). Read `src/usv_spectrogram/corpus.py` only if the change could
   touch a Layer-1 constant. Never redeclare canonical constants.
4. **Pre-existing tests** — any test file the handoff references is SPEC.
   Do NOT modify expectations during implementation without explicit
   user discussion.
5. **Vault memory references** — if the handoff cites a vault note
   (e.g., `feedback_rig_artifact_mean_power_db`), read it before reasoning.

## Phase 5 — Worktree and kcheck

1. **Worktree check.** If cwd is NOT under `.claude/worktrees/` AND the work
   modifies code:
   - This session is a background job → use `EnterWorktree` per the bg
     session contract.
   - Otherwise propose creating one with `git worktree add` and ask. Don't
     edit shared paths in parallel with other sessions.

2. **/kcheck.** If ANY HIGH-risk canary file is in scope, run /kcheck BEFORE
   the approval request. This is mandatory per CLAUDE.md "Vault Canary
   Comments". Surface findings in the proposal under "Vault constraints".

## Phase 6 — Approval request

**Even with Auto Mode active, executing a handoff is non-trivial and the
CLAUDE.md state machine forbids ANALYSIS → EXECUTION skip.** Present this
proposal verbatim shape (skip empty sections):

```
## Execute proposal: <handoff filename>

**Intent:** <one sentence — what the handoff asks for>
**Mode:** <single-stream | orchestrator with N streams>
**Worktree:** <none needed | already in <path> | proposing new at <path>>

**Pointer chain followed:**
- <entry handoff> (N lines)
- → <canonical spec> (M lines)
- → <plan file> (P lines)  [if applicable]

**Dispatch graph:** [orchestrator mode only]
- Parallel: Stream A (agent: X), Stream B (agent: Y)
- Serial: Stream C (waits on A), Stream D (waits on A+B)
- Background: <I/O-bound job> while coding

**Binding constraints:**
- 🔒 NOT to touch: <file1>, <file2>, ...
- Production model: <path>
- Corpus constants: <imported from corpus.py — not redeclared>
- Pre-existing tests as spec: <list>
- Vault: <constraint summaries from /kcheck>

**Assumptions I'm making:**
- <each assumption, one line>

**Risks:**
- <each risk + mitigation>

**Validation plan:**
- <how I'll know each phase succeeded>
- <Done means: copy the handoff's exit criteria verbatim>

**Learning opportunity:**
- <what concept this work teaches; defaults to the domain area touched>

Proceed?
```

For trivial single-stream handoffs (<50 LOC, no canaries, no /implement):
collapse to one paragraph + "Proceed?" — don't bureaucratize small jobs.

## Phase 7 — Execution

After approval:

1. **If /implement is invoked:** Step 0 first — spawn `test-architect` for
   pre-implementation tests on any new module. Do NOT silently skip even if
   the handoff doesn't spell it out — CLAUDE.md "Implementation Completion
   Sequence" makes Step 0 mandatory. If the user opts out, record that
   explicitly.

2. **Stage by exact path.** Never `git add -A` or `git add .` in a parallel
   chat workflow (per `feedback_no_bulk_stage_in_parallel_chats`). The
   2026-05-21 incident: a Stream 5 memo got swept into a "9252-analysis"
   commit. Stage only what this stream produced.

3. **Narrate.** This session is a bg job. One line on your approach before
   acting. After each chunk: what happened, what's next. Tool output is not
   visible to the extractor — restate results in your own text.

4. **Verify before claiming done.** Per CLAUDE.md Verification Protocol:
   `py_compile` on every modified `.py`, `pytest <relevant_test>` if tests
   exist. If no tests exist, say so explicitly. Never report success with
   failing checks.

## Phase 8 — End-of-execution

**Before writing `result:`,** auto-trigger `/wrap-session` if any TWO of:
- Ran ≥2 scripts (compute, not lookups).
- Wrote/edited ≥3 source files.
- Produced ≥3 output artifacts the user might revisit.
- Executed a handoff (which is true by definition here — counts as 1).
- Made a non-trivial decision.
- Deferred a phase / left work for a follow-up.

If wrap-session runs, the `result:` line should reference the HTML report
and (if a successor handoff was written) point to it.

If the work produced HTML / PDF / large PNG outputs on WSL: include the
`file://wsl.localhost/Ubuntu/<path>` URL in the user-facing message, not
just `SendUserFile`. This is mandatory per `feedback_wsl_file_viewing` and
applies to subagent outputs too.

## Anti-patterns this skill exists to prevent

| Anti-pattern | Why it bites | Where it bit before |
|---|---|---|
| Editing code from the handoff text without re-reading the target file | Context decay after long reads; file may differ from handoff snippet | CLAUDE.md "Context Decay" |
| Skipping test-architect because the handoff didn't mention it | Tests-after-impl drift from spec; reviewer can't audit | CLAUDE.md "Implementation Completion Sequence" Step 0 |
| `git add -A` to commit "all stream output" | Sweeps unrelated WIP into the commit | `feedback_no_bulk_stage_in_parallel_chats` — incident 375d4bdc |
| Treating the handoff's prose summary as the spec | Pointer chain skipped; misses the canonical constraints | Stream X handoff explicitly says "do not re-derive" |
| Running detection without all 5 pipeline flags | Silently produces an unreliable triage tier | CLAUDE.md "Running Batch Detection" |
| Citing memory's "X exists" without verifying | Memory may name a renamed/removed file | CLAUDE.md "Before recommending from memory" |
| Modifying test expectations to make tests pass | Test-greenwashing | CLAUDE.md "Test Protocol" |

## Reference

- Approval template: `docs/workflow/approval-request-template.md`
- Wrap-session contract: `~/.claude/skills/wrap-session/SKILL.md`
- Vault canary registry: `ops/vault-canary-map.md`
- Orchestrator-mode rule: `auto-memory/feedback_orchestrator_mode.md`
- Corpus invariant: `docs/modules/corpus-constants.md`
