# Handoff: Priority 4 Investigations — arscontexta Health Restoration

> **Created:** 2026-03-22
> **Origin:** Re-evaluation of Priority 4 from `arscontexta_health_restoration_plan.md`
> **Status:** COMPLETE (2026-03-22) — all 4 investigations finished
> **Prerequisite fix (DONE):** /remember SKILL.md restored from Cloudy-Claude backup

## Background

The arscontexta health restoration plan had 4 priority tiers. P1-P3 are ~85% done (schema 100%, blocking validation, archive step all done; reference docs 1.4 and /rethink threshold 3.3 still pending). This handoff covers the P4 investigations, re-evaluated against the current repo state.

## What Changed Since the Original Plan

| Original Assumption | Reality Found | Impact |
|---------------------|---------------|--------|
| 324 session transcripts available for mining | Only 10 survive in Claude cache (314 GC'd) | 4.2 dramatically descoped |
| /remember skill works | Was a truncated stub (12 lines) | **FIXED** — restored from Cloudy-Claude |
| session-orient misses *task-specific* context | Misses *ALL* context (0 matches for both active threads) | 4.1 is worse than expected |
| BM25 dilution is the main retrieval issue | Also found: hyphen-as-negation bug, score uniformity, qmd instability under parallel load | 4.3 has more issues than expected |

## qmd System Info

- **769 docs indexed** (523 mickey-lab, 143 cloudy-claude, 103 notes)
- **Vector index current** (0 needing embedding, last updated mickey-lab 2026-03-10)
- **Three search modes:** lex (BM25 keyword), vec (embedding), hyde (hypothetical doc + reranking)
- **Known stability issue:** vec queries crash with "Object is disposed" under parallel load
- **Config:** `.mcp.json` in repo root

---

## Investigation 4.3: qmd Retrieval Precision — START HERE

**Why first:** 4.1 (context surfacing) depends on knowing whether the problem is qmd retrieval quality or query construction in the hooks.

**Partial results saved:** `docs/investigations/4.3-qmd-retrieval-precision-benchmark.md`

### What's Done (6 of ~40 query-mode combinations)

3 domain queries × lex mode = all worked well (0.93 scores, relevant results)
3 domain queries × vec mode = 1 good, 1 imprecise, 1 crashed

### 5 Findings Already Confirmed

1. **qmd crashes under parallel vec queries** — "Object is disposed" error. Run queries sequentially, not in parallel.
2. **Hyphen-as-negation bug** — "VQ-VAE" in vec/hyde is parsed as "VQ" minus "VAE". Use "VQVAE" instead. Affects all hyphenated domain terms (self-supervised, cross-domain, pre-trained, etc.).
3. **Lex works well for 3-5 keyword queries** — all domain lex queries found relevant top results.
4. **Vec precision is lower than lex** — returns semantic neighbors, not exact intent matches.
5. **Uniform 0.93 top scores** — every query returns 0.93 regardless of actual relevance. Score-based thresholds may be meaningless.

### What Remains

**Run these queries (sequentially to avoid crashes):**

Operational:
- Q4 lex+vec: "arscontexta health maintenance schema compliance"
- Q5 lex+vec: "description quality search retrieval"
- Q6 lex+vec: "session mining patterns"

Cross-domain:
- Q7 lex+vec: "LMT behavioral event triggered analysis"
- Q8 lex+vec: "Raven export selection table format"

**Goal-thread queries (CRITICAL — this tests session-orient's failure):**
- Q9 full lex+vec: "DeepSqueak Classification Bridge Phase 2 Raven export DONE Phase 3 MATLAB import clustering IN PROGRESS"
- Q9c condensed lex: "DeepSqueak Raven classification bridge"
- Q10 full lex+vec: "Phase 5.3 Next validation checkpoint maintenance overhead reduce fixes rethink threshold"
- Q10c condensed lex: "arscontexta health validation checkpoint"

**Self-retrieval test (tests BM25 dilution directly):**
Use each note's `description:` field as the lex query, check if the note self-retrieves in top 5:
1. "LLM agents without tasks show model-specific determinism..."
2. "Spectrograms contain identity information beyond call type..."
3. "MUPET's unsupervised approach discovers 100-140 data-driven types..."
4. "Greptile scored 82 percent in own benchmark but 45 percent from Augment Code..."
5. "LoRA, PEFT variants, hypernetworks..." (model-adaptation.md topic map)
6. "Covers best practices for training data, evaluation metrics..."
7. "Graph traversal theory, emergent inter-note knowledge..." (graph-structure.md topic map)
8. "DSP foundation — 300 kHz sample rate..." (signal-processing.md topic map)
9. "Core STFT parameter choice balancing temporal precision..."
10. "Open question on whether mouse ID, sex, strain..."

**Hyde mode:** Test 3-5 queries in hyde mode to compare against lex and vec.

### Deliverable

Update `docs/investigations/4.3-qmd-retrieval-precision-benchmark.md` with:
- Complete results table
- Precision@5 for each mode
- Self-retrieval success rate
- Comparison: full prose vs condensed queries (quantifies BM25 dilution)
- Specific recommendations ranked by effort/impact

---

## Investigation 4.1: Task-Aware Context Surfacing — AFTER 4.3

**The problem:** `session-relevance.md` shows "No strong matches above relevance threshold" for BOTH active goal threads. The orient hook is not surfacing any vault context at session start.

**Root cause hypothesis:** session-orient.ps1 (~492 lines) constructs queries by truncating goal thread prose to 120 chars. For "DeepSqueak Classification Bridge — Phase 2 (Raven export) DONE, Phase 3 (MATLAB import+clustering) IN PROGRESS", this creates a diluted query full of status words.

**Critical file:** `.claude/hooks/session-orient.ps1` (query construction around lines 250-310)

### Steps

1. Read session-orient.ps1 to understand exact query construction logic
2. Use 4.3's goal-thread query results (Q9 full vs Q9c condensed) to confirm whether query construction is the issue
3. If confirmed: propose a fix — extract domain terms, strip status words ("DONE", "IN PROGRESS", "Phase N")
4. Separately evaluate: should the hook also use the user's first message (task-specific context)?
5. Test the fix by running the improved queries against qmd

### Deliverable

Analysis document + concrete code change proposal for session-orient.ps1

---

## Investigation 4.2: Session Pattern Mining — RESTRUCTURED

**Dramatically different from original plan.** Cannot mine 324 sessions because only 10 transcripts survive.

**Prerequisite (DONE):** /remember SKILL.md restored from `/mnt/d/we_do_this/Cloudy-Claude/.claude/skills/remember/SKILL.md`

### Phase A: Infrastructure Reality Check

1. **Read 2-3 surviving transcripts.** They're JSONL files at `/home/light/.claude/projects/-mnt-d-mickey-london-lab/*.jsonl` (10 files exist). Understand format, content, what's extractable.

2. **Deduplicate session pointers.** 324 JSON files in `ops/sessions/` but many share transcript UUIDs (conversation resumes create new pointers). Run:
   ```bash
   cat ops/sessions/*.json | grep transcript_path | sort -u | wc -l
   ```
   to get actual conversation count.

3. **Key design question:** If Claude Code GCs transcripts after ~10 sessions, batch mining (`/remember session`) is architecturally wrong — by the time you mine, the data is gone. Should extraction happen at capture-time in `session-capture.ps1` instead?

4. Check whether there's a Claude Code setting to extend transcript retention.

### Phase B: Design Proposal (only if Phase A shows value)

5. Define what patterns to extract (failed searches, repeated context gathering, correction patterns)
6. Propose architecture: capture-time extraction (enrich session JSON at creation) vs batch mining
7. Update /remember skill if needed, or propose a standalone extraction script

### Deliverable

Feasibility assessment answering: "Is session mining worth investing in given transcript ephemerality, and if so, should it be capture-time or batch?"

---

## Investigation 4.4: OpenViking Re-evaluation — CONDITIONAL

**Do not start until 4.3 is complete.**

Three trigger conditions:
- [ ] qmd ceiling hit (4.3 concludes query fixes are insufficient)
- [ ] Session mining insufficient (4.2 shows transcripts are too ephemeral)
- [ ] /reflect bottleneck (no evidence currently)

**If no triggers fire:** Close this item. Write a one-paragraph decision note: "OpenViking deferred — qmd retrieval is sufficient with query construction fixes."

**If triggers fire:** Investigate OpenViking specifically for the triggered use case.

---

## Execution Sequence

```
1. Finish 4.3 benchmark (qmd retrieval precision)     — ~30 min
2. Run 4.1 analysis (context surfacing)                — ~20 min, uses 4.3 data
3. Run 4.2A infrastructure check (session mining)      — ~15 min, independent
4. Write 4.2B design proposal (if 4.2A shows value)    — ~15 min
5. Make 4.4 decision (OpenViking close-or-open)        — ~5 min
6. Update arscontexta_health_restoration_plan.md        — mark P4 complete or archive
```

## Also Still Pending from P1-P3

These are NOT part of P4 but were still open when P4 started:
- **1.4 Reference doc compliance** — 11 of 15 reference files missing PRD template sections
- **3.3 /rethink threshold** — may already be lowered to 7 in queue.json (verify)

## Files to Read First

| File | Why |
|------|-----|
| `docs/investigations/4.3-qmd-retrieval-precision-benchmark.md` | Partial benchmark results + 5 findings |
| `arscontexta_health_restoration_plan.md` | Original P4 definitions |
| `.claude/hooks/session-orient.ps1` | Query construction logic (4.1) |
| `.claude/hooks/session-capture.ps1` | Session pointer creation (4.2) |
| `ops/session-relevance.md` | Current orient output showing 0 matches |
| `methodology/BM25 retrieval fails on full-length descriptions because query term dilution reduces match scores.md` | Theoretical foundation for 4.3 |
