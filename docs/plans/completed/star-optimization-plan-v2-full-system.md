# STAR Optimization Plan v2 — Full System

> **Scope:** USV detection pipeline, knowledge management skills, agent prompts, commands, and Cloudy Claude API prompts (from v1 plan).
>
> **Core principle from arXiv:2602.21814:** Forced goal articulation before inference outperforms context injection by 2.83× for tasks requiring implicit constraint reasoning. Your system already does this at the CLAUDE.md level (state machine, approval templates). This plan extends it downward into the layers that currently lack it.

---

## Architecture Map — Where STAR Exists vs Where It's Missing

```
Layer                          STAR-like?    Status
──────────────────────────────────────────────────────
CLAUDE.md behavioral contract  ✅ Strong     State machine + approval template + stop conditions
Commands (implement, verify)   ⚠️  Partial   Phased workflow but no explicit goal articulation per phase
Agent prompts (6 agents)       ⚠️  Partial   Structured checklists but no situation assessment step
Knowledge skills (21 skills)   ⚠️  Partial   Multi-step workflows but imperative rather than STAR
USV pipeline parameters        ❌ None       Implicit constraint reasoning in 14+ parameters
Cloudy Claude API prompts      ❌ None       Pure role + instruction (covered in v1 plan)
```

**The pattern:** Reasoning quality degrades the further you get from CLAUDE.md. The behavioral contract forces Claude to articulate intent before acting. The skills and agents say "do X" without forcing "first understand WHY you're doing X."

---

## Priority Tiers

### Tier 1 — Highest Impact (do first)
These involve **implicit physical/mathematical constraint reasoning** — the exact failure mode from the car wash paper.

1. **DSP Reviewer agent** — reviews code that encodes physical constraints
2. **Detection Validator agent** — validates parameter choices with implicit DSP reasoning
3. **Master Reviewer agent** — needs to catch constraint violations across 6 categories

### Tier 2 — High Impact (do second)
These involve **semantic judgment** where forced articulation prevents false positives.

4. **reduce skill** — extraction quality depends on recognizing what IS vs ISN'T a distinct insight
5. **reflect skill** — connection quality depends on articulating WHY notes relate
6. **verify skill** — cold-read test is already STAR-like, but schema/health checks aren't

### Tier 3 — Medium Impact (do third)
These benefit from STAR but already have decent scaffolding.

7. **implement command** — Phase 1 plan mode needs STAR for constraint identification
8. **architect skill** — 7-phase workflow could use STAR at the recommendation step
9. **ask skill** — query routing could use STAR for disambiguation

### Tier 4 — Cloudy Claude API prompts (already covered in v1 plan)

10. Parts Finder fallback
11. Notion atomize, link, tag

---

## Tier 1: Agent Prompt Rewrites

### 1.1 DSP Reviewer — `.claude/agents/dsp-reviewer.md`

**Current problem:** The prompt lists what to check (FFT size, bin indexing, dB conversion) but doesn't force the reviewer to first articulate the physical constraints that the code MUST satisfy. This is the car wash problem — the model knows Nyquist exists but doesn't activate that knowledge unless forced.

**Add this section BEFORE "Review Focus":**

```markdown
## Reasoning Protocol (apply to EVERY finding)

Before reporting any issue, articulate your reasoning using this structure:

**SITUATION:** What physical/mathematical constraint applies here? State it precisely.
(Example: "At sample rate 300,000 Hz, Nyquist frequency is 150,000 Hz. The USV band
is 25,000–110,000 Hz, well within Nyquist. Any frequency parameter exceeding 150,000 Hz
is physically meaningless.")

**TASK:** What must this code accomplish given that constraint?
(Example: "The frequency-to-bin conversion must map 25,000 Hz to bin index
floor(25000 * n_fft / sample_rate) = floor(25000 * 512 / 300000) = bin 42.")

**ACTION:** What did the code actually do? Quote the specific line(s).

**RESULT:** Does the code satisfy the constraint? If not, what's the concrete impact?
(Example: "Off-by-one error means bin 43 is used, excluding frequencies 24,414–25,000 Hz.
This drops ~586 Hz of the USV band — approximately 0.7% of range, LOW severity but
should be fixed.")

This prevents the failure mode where you flag "check frequency handling" without
computing whether the actual values are correct for THIS specific sample rate and
FFT configuration.
```

**Also add to "Review Focus" item 1 (Mathematical correctness):**

```markdown
1. **Mathematical correctness**
   - Verify FFT size calculations
   - Check for off-by-one errors in bin indexing
   - Validate dB conversion formulas (10*log10 vs 20*log10)
   - **CRITICAL: Compute expected values yourself before checking code.**
     Don't just verify the formula looks right — plug in the actual
     parameters (sr=300000, n_fft=512, hop=128) and confirm the output
     matches what the code produces.
```

### 1.2 Detection Validator — `.claude/agents/detection-validator.md`

**Current problem:** The validation steps are generic ("Check algorithm correctness"). For detection parameter changes, the validator needs to reason about the PHYSICAL meaning of each parameter.

**Replace the current "Validation Steps" section with:**

```markdown
## Validation Protocol

For EVERY parameter change, apply this reasoning chain:

### Step 1: SITUATION — Physical Meaning
What does this parameter control physically? What happens to the audio signal
when this value changes?

| Parameter | Physical meaning |
|-----------|-----------------|
| energy_threshold_db | Minimum signal strength to consider as potential USV. Lower = more candidates (higher recall, lower precision) |
| min/max_duration_ms | Mouse USVs are 10-300 ms. Anything outside this range is likely noise or artifact |
| merge_gap_ms | Bridges brief energy dips within a single call. Too small = splits one USV into two. Too large = merges two distinct USVs |
| segment_continuity_max_gap_ms | The KEY parameter — how long an energy dip can last before we consider it a new syllable vs a dip within one syllable |
| max_bandwidth_hz | USVs are narrow-band (5-15 kHz). Broadband signals (>20 kHz bandwidth) are noise |
| freq_min/max_hz | Species-specific vocalization range. Below 25 kHz = audible range artifacts. Above 110 kHz = unlikely for Mus musculus |

### Step 2: TASK — What Must Be True
Given the physical meaning, what constraints must this parameter satisfy?
Cross-check against:
- Vault baselines (89.7% precision, 93.8% recall at threshold 0.05)
- ADR-001 (sr=300000) and ADR-002 (n_fft=512, hop=128)
- Any notes in `notes/detection.md` topic map

### Step 3: ACTION — Verify the Change
- Does the new value still satisfy the constraints from Step 2?
- Run detection tests with the change
- Compare detection counts before/after on test data if available

### Step 4: RESULT — Impact Assessment
- Quantify the expected impact on precision/recall
- Flag if the change moves away from established baselines without justification
- Report as PASS/FAIL with severity
```

### 1.3 Master Reviewer — `.claude/agents/master-reviewer.md`

**Current problem:** The 9-step workflow is strong but doesn't require the reviewer to articulate WHAT constraints apply before checking WHETHER they're satisfied. Steps 1-4 gather context, Step 5 checks for problems — but Step 5 jumps straight to checking without a "here's what I expect to find" step.

**Add between current Step 4 (Run tests) and Step 5 (Check for problems):**

```markdown
### 4.5 Articulate Expected Constraints (before looking for problems)

Before checking for problems, write down what you EXPECT to be true based
on your reading of the ROADMAP, DECISIONS, patterns, and vault:

- What DSP parameters must this module use? (cite ADR numbers)
- What data flow pattern should it follow? (cite patterns.md)
- What invariants must hold? (cite vault notes if they establish baselines)
- What are the most likely failure modes for this type of module?

This step prevents the reviewer failure mode where you read the code and
nod along because it "looks reasonable" without checking it against the
actual specification. Write your expectations BEFORE reading the implementation.
```

---

## Tier 2: Knowledge Pipeline Skill Enhancements

### 2.1 reduce — Extraction Constraint Reasoning

The reduce skill is already your most sophisticated prompt (~500 lines). Its core problem isn't missing structure — it's that the **selectivity gate** (what to extract vs skip) requires implicit constraint reasoning about domain relevance.

**Add to the EXECUTE NOW section, right after "Read target source":**

```markdown
### Extraction Constraint Assessment (before extracting anything)

SITUATION: What domain is this source from? What is the source type
(paper, lecture notes, documentation, experiment log)? What extraction
categories from ops/config.yaml apply?

TASK: Based on the domain and source type, what kinds of insights should
you expect to find? Set your extraction expectations:
- Paper → findings, methods, hypotheses, baselines, open-questions
- Lecture notes → concepts, mechanisms, definitions
- Experiment log → findings, decisions, baselines
- Documentation → methods, patterns, decisions

ACTION: Now extract. For each candidate insight, test it against
your pre-set expectations. If it doesn't fit any expected category,
it's either (a) a surprising cross-domain finding (EXTRACT — these are
the most valuable) or (b) genuinely off-topic (skip with justification).

RESULT: Your skip rate should be < 10% for domain-relevant sources.
If you're skipping > 10%, re-examine your situation assessment —
you may have miscategorized the source domain.
```

### 2.2 reflect — Connection Articulation Strengthening

The reflect skill already has an "articulation test" which is essentially STAR's Result step. But it's applied at the end (when evaluating connections) rather than at the beginning (when identifying what to look for).

**Add to the EXECUTE NOW section, before "Dual discovery":**

```markdown
### Pre-Reflection Assessment

SITUATION: What domain is this note in? What are its 2-3 core claims
or concepts? What other domains or concepts SHOULD connect to these
claims based on your understanding of the knowledge graph?

TASK: You're looking for connections that a student would benefit from
seeing. Set your connection expectations:
- What notes would EXTEND this claim?
- What notes would CONTRADICT this claim?
- What notes share an underlying MECHANISM with this claim?
- What notes from OTHER DOMAINS apply the same principle?

This pre-assessment prevents the failure mode where semantic search
returns superficially similar notes and you accept them without asking
"does this ACTUALLY help understanding?"

Now proceed to dual discovery, but evaluate every candidate against
your pre-set expectations. Unexpected connections that pass the
articulation test are the most valuable finds.
```

### 2.3 verify — STAR for Schema and Health Checks

The recite phase (cold-read prediction) is already STAR-like. But schema validation and health checks are currently checklist-based.

**Add STAR framing to the schema validation phase:**

```markdown
### Schema Validation — Constraint Reasoning

SITUATION: What type of note is this? (finding, method, hypothesis, etc.)
What required fields does this type demand per the schema?

TASK: Check each required field. But don't just check presence — check
QUALITY. A description field that says "about X" fails the cold-read
test even though it's technically present.

ACTION: For each field, assess whether it would survive a context wipe —
would someone reading ONLY this note (not the source, not surrounding notes)
understand what it claims?

RESULT: PASS only if every required field is both present AND sufficient
for standalone comprehension.
```

---

## Tier 3: Command and Workflow Enhancements

### 3.1 implement — Phase 1 Constraint Identification

**Add to Phase 1: PLAN, after step 7 ("Note edge cases, DSP parameters, integration points"):**

```markdown
8. **Constraint articulation** — Before writing the plan, explicitly list:
   - Physical constraints (sample rates, frequency ranges, Nyquist limits)
   - Architectural constraints (from DECISIONS.md ADRs)
   - Pattern constraints (from patterns.md)
   - Data flow constraints (what data structures must flow between components)
   
   The plan MUST reference each constraint and explain how the
   implementation will satisfy it. A plan without explicit constraint
   listing is incomplete — request this from the implementer.
```

### 3.2 review-all — Constraint Pre-Check

**Add before step 2 ("Spawn master-reviewer"):**

```markdown
1.5 **Pre-review constraint compilation**
   Before spawning any reviewer, compile the constraint list for the module:
   - Read DECISIONS.md ADRs that apply
   - Read vault notes for established baselines
   - Read ROADMAP exit criteria
   
   Pass this constraint list to EVERY spawned reviewer as context.
   This ensures reviewers check against the SAME constraint set
   rather than each discovering constraints independently (or missing them).
```

---

## USV Pipeline — Parameter Documentation as Constraint Contracts

This is different from the other optimizations. The USV pipeline's implicit constraints aren't in prompts — they're in code config. But every time Claude reviews, modifies, or discusses these parameters, it needs to reason about physical constraints. The optimization here is to **document the constraints so they're always available as context**.

### Create: `docs/usv-constraint-contracts.md`

```markdown
# USV Detection Pipeline — Constraint Contracts

These constraints MUST be verified whenever the associated parameter is changed.
Reviewers (DSP, detection-validator, master-reviewer): check this document.

## Physical Constraints

| Constraint | Value | Source | Violation Impact |
|-----------|-------|--------|-----------------|
| Nyquist limit | f_max < sr/2 = 150,000 Hz | Physics | Aliased frequencies — meaningless data |
| Mouse USV range | 25,000–110,000 Hz | Portfors 2007; Scattoni 2009 | Missing vocalizations or including non-USV |
| USV duration | 10–300 ms typical | Scattoni 2009 | Over/under-detection |
| USV bandwidth | 5–15 kHz typical | Species biology | Broadband = noise |
| Temporal resolution | hop/sr = 0.427 ms at hop=128, sr=300k | STFT math | Limits event timing precision |
| Frequency resolution | sr/n_fft = 586 Hz at n_fft=512, sr=300k | STFT math | Limits frequency discrimination |

## Architectural Constraints (ADRs)

| ADR | Constraint | Rationale |
|-----|-----------|-----------|
| ADR-001 | sr = 300,000 Hz | Recording hardware standard |
| ADR-002 | n_fft=512, hop=128 | Balanced time-frequency resolution for USV characteristics |

## Calibration Baselines

| Metric | Value | Conditions | Date |
|--------|-------|-----------|------|
| Precision | 89.7% | threshold=0.05, full retrain dataset | 2026-02 |
| Recall | 93.8% | threshold=0.05, full retrain dataset | 2026-02 |
| F1 | 91.7% | threshold=0.05, full retrain dataset | 2026-02 |

## Parameter Dependency Graph

energy_threshold_db ←→ precision/recall tradeoff
    ↓
merge_gap_ms ←→ single-USV integrity
    ↓
segment_continuity_max_gap_ms ←→ syllable boundary detection
    ↓
max_bandwidth_hz ←→ noise rejection

Changing any parameter in this chain may affect downstream parameters.
Always re-evaluate the full chain after a change.

## Known Inconsistencies

| Issue | Status | Impact |
|-------|--------|--------|
| SpectrogramConfig expects 250 kHz, DetectionConfig expects 300 kHz | Documented, unresolved | Config mismatch on sample rate |
| VQ-VAE frequency range (20-120 kHz) differs from detection (25-110 kHz) | By design | Different processing stages use different ranges |
```

### Wire this into agent prompts

Add to dsp-reviewer.md, detection-validator.md, and master-reviewer.md:

```markdown
## Constraint Contracts
Before reviewing, read `docs/usv-constraint-contracts.md` for the full set of
physical, architectural, and calibration constraints that apply to this codebase.
Every finding should reference which constraint it relates to.
```

---

## Implementation Sequence

### Phase 1: Constraint Contracts Document (30 min)
Create `docs/usv-constraint-contracts.md` as above. This is foundational —
everything else references it.

### Phase 2: Agent Prompt Rewrites (45 min)
Apply STAR reasoning protocol to:
- dsp-reviewer.md (Section 1.1)
- detection-validator.md (Section 1.2)
- master-reviewer.md (Section 1.3)

### Phase 3: Knowledge Skill Enhancements (30 min)
Add STAR pre-assessment blocks to:
- reduce (Section 2.1)
- reflect (Section 2.2)
- verify (Section 2.3)

### Phase 4: Command Enhancements (15 min)
Add constraint articulation to:
- implement.md (Section 3.1)
- review-all.md (Section 3.2)

### Phase 5: Cloudy Claude API Prompts (from v1 plan)
Apply v1 plan changes to:
- Parts Finder fallback.py
- Notion tag.py, atomize.py, link.py

### Phase 6: Evaluation
For each tier, pick 3-5 representative tasks and run them with/without the STAR
additions. Compare output quality blind. The most important metric per tier:

| Tier | Key metric |
|------|-----------|
| Agent prompts | False negative rate (constraints the reviewer MISSED) |
| Knowledge skills | Skip rate for reduce; false positive rate for reflect |
| Commands | Plan completeness (does the plan list all relevant constraints?) |
| API prompts | Hallucination rate (Parts Finder); over-split rate (atomize) |

---

## Quick-Start for Claude Code

> Read this plan file. Execute phases 1-4 in order:
>
> Phase 1: Create `docs/usv-constraint-contracts.md` with the content specified in the "USV Pipeline — Parameter Documentation" section.
>
> Phase 2: Edit the three agent files in `.claude/agents/` (dsp-reviewer.md, detection-validator.md, master-reviewer.md) by adding the STAR reasoning sections specified in Tier 1.
>
> Phase 3: Edit the three skill files in `.claude/skills/` (reduce, reflect, verify) by adding the STAR pre-assessment sections specified in Tier 2.
>
> Phase 4: Edit implement.md and review-all.md in `.claude/commands/` by adding the constraint articulation steps specified in Tier 3.
>
> After ALL changes: run the test suite to confirm nothing breaks. Report what was changed.
>
> Do NOT modify CLAUDE.md, ops/ files, or any Python source code.
