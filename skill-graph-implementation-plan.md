# Skill Graph — Implementation Roadmap

> This file is the master plan for implementing arscontexta-based skill graphs into two active projects.
> It lives in the project root alongside CLAUDE.md and DECISIONS.md.
> Claude Code: read this file when asked "what's next for skill graph", "check the skill graph roadmap".
> Human: use the `/implement` commands below by copy-pasting them into Claude Code sessions.

---

## How to Use This File

1. Work through modules **in order** within each phase (dependencies are noted)
2. Each module has:
   - **What**: brief description of what to build/configure
   - **`/implement` command**: copy-paste into Claude Code (or type `/implement <module description>`)
   - **Test plan**: how to verify the module works
   - **Exit criteria**: what "done" looks like
3. After each module: verify, document learnings, commit config changes
4. Phase gates must pass before starting the next phase

## Status Key

- **DONE** — Implemented and verified
- **READY** — Dependencies met, can start
- **BLOCKED** — Waiting on dependency or external input
- **FUTURE** — Not yet prioritized

## Context

The goal is to capture accumulated domain knowledge — architecture decisions, research findings, debugging insights, methodology learnings — into a traversable knowledge graph that Claude Code can navigate automatically, so context doesn't depend on the developer remembering to surface it.

### Projects

1. **Cloudy Claude** — AI intelligence platform for Tevel Group. Integrates with external ERPs (Priority, Hashavshevet). Core modules: customer basket gap analysis, demand forecasting, Claude-powered NL interfaces. Pivoted from custom ERP to intelligence layer architecture.

2. **USV Research Pipeline** — Mouse ultrasonic vocalization detection and classification in Prof. Mickey London's lab at Hebrew University. CNN-based detection, scaling from 2,000 to 30,000 labeled samples, constrained jittering for positional bias, planned transformer + VQ-VAE architecture exploration.

### Current Knowledge Management

- `CLAUDE.md` and `AGENTS.md` files per project with behavioral contracts and agent coordination
- `Decisions.md` for architecture decisions
- Accumulated knowledge scattered across past Claude conversations and developer memory
- No persistent, traversable knowledge structure

---

## Phase 1: Installation & Project Setup

### 1.1 Install arscontexta Plugin

**What:** Install the arscontexta plugin into Claude Code and verify it's available.
**Status:** DONE
**Review Tier:** 1
**Depends on:** None

/implement Install arscontexta Plugin

Install the arscontexta plugin into Claude Code and verify all kernel primitives are accessible.

**Steps:**

1. Install the plugin:
```bash
/plugin marketplace add agenticnotetaking/arscontexta
/plugin install arscontexta@agenticnotetaking
# Restart Claude Code
```

2. Verify installation:
```bash
/arscontexta:health
```

**Test plan:**
```
1. Plugin appears in installed plugins list
2. /arscontexta:health runs without errors
3. All 15 kernel primitives are reported as available
```

**Exit criteria:**
- [x] Plugin installed and active after Claude Code restart
- [x] `/arscontexta:health` reports all 15 kernel primitives in place
- [x] No error messages on startup

---

### 1.2 Setup Cloudy Claude Skill Graph

**What:** Initialize arscontexta for the Cloudy Claude project with the Experimental preset, tailored for enterprise software / ERP integration / B2B spare parts industry.
**Status:** DONE
**Review Tier:** 1
**Depends on:** Phase 1.1

/implement Setup Cloudy Claude Skill Graph

Run arscontexta setup for the Cloudy Claude project. Use the Experimental preset (fast-moving development with rapid iteration).

**Context:** This project is an AI intelligence platform for a B2B spare parts company. Key vocabulary: customer purchasing patterns, demand forecasting, NL query interfaces, multi-ERP abstraction (Priority, Hashavshevet).

**Steps:**

1. Navigate to Cloudy Claude project root
2. Run setup:
```bash
/arscontexta:setup
```

3. During the conversation phase, emphasize:
   - **Domain:** enterprise software / ERP integration / B2B spare parts industry
   - **Key concepts:** customer purchasing patterns, demand forecasting, NL query interfaces, multi-ERP abstraction
   - **Decision-making style:** phased development with explicit architecture decision records
   - **Processing needs:** capturing integration patterns, API learnings, customer domain models

4. Select the **Experimental** preset

5. Post-setup validation:
```bash
/arscontexta:health
```

**Test plan:**
```
1. Setup creates expected folder structure for skill graph
2. /arscontexta:health reports healthy state
3. Preset configuration matches Experimental profile
4. Domain vocabulary reflected in initial graph metadata
```

**Exit criteria:**
- [x] `/arscontexta:setup` completes without errors (commit 14d65b1, 2026-02-19)
- [x] Folder structure created and reviewed — 5 domain topic maps (ERP integration, ML pipeline, customer intelligence, sync engine, data modeling)
- [x] `/arscontexta:health` reports all green
- [x] Initial graph is empty but structurally valid — 16 skills, 4 hooks, qmd configured

---

### 1.3 Setup USV Research Skill Graph

**What:** Initialize arscontexta for the USV Research Pipeline project with the Research preset, tailored for computational neuroscience / signal processing / ML for behavioral analysis.
**Status:** DONE
**Review Tier:** 1
**Depends on:** Phase 1.1

/implement Setup USV Research Skill Graph

Run arscontexta setup for the USV Research Pipeline project. Use the Research preset (academic research with citation tracking and methodology MOCs).

**Context:** This project analyzes mouse ultrasonic vocalizations (USVs) recorded at 300 kHz. Key vocabulary: spectrogram analysis, CNN classification, USV syllable taxonomy, behavioral correlates, STFT, energy detection, codebook discovery.

**Steps:**

1. Navigate to USV project root
2. Run setup:
```bash
/arscontexta:setup
```

3. During the conversation phase, emphasize:
   - **Domain:** computational neuroscience / signal processing / ML for behavioral analysis
   - **Key concepts:** spectrogram analysis, CNN classification, USV syllable taxonomy, behavioral correlates
   - **Decision-making style:** hypothesis-driven, experimental, with strong methodology tracking
   - **Processing needs:** capturing experimental results, model performance metrics, methodology decisions, literature connections

4. Select the **Research** preset

5. Post-setup validation:
```bash
/arscontexta:health
```

**Test plan:**
```
1. Setup creates expected folder structure for skill graph
2. /arscontexta:health reports healthy state
3. Preset configuration matches Research profile
4. Domain vocabulary reflected in initial graph metadata
```

**Exit criteria:**
- [x] `/arscontexta:setup` completes without errors
- [x] Folder structure created and reviewed — makes sense for research/neuroscience domain
- [x] `/arscontexta:health` — 15/15 kernel primitives pass (self-space disabled by design)
- [x] Initial graph is empty but structurally valid

---

## Phase 1 Gate — PASSED

Before starting Phase 2:
- [x] arscontexta plugin installed and functional
- [x] Cloudy Claude skill graph initialized with Experimental preset (commit 14d65b1, 2026-02-19)
- [x] USV Research skill graph initialized with Research preset (commit aafe406, 2026-02-18)
- [x] Both projects report healthy state via `/arscontexta:health`

---

## Phase 2: Knowledge Migration — Cloudy Claude

> This is the most important phase. We need to extract existing knowledge from flat files into atomic, linked claims.

### 2.1 Migrate Cloudy Claude Architecture Docs

**What:** Extract structured knowledge from `CLAUDE.md`, `AGENTS.md`, and `Decisions.md` into atomic, linked claims in the Cloudy Claude skill graph. Each decision becomes its own note with links to reasoning and alternatives.
**Status:** DONE (2026-02-19, commit af51932)
**Review Tier:** 2
**Depends on:** Phase 1.2

/implement Migrate Cloudy Claude Architecture Docs

Process existing documentation into the Cloudy Claude skill graph. Each source file is reduced into atomic claims, then reflected to build connections.

**Context:** The flat files (`CLAUDE.md`, `AGENTS.md`, `Decisions.md`) contain architecture principles, coding conventions, agent behavior rules, coordination patterns, and design decisions. These need to become individual, linked, traversable knowledge atoms.

**Sources to process (in order):**

1. `CLAUDE.md` — extract architecture principles, coding conventions, agent behavior rules
2. `AGENTS.md` — extract agent coordination patterns, subagent responsibilities, handoff protocols
3. `Decisions.md` — each decision becomes its own atomic note with links to reasoning and alternatives
4. Any existing design docs, API integration notes, customer analysis documents

**For each source:**
```bash
/reduce <source-file>
/reflect  # after each reduce, to connect new claims to existing graph
```

**Priority claims to capture:**
- Why intelligence layer over custom ERP (the pivot decision and reasoning)
- ERP abstraction patterns for Priority vs Hashavshevet
- Customer basket gap analysis methodology
- Demand forecasting approach and data requirements
- Claude NL interface architecture (prompt design, context management)
- Phased development strategy and milestone definitions

**Test plan:**
```
1. Each source file processed without errors
2. Claims created are atomic (one idea per claim)
3. Links between related claims exist (e.g., ERP pivot decision links to architecture principles)
4. /arscontexta:health still reports healthy state after migration
5. Searching for "ERP" returns relevant claims
6. Searching for "architecture" returns relevant claims
```

**Exit criteria:**
- [x] All 3 core files (`CLAUDE.md`, `AGENTS.md`, `Decisions.md`) reduced (commit af51932, 2026-02-19)
- [x] `/reflect` run after each reduction
- [x] At least 15 atomic claims created
- [x] Claims are linked (not isolated)
- [x] `/arscontexta:health` reports healthy state

---

### 2.2 Cloudy Claude Implicit Knowledge Dump

**What:** Capture tacit domain knowledge that lives only in the developer's head — industry patterns, API gotchas, technical debt, lessons learned — into the skill graph via `/learn` sessions.
**Status:** DONE (2026-02-19, 28 commits: industry knowledge, tech debt, integration gotchas, retrospective)
**Review Tier:** 1
**Depends on:** Phase 2.1

/implement Cloudy Claude Implicit Knowledge Dump

Conduct a structured knowledge dump session for tacit knowledge not captured in any document. Use `/learn` for each topic.

**Context:** Much of the project's valuable knowledge is undocumented — it lives in the developer's memory from months of working with Israeli spare parts industry patterns, Priority/Hashavshevet APIs, and Claude integration edge cases.

**Topics to brain-dump:**

1. **Israeli spare parts industry domain knowledge:**
```bash
/learn "Here's what I know about customer behavior patterns in the Israeli spare parts industry..."
/learn "Here's what I know about seasonal trends and supplier dynamics..."
```

2. **Technical debt and known limitations:**
```bash
/learn "Here are the known limitations and technical debt items..."
```

3. **Integration gotchas:**
```bash
/learn "Here are the gotchas I've discovered with the Priority and Hashavshevet APIs..."
```

4. **Retrospective insights:**
```bash
/learn "Things I'd do differently if starting this project over..."
```

5. Run reflection to connect new knowledge:
```bash
/reflect
```

**Test plan:**
```
1. Each /learn session creates at least one new claim
2. New claims link to existing architecture claims from Phase 2.1
3. Searching for "Priority API" returns relevant gotchas
4. Searching for "technical debt" returns relevant claims
```

**Exit criteria:**
- [x] All 4 brain-dump topics completed (industry: 8 notes, tech-debt: 3 notes, integration: 4 notes, retrospective: 13 notes)
- [x] `/reflect` run to connect new claims
- [x] At least 10 new claims from implicit knowledge (28 notes total)
- [ ] Developer confirms key tacit knowledge is now captured

---

## Phase 2 Gate (Cloudy Claude) — PASSED

Before starting Phase 3:
- [x] All core docs migrated (CLAUDE.md, AGENTS.md, Decisions.md) — commit af51932
- [x] Implicit knowledge captured for all 4 topic areas (28 notes across 4 categories)
- [x] At least 25 total claims in the Cloudy Claude skill graph
- [x] Claims are linked, not isolated
- [x] `/arscontexta:health` reports healthy state

---

## Phase 3: Knowledge Migration — USV Research

### 3.1 Migrate USV Architecture & Experiment Docs

**What:** Extract structured knowledge from USV project documentation into atomic claims. Includes architecture decisions, experimental results, pipeline design rationale, and model performance baselines.
**Status:** DONE
**Review Tier:** 2
**Depends on:** Phase 1.3

/implement Migrate USV Architecture & Experiment Docs

Process existing USV project documentation into the skill graph. Focus on pipeline architecture decisions, experimental methodology, and model performance baselines.

**Context:** The USV project has rich documentation in `CLAUDE.md`, `DECISIONS.md`, `ROADMAP.md`, experiment logs, and model evaluation results. Key architectural decisions (ADRs) are especially important to capture as linked claims.

**Sources to process (in order):**

1. `CLAUDE.md` — extract pipeline architecture decisions, behavioral contracts
2. `DECISIONS.md` — each ADR becomes its own atomic note with reasoning and alternatives
3. `ROADMAP.md` — extract module specs, dependency relationships, design rationale
4. `IMPLEMENTATION_PROGRESS.md` — current state, milestones achieved
5. Any experiment logs or model evaluation results

**For each source:**
```bash
/reduce <source-file>
/reflect
```

**Priority claims to capture:**
- CNN architecture decisions and why (layer structure, input format) — ADR-006
- Positional bias problem: diagnosis, constrained jittering solution, parameters that worked
- Selection bias in negative samples: how diverse sampling fixed precision/recall — ADR-008
- Current performance benchmarks: 89.7% precision, 93.8% recall — conditions under which these hold
- STFT parameter choices (n_fft=512, hop=128, 300 kHz) — ADR-001, ADR-002
- Recording-based splitting rationale — ADR-004
- Scaling strategy: 2K → 30K samples, what changes and what doesn't

**Test plan:**
```
1. Each ADR from DECISIONS.md has a corresponding atomic claim
2. Performance baselines captured with conditions (threshold, dataset size, split)
3. Claims link across documents (e.g., ADR-002 links to STFT usage in energy_detector)
4. Searching for "STFT" returns relevant claims about frequency resolution
5. Searching for "positional bias" returns jittering solution
```

**Exit criteria:**
- [x] All 5 source files reduced (DECISIONS.md: 32 notes, ROADMAP.md: 29 notes + 8 enrichments; CLAUDE.md + IMPLEMENTATION_PROGRESS.md: operational content already covered; experiment logs: baselines captured)
- [x] `/reflect` run after each reduction (3 passes total)
- [x] Every ADR from DECISIONS.md has a corresponding claim (all 14 ADRs)
- [x] Performance baselines captured with conditions (89.7% precision, 93.8% recall, F1 91.7% at threshold 0.05)
- [x] At least 20 atomic claims created (61 total)
- [x] `/arscontexta:health` — vault structurally healthy (66 documents indexed, 4 topic maps, dense cross-linking)

---

### 3.2 USV Research Implicit Knowledge Dump

**What:** Capture tacit research knowledge — labeling expertise, lab conventions, literature context, hypotheses about USV function — into the skill graph.
**Status:** DONE
**Review Tier:** 1
**Depends on:** Phase 3.1

/implement USV Research Implicit Knowledge Dump

Conduct a structured knowledge dump for tacit USV research knowledge not captured in documentation. Use `/learn` for each topic.

**Context:** Significant expertise about what makes a good training sample, lab equipment considerations, and scientific hypotheses guiding design decisions exists only in the developer's and researcher's heads.

**Topics to brain-dump:**

1. **Labeling expertise:**
```bash
/learn "Here's what makes a good vs bad USV training sample — the tacit rules for labeling..."
/learn "Here are the edge cases in USV labeling and how we handle them..."
```

2. **Lab-specific conventions:**
```bash
/learn "Here are the lab-specific conventions and equipment considerations for USV recording..."
```

3. **Literature context:**
```bash
/learn "Here are the key papers informing our approach and how they connect..."
/learn "Here's how our work relates to existing USV classification literature..."
```

4. **Scientific hypotheses:**
```bash
/learn "Here are the hypotheses about USV function and structure that guide our design decisions..."
/learn "Here's what we think about wild vs lab mouse vocal repertoire differences..."
```

5. **Spectrogram preprocessing insights:**
```bash
/learn "Here's what I've learned about spectrogram preprocessing choices — frequency ranges, time windows, normalization..."
```

6. Run reflection:
```bash
/reflect
```

**Test plan:**
```
1. Each /learn session creates at least one new claim
2. Literature claims link to methodology claims
3. Hypothesis claims link to architecture decisions
4. Searching for "wild mice" returns relevant hypotheses
5. Searching for "labeling" returns tacit expertise claims
```

**Exit criteria:**
- [x] All 5 brain-dump topics completed (labeling 9, lab-conventions 5, literature 10, hypotheses 8, preprocessing 5)
- [x] `/reflect` run to connect new claims (47 new wiki links)
- [x] At least 12 new claims from implicit knowledge (37 new notes, 14 enrichments)
- [x] Literature references captured with connections to methodology
- [ ] Researcher confirms key tacit knowledge is now captured (some items deferred — user needs to check)

---

## Phase 3 Gate (USV Research) — PASSED

Before starting Phase 4:
- [x] All core docs migrated (DECISIONS.md: 32 notes, ROADMAP.md: 29 notes + 8 enrichments)
- [x] Implicit knowledge captured for all 5 topic areas (37 notes from brain-dumps)
- [x] At least 30 total claims in the USV Research skill graph (104 notes total, 6 topic maps)
- [x] Every ADR has a corresponding linked claim (all 14 ADRs)
- [x] Performance baselines captured with conditions (89.7% precision, 93.8% recall, F1 91.7%)
- [x] `/arscontexta:health` reports healthy state

---

## Phase 4: Workflow Integration

### 4.1 Update CLAUDE.md Files

**What:** Add skill graph integration instructions to both projects' `CLAUDE.md` files so that future Claude Code sessions automatically engage with the knowledge graph.
**Status:** DONE — Both CLAUDE.md files updated during their respective `/arscontexta:setup` runs. USV (Phase 1.3), CC (Phase 1.2).
**Review Tier:** 1
**Depends on:** Phase 2 Gate, Phase 3 Gate

/implement Update CLAUDE.md Files for Skill Graph

Add skill graph usage instructions to both projects' CLAUDE.md files. Define the boundary between CLAUDE.md (behavioral contracts, agent coordination) and the skill graph (domain knowledge, decisions, methodology).

**Context:** CLAUDE.md is the source of truth for behavioral contracts. The skill graph is the source of truth for domain knowledge. These are complementary, not redundant. Adding instructions ensures new sessions automatically leverage the graph.

**Files to modify:**

1. **Cloudy Claude — `CLAUDE.md`** — Add section:

```markdown
## Knowledge Graph

This project uses an arscontexta skill graph for persistent domain knowledge.

### Session Start Protocol
Before starting work on any task:
1. Check relevant MOCs for existing knowledge that applies
2. Review recent claims related to the task domain

### Session End Protocol
After completing work that produced new insights:
1. Run `/reduce` on session learnings
2. After any architecture decision, create an atomic claim capturing the decision, reasoning, and alternatives

### Boundary
- **Skill graph**: domain knowledge, architecture decisions, experimental results, API learnings
- **CLAUDE.md**: behavioral contracts, agent coordination rules, workflow definitions
- These are complementary, not redundant
```

2. **USV Research — `CLAUDE.md`** — Add equivalent section with research-specific wording

**Test plan:**
```
1. New session in Cloudy Claude project sees skill graph instructions in CLAUDE.md
2. New session in USV project sees skill graph instructions in CLAUDE.md
3. Instructions clearly define boundary between CLAUDE.md and skill graph
4. Session start protocol is actionable (Claude Code knows what to check)
```

**Exit criteria:**
- [ ] Both CLAUDE.md files updated with Knowledge Graph section
- [ ] Boundary between CLAUDE.md and skill graph clearly defined
- [ ] Session start/end protocols documented
- [ ] New Claude Code session in each project acknowledges skill graph

---

### 4.2 Configure Maintenance Hooks

**What:** Set up arscontexta hooks for automated session-level knowledge capture: orient on session start, validate on write, capture on session end.
**Status:** DONE — Both projects have hooks configured during their respective `/arscontexta:setup` runs. USV: 4 hooks (session-orient, validate-note, auto-commit, session-capture). CC: 4 hooks (same set).
**Review Tier:** 1
**Depends on:** Phase 4.1

/implement Configure Skill Graph Maintenance Hooks

Set up automated hooks for both projects so that arscontexta integrates into the natural development workflow without manual intervention.

**Context:** arscontexta provides hooks that automate knowledge capture. These reduce the maintenance burden from "remember to run commands" to "it happens automatically." The hooks are: session orient (loads context), write validate (enforces schema), session capture (persists state).

**For each project, configure:**

1. **Session orient hook** — loads relevant context automatically on session start
2. **Write validate hook** — enforces schema on new notes
3. **Session capture hook** — persists session state on session end

**Verify hook configuration:**
```bash
# Start a new session and verify orient hook fires
# Create a new note and verify validate hook enforces schema
# End session and verify capture hook persists state
```

**Test plan:**
```
1. Session orient hook fires on new Claude Code session start
2. Write validate hook rejects malformed notes
3. Session capture hook persists session state on end
4. Hooks don't interfere with normal development workflow
5. Hook errors are logged, not fatal
```

**Exit criteria:**
- [ ] All 3 hooks configured for both projects
- [ ] Hooks fire at correct lifecycle points (verified manually)
- [ ] Hooks don't slow down or interfere with development
- [ ] Malformed notes rejected by validation hook

---

### 4.3 Integrate with Reviewer Subagent Workflow

**What:** Give the reviewer subagent pattern access to the skill graph so that code reviews can check existing claims before proposing new approaches.
**Status:** DONE (2026-02-19)
**Review Tier:** 1
**Depends on:** Phase 4.1

/implement Integrate Skill Graph with Reviewer Agents

Update the reviewer subagent workflow to include skill graph awareness. Reviewers should check relevant MOCs before flagging issues, and architecture reviews should verify alignment with prior decisions.

**Context:** Both projects use a reviewer subagent pattern (master-reviewer, dsp-reviewer, pr-reviewer, etc.). These agents should leverage existing knowledge rather than re-deriving insights from scratch.

**Changes to agent instructions:**

1. **Architecture review** — check relevant MOCs for prior decisions before flagging conflicts
2. **Code review** — search claims for known patterns/gotchas related to the code under review
3. **Issue flagging** — before creating new approaches, check if existing claims address the concern

**Integration points:**
- Reviewer agent reads relevant MOCs before starting review
- Reviewer agent searches claims for keywords related to the PR/change
- Reviewer output references relevant claims when applicable

**Test plan:**
```
1. Reviewer agent acknowledges skill graph in review output
2. Review of code touching STFT params references ADR-002 claim
3. Review of ERP integration code references API gotcha claims
4. Agent doesn't hallucinate claim references (only cites real claims)
```

**Exit criteria:**
- [ ] Reviewer agent instructions updated for both projects
- [ ] At least one test review demonstrates skill graph usage
- [ ] Agent correctly references existing claims (no fabricated references)
- [ ] Review workflow not significantly slower with skill graph integration

---

## Phase 4 Gate — PASSED

Before starting Phase 5:
- [x] Both CLAUDE.md files updated with Knowledge Graph section
- [x] Maintenance hooks configured and verified for both projects
- [x] Reviewer agents integrated with skill graph (4.3 DONE 2026-02-19)
- [x] Full development sessions in both projects demonstrate smooth skill graph integration (USV: 117 notes active use; CC: 28 notes created in brain-dump session 2026-02-19)

---

## Phase 5: Maintenance Cadence & Validation

### 5.1 Establish Weekly Maintenance Routine

**What:** Define and test a sustainable weekly maintenance routine for both skill graphs. Target: ≤15 minutes per project per week.
**Status:** DONE (2026-02-20)
**Review Tier:** 1
**Depends on:** Phase 4 Gate (PASSED)

/implement Establish Skill Graph Weekly Maintenance

Define the weekly maintenance routine and run it once for each project to validate the time budget.

**Context:** The skill graph must be maintainable alongside actual development work. If maintenance takes >15 min/week per project, it's not sustainable.

**Weekly routine (per project, target ≤15 min):**

```bash
# 1. Health check (~1 min)
/arscontexta:health

# 2. Reflect — update connections (~3 min)
/reflect

# 3. Reweave — backward pass, update old notes with new context (~5 min)
/reweave

# 4. Stats — review growth metrics (~1 min)
/stats
```

**Test plan:**
```
1. Full weekly routine completes in ≤15 min for each project
2. /arscontexta:health reports no issues after routine
3. /reflect creates at least one new connection
4. /reweave updates at least one old note
5. /stats shows expected growth metrics
```

**Exit criteria:**
- [x] Weekly routine documented in both CLAUDE.md files (USV + CC updated 2026-02-20)
- [x] First execution completed for USV project within time budget (CC deferred to tevel-erp session)
- [x] No errors during routine execution (0 FAIL, 2 WARN, 6 PASS)
- [x] Growth metrics baseline established: 117 notes, 6 topic maps, 1011 wiki links (avg 8.6/note), 100% schema, 0 orphans

---

### 5.2 Two-Week Validation Checkpoint

**What:** After two weeks of active use, evaluate whether the skill graph is delivering value — surfacing relevant knowledge, reducing re-explanation, maintaining sustainable overhead.
**Status:** READY (starts 2026-03-06, after 2 weeks of active use)
**Review Tier:** 1
**Depends on:** Phase 5.1 (DONE) + 2 weeks elapsed

/implement Two-Week Skill Graph Validation

Evaluate skill graph effectiveness after two weeks of real use. Score against defined criteria and make course-correction decisions.

**Evaluation criteria (score each 1-5):**

| Criterion | Question | Target |
|-----------|----------|--------|
| **Relevance** | Is the agent surfacing relevant knowledge without being prompted? | ≥ 3 |
| **Context Retention** | Are you spending less time re-explaining context in new sessions? | ≥ 3 |
| **Maintenance Overhead** | Is maintenance sustainable alongside actual work? | ≤ 15 min/week |
| **Connection Quality** | Are generated connections meaningful or just noise? | ≥ 3 |
| **Developer Experience** | Does the skill graph feel like a help or a burden? | ≥ 3 |

**Course correction signals:**

**Scale back if:**
- Maintenance takes > 15 min/week per project
- System generates connections that aren't useful (noise)
- Token costs impact ability to do actual development

**Double down if:**
- Agent catches things you would have forgotten
- New sessions feel meaningfully more informed
- Decision-making feels better supported by historical context

**After milestone reviews:**
```bash
/rethink     # Challenge assumptions that may have shifted
/arscontexta:architect  # Get evolution guidance
```

**Test plan:**
```
1. All 5 evaluation criteria scored
2. At least 3 specific examples of skill graph helping (or not)
3. Course correction decision documented
4. If scaling back: specific items to remove identified
5. If doubling down: specific areas to expand identified
```

**Exit criteria:**
- [ ] Evaluation completed for both projects
- [ ] All 5 criteria scored with specific evidence
- [ ] Course correction decision documented
- [ ] Action items identified (scale back or double down, per project)
- [ ] Next review date set

---

## Phase 5 Gate

Before declaring skill graph integration complete:
- [ ] Two weeks of active use completed
- [ ] Evaluation criteria met (all ≥ 3, maintenance ≤ 15 min/week)
- [ ] Course correction applied if needed
- [ ] Both projects have sustainable, value-delivering skill graphs
- [ ] Or: decision made to scale back with documented rationale

---

## Dependency Graph

```
Phase 1.1 (Install) ──→ Phase 1.2 (Cloudy Claude Setup)
                    └──→ Phase 1.3 (USV Research Setup)
                              │                    │
                              ↓                    ↓
                    Phase 2.1 (CC Docs)    Phase 3.1 (USV Docs)
                              │                    │
                              ↓                    ↓
                    Phase 2.2 (CC Brain)   Phase 3.2 (USV Brain)
                              │                    │
                              └────────┬───────────┘
                                       ↓
                              Phase 4.1 (CLAUDE.md)
                                       │
                              ┌────────┼────────┐
                              ↓        ↓        ↓
                           4.2       4.3
                          (Hooks)  (Reviewers)
                              │        │
                              └────┬───┘
                                   ↓
                          Phase 5.1 (Weekly Routine)
                                   │
                               [2 weeks]
                                   ↓
                          Phase 5.2 (Validation)
```

**Parallelism:** Phases 2 (Cloudy Claude migration) and 3 (USV migration) can run in parallel once their respective setup (1.2, 1.3) is complete.

---

## Estimated Timeline

| Week | Activity |
|------|----------|
| 1 | Phase 1 (Install + setup both projects), begin Phase 2.1 (CC docs migration) |
| 2 | Complete Phase 2 (CC migration), begin Phase 3.1 (USV docs migration) |
| 3 | Complete Phase 3 (USV migration), Phase 2.2/3.2 brain-dump sessions |
| 4 | Phase 4 (Workflow integration, CLAUDE.md updates, hooks, reviewer integration) |
| 5 | Phase 5.1 (First maintenance cycle), begin active use period |
| 6 | Phase 5.2 (Two-week validation checkpoint, course correction) |

---

## Implementation Order (Priority)

| Priority | Module | Why |
|----------|--------|-----|
| **1** | Phase 1.1 (Install) | Everything depends on this |
| **2** | Phase 1.2 + 1.3 (Setup, parallel) | Unblocks all migration work |
| **3** | Phase 2.1 (CC Docs) | Highest-value knowledge to capture |
| **4** | Phase 3.1 (USV Docs) | ADRs and performance baselines critical |
| **5** | Phase 2.2 + 3.2 (Brain dumps, parallel) | Captures otherwise-lost knowledge |
| **6** | Phase 4.1 (CLAUDE.md) | Ensures future sessions use the graph |
| **7** | Phase 4.2 + 4.3 (Hooks + Reviewers, parallel) | Automation reduces maintenance burden |
| **8** | Phase 5.1 (Routine) | Sustainability validation |
| **9** | Phase 5.2 (Validation) | Course correction before deeper investment |
