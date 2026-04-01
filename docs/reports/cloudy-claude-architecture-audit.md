# Cloudy Claude Prompt Architecture Audit

> **Audit date:** 2026-02-26
> **Scope:** Full read-only audit of prompt layers, context injection, task taxonomy, orchestration, and configuration.
> **Purpose:** Map current state for optimization using STAR reasoning framework findings (arXiv:2602.21814).

---

## 1. System Prompts & Prompt Layers

### 1.1 Prompt Layering Architecture

Cloudy Claude uses a **6-layer prompt architecture**, each layer enforced by different mechanisms:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 6: Execution Hooks (behavioral guardrails)           │
│  .claude/settings.local.json — gates, validators, auto-commit│
├─────────────────────────────────────────────────────────────┤
│  Layer 5: Knowledge Context (injected per-session)          │
│  ROADMAP.md, DECISIONS.md, patterns.md, module docs         │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Workflow Orchestration (multi-step commands)       │
│  .claude/commands/*.md — implement, verify, commit-push-pr  │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Specialist Agents (role definitions)              │
│  .claude/agents/*.md — dsp-reviewer, master-reviewer, etc.  │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Skills (25+ domain tools)                         │
│  .claude/skills/ — reduce, reflect, reweave, pipeline, etc. │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Behavioral Contract (constitutional rules)        │
│  CLAUDE.md — state machine, stop conditions, red flags      │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Full System Prompts (Verbatim)

#### A. Parts Finder — AI Fallback System Prompt

**File:** `parts-finder/src/parts_finder/api/fallback.py:80-86`

```python
SYSTEM_PROMPT = (
    "You are a vehicle parts specification expert. Given a vehicle's "
    "make, model, year, engine code, and fuel type, provide accurate "
    "maintenance specifications for the requested categories.\n\n"
    "Return ONLY valid JSON matching the exact schema described in the "
    "user prompt. Do not include explanations or markdown formatting."
)
```

**Observations:**
- Pure role-based prompt with output format constraint
- No reasoning scaffold (no STAR, no chain-of-thought, no step-by-step)
- No explicit constraint inference instructions — relies on implicit domain knowledge
- No examples or few-shot demonstrations
- No confidence calibration ("if unsure, return empty")

#### B. Notion Notes — Tagging System Prompt

**File:** `notion_notes/commands/tag.py:52-56`

```python
_SYSTEM_PROMPT = (
    "You are a note categorizer for a university knowledge base. "
    "Your job is to assign exactly ONE domain and up to {max_tags} tags to each note. "
    "Prefer existing tags from the taxonomy. Only suggest new tags when nothing fits."
)
```

**User prompt template** (`tag.py:58-82`):
```
Categorize this note.

## Taxonomy

Domains: {domains}

Existing tags by domain:
{tags_by_domain}

## Note content

Title: {title}

{content}

## Instructions

1. Pick exactly ONE domain from the list above.
2. Pick 1-{max_tags} tags. Prefer existing tags for the chosen domain.
3. If no existing tag fits, suggest a new short tag (2-3 words max).
4. Put new tags in the "new_tags" array (empty if all tags already exist).

Respond with ONLY valid JSON, no markdown fences:
{"domain": "...", "tags": ["..."], "new_tags": ["..."]}
```

**Observations:**
- Structured with clear sections (Taxonomy, Note content, Instructions)
- Numbered step-by-step instructions (partial reasoning scaffold)
- Injects full taxonomy as context (domain names + existing tags per domain)
- Content truncated at 3000 chars for Haiku (`tag.py:263`)
- No STAR framework — no Situation/Task/Action/Result structure
- No explicit constraint reasoning ("given that the taxonomy has X domains, consider...")

#### C. Notion Notes — Linking System Prompt

**File:** `notion_notes/commands/link.py:70-103`

```python
_SYSTEM_PROMPT = (
    "You are analyzing a knowledge base to find genuine semantic connections "
    "between notes. Only identify connections that would be useful for "
    "cross-domain understanding. Skip trivial connections."
)
```

**User prompt template** (`link.py:76-103`):
```
Given this note and these candidates, identify which candidates are genuinely
related in a way useful for cross-domain understanding.

## Target note

Title: {title}

{content}

## Candidates

{candidates}

## Instructions

1. For each candidate that has a genuine semantic connection to the target
   note, identify the connection type and explain it in one sentence.
2. Connection types: {connection_types}
3. Skip trivial connections (e.g., same course, adjacent lecture topics
   with no deeper link).
4. Only include connections that would help a student see cross-cutting
   themes or deeper relationships.
5. If no candidates have genuine connections, return an empty array.

Respond with ONLY valid JSON, no markdown fences:
{"connections": [{"candidate_title": "...", "connection_type": "...",
"explanation": "..."}]}
```

**Connection types:** `same-concept-different-context`, `prerequisite-builds-on`, `contradicts-alternative-view`, `shares-mechanism-principle`, `part-of-same-system`

**Observations:**
- Requires **implicit constraint inference** — must determine "genuine" vs "trivial" connections
- Pre-filtered candidates injected (max 20, sorted by tag overlap)
- Content truncated at 6000 chars
- Has typed connection categories — a good structural scaffold
- Missing: no examples of correct vs incorrect connection identification
- Missing: no reasoning about WHY a connection is genuine (just "explain in one sentence")

#### D. Notion Notes — Atomization System Prompt

**File:** `notion_notes/commands/atomize.py:61-99`

```python
_SYSTEM_PROMPT = (
    "You are processing a student's class notes into atomic concept notes. "
    "Each concept note should be self-contained and make sense without the "
    "surrounding lecture context."
)
```

**User prompt template** (`atomize.py:67-99`):
```
Analyze these class notes and identify distinct concepts, claims, mechanisms,
or ideas. For each one, create an atomic note.

## Taxonomy

Domains: {domains}

Existing tags by domain:
{tags_by_domain}

## Source note

Title: {title}

{content}

## Instructions

For each distinct concept in the source:
1. Give it a concise title (the concept name, not a sentence).
2. Write a clear, self-contained explanation that captures the idea fully.
   Use the student's own phrasing as a foundation but ensure the note makes
   sense on its own. If the notes are brief or shorthand, flesh it out into
   a complete explanation.
3. Pick exactly ONE domain from the taxonomy above.
4. Pick 1-4 tags. Prefer existing tags for the chosen domain.
5. Only create separate notes for genuinely distinct concepts. Don't
   over-split — a single mechanism described in detail is one note, not five.
6. Maximum {max_concepts} concept notes.

Respond with ONLY valid JSON, no markdown fences:
{"concepts": [{"title": "...", "content": "...", "domain": "...", "tags": ["..."]}]}
```

**Observations:**
- Most complex extraction task — requires judgment about "genuinely distinct" concepts
- Explicit anti-over-splitting guidance (instruction 5)
- Content truncated at 8000 chars
- No STAR reasoning — no "first assess the situation, then..." structure
- No examples of good vs bad atomization
- Missing: no guidance for ambiguous cases (overlapping concepts, nested hierarchies)

### 1.3 Behavioral Contract (CLAUDE.md)

**File:** `CLAUDE.md` (366 lines)

This is the **meta-system prompt** governing all Claude Code interactions. Key reasoning scaffolds already in use:

**State Machine (workflow gate):**
```
IDLE → ANALYSIS → APPROVAL_PENDING → EXECUTION → VALIDATION → DONE
```

**Stop Conditions (explicit constraint reasoning):**
- Assumption count ≥3 on critical path → STOP
- Same approach tried twice without new rationale → STOP
- Evidence contradicts hypothesis → STOP

**Test Protocol (decision table — a form of structured reasoning):**

| Code State | Test Result | Action |
|------------|-------------|--------|
| Correct    | Pass        | Good |
| Buggy      | Fail        | Fix code |
| Correct    | Fail        | Discuss |
| Buggy      | Pass        | **DANGEROUS** |

**USV Red Flags (domain-specific constraint triggers):**
- STFT parameter changes without explaining frequency resolution impact
- Detection threshold changes without baseline comparison
- Modifying test expected values to pass
- Any change to `energy_detector.py` without DSP review

**Observations:**
- CLAUDE.md uses **decision tables**, **state machines**, and **stop conditions** — these ARE structured reasoning frameworks, just not labeled as STAR
- The state machine is essentially a STAR variant: Situation (current state) → Task (what's needed) → Action (transition) → Result (next state + validation)
- The approval request template requires: Intent, Context, Scope, Plan, Assumptions, Risks, Validation — a richer reasoning scaffold than STAR
- **This is the strongest reasoning scaffold in the system**, but it only governs the human-in-the-loop workflow, not the API-level prompts

### 1.4 Agent Prompts (Specialist Roles)

6 specialist agents defined in `.claude/agents/`:

| Agent | Model | Purpose |
|-------|-------|---------|
| `dsp-reviewer.md` | Opus | DSP/signal processing mathematical correctness |
| `master-reviewer.md` | Sonnet | Senior review against ROADMAP spec + DECISIONS.md |
| `pr-reviewer.md` | — | Final pre-commit quality review |
| `detection-validator.md` | — | Detection algorithm validation |
| `streamlit-expert.md` | — | Streamlit UI best practices |
| `test-writer.md` | — | Test generation |

**Example — DSP Reviewer prompt** (`dsp-reviewer.md`):
```markdown
You are a specialist in digital signal processing, particularly for audio
analysis at high sample rates (250 kHz).

## Your Expertise
- STFT computation and windowing functions (Hann, Hamming, Blackman)
- FFT bin calculations and frequency resolution
- dB scaling and dynamic range
...

## Review Focus
1. **Mathematical correctness** — FFT size, bin indexing, dB conversion
2. **Frequency handling** — Nyquist, bin conversions, hop size
3. **Numerical stability** — division by zero, epsilon, overflow
4. **Performance** — powers of 2, streaming trade-offs
```

**Example — Master Reviewer prompt** (`master-reviewer.md`):
```markdown
You are the senior technical reviewer for the USV Detection & Analysis project.
Your context is fresh — you have NOT seen the implementation happen.

## Review Categories (in order of importance):
1. DSP CORRECTNESS (most critical)
2. ML RIGOR (second most critical)
3. SPEC COMPLIANCE
4. INTEGRATION CORRECTNESS
5. CODE QUALITY
6. DOCUMENTATION
```

**Observations:**
- Agent prompts are **role-based** with explicit expertise domains
- Master reviewer has a **prioritized review checklist** — another structured reasoning scaffold
- The master reviewer's 9-step protocol IS a STAR-like sequence (Situation: read handoff → Task: understand spec → Action: run tests + review → Result: verdict)
- These are Claude Code subagent prompts, not API-level system prompts

### 1.5 Codex Agent System (Secondary)

**File:** `.codex/skills/implementor-stage-gate/SKILL.md`

A parallel agent system using OpenAI Codex with a stage-gate workflow:
```
1. Read the task brief → identify stages
2. Implement only the first incomplete stage
3. Update implementation notes
4. Ask for confirmation before proceeding
```

**Observations:**
- Confirms **dual-AI strategy exists** (Claude + Codex)
- Codex skills are simpler than Claude skills (4 skills vs 25+)
- Stage-gate pattern = another structured reasoning framework

### 1.6 Dynamic Prompt Templates

All prompts that are filled dynamically per-request:

| Template | Dynamic Variables | File |
|----------|-------------------|------|
| Tag user prompt | `{domains}`, `{tags_by_domain}`, `{title}`, `{content}`, `{max_tags}` | `tag.py:58-82` |
| Link user prompt | `{title}`, `{content}`, `{candidates}`, `{connection_types}` | `link.py:76-103` |
| Atomize user prompt | `{domains}`, `{tags_by_domain}`, `{title}`, `{content}`, `{max_concepts}` | `atomize.py:67-99` |
| Parts Finder prompt | Vehicle make, model, year, engine code, fuel type, unmatched categories + JSON schemas | `fallback.py:190-218` |

---

## 2. Context Injection Pipeline

### 2.1 Parts Finder — Context Injection

**Strategy:** Deterministic database lookup first, AI fallback only for unmatched categories.

```
License Plate (user input)
    │
    ▼
Government API (data.gov.il) → VehicleRecord
    │                            (make, model, year, engine_code, fuel_type)
    ▼
Local SQLite Database → CategoriesResponse (7 categories)
    │
    ├── All matched? → Return database results
    │
    └── Unmatched categories? → AI Fallback
         │
         ▼
    Build prompt with:
      1. Vehicle info (injected from Gov API)         ← BEFORE instructions
      2. "Provide specifications for these categories" ← INSTRUCTION
      3. Per-category JSON schemas (hardcoded)         ← AFTER instructions
      4. Expected output format                        ← END
```

**What gets injected:** Vehicle make, model, year, engine code, fuel type, plus JSON schemas for only the unmatched categories.

**Injection point:** User message only (system prompt is static). Vehicle context appears first, schemas appear after the instruction line.

**Retrieval strategy:** Structured query to Israel Government API (`data.gov.il/api/3/action/datastore_search`), then SQLite lookup. No vector DB, no RAG, no similarity search.

**Prompt construction** (`fallback.py:190-218`):
```python
def _build_prompt(self, vehicle, unmatched):
    lines = [
        f"Vehicle: {make} {model} {vehicle.year}",     # Context first
        f"Engine code: {vehicle.engine_code}",
        f"Fuel type: {vehicle.fuel_type}",
        "",
        "Provide specifications for these categories ONLY:",  # Instruction
    ]
    for cat in unmatched:
        schema = self._CATEGORY_SCHEMAS.get(cat, "{}")
        lines.append(f"  {cat}: {schema}")              # Schema per category
    lines.extend([
        "",
        'Return a single JSON object with category names as keys.',
        'Example: {"oil": {...}, "brakes": {...}}',
        "Only include the categories listed above.",
    ])
```

**Observations:**
- Smart selective prompting — only asks for unmatched categories (50-70% token savings)
- Context (vehicle info) appears BEFORE instructions — this is the conventional pattern
- No RAG, no vector DB — purely structured data injection
- Schemas are hardcoded strings, not dynamically generated from Pydantic models
- **STAR opportunity:** Could benefit from "Situation: This vehicle is a {make} {model} from {year}. Task: Given this vehicle profile, determine the correct specifications for {categories}. Action: Look up specifications considering the engine code and fuel type. Result: Return JSON matching the schema."

### 2.2 Notion Notes — Context Injection

**Strategy:** Taxonomy + existing tags injected into every prompt call.

```
Notion Page (fetched via API)
    │
    ├── Page content → blocks_to_markdown()
    │     (truncated: 3000 chars for tag, 6000 for link, 8000 for atomize)
    │
    ├── Taxonomy (taxonomy.json)
    │     ├── Domain names
    │     └── Existing tags per domain
    │
    └── For linking: candidate pages
          ├── Pre-filtered by tag overlap (min_tag_overlap=1)
          ├── Sorted by overlap score descending
          └── Limited to max_candidates=20
```

**Injection points:**
1. **System prompt:** Role definition + high-level constraints (static)
2. **User prompt — Section 1:** Taxonomy context (injected, dynamic)
3. **User prompt — Section 2:** Page content (injected, truncated)
4. **User prompt — Section 3:** Instructions + output format (static)

**Observations:**
- Taxonomy injection grows over time (tags accumulate) — prompt size increases
- Two-stage filtering for links (fast pre-filter + Claude evaluation) is a good cost optimization
- Pair caching prevents O(n²) re-evaluation
- Content truncation is per-command, not globally consistent (3000/6000/8000)

### 2.3 Claude Code — Context Injection (Meta-Level)

The CLAUDE.md behavioral contract itself is a form of **persistent context injection**. Every Claude Code session inherits:

1. `CLAUDE.md` — behavioral rules, state machine, stop conditions
2. `ops/goals.md` — current goals (via session-orient hook)
3. `ops/reminders.md` — time-bound commitments (via session-orient hook)
4. Knowledge graph notes (via qmd MCP server — semantic search)

**Session orientation hook** (`.claude/hooks/session-orient.ps1`):
- Reads goals, reminders, and vault state at session start
- Injects current context into the conversation

---

## 3. Task Taxonomy

### 3.1 Parts Finder Tasks

| Task | Prompt Template | Requires Implicit Reasoning? | Known Failure Modes |
|------|----------------|------------------------------|---------------------|
| **Vehicle Parts Lookup** | `fallback.py` SYSTEM_PROMPT + dynamic user prompt | **Yes** — must infer correct specs from vehicle profile (make + model + year + engine code → correct viscosity, filter OEM, etc.) | Hallucinated part numbers; wrong viscosity for engine type; JSON parse failures |
| **Miss Logging** | N/A (structured JSONL) | No | Disk full; concurrent writes |

**Examples of correct vs incorrect outputs:**

Correct:
```json
{"oil": {"viscosity": "0W-20", "capacity_l": 4.2, "spec": "API SP", ...}}
```

Incorrect (hallucinated):
```json
{"oil": {"viscosity": "5W-30", "capacity_l": 3.5, "spec": "API SN", ...}}
```
(Wrong viscosity for a vehicle that requires 0W-20 — Claude doesn't have access to the OEM spec database)

### 3.2 Notion Notes Tasks

| Task | Prompt Template | Requires Implicit Reasoning? | Known Failure Modes |
|------|----------------|------------------------------|---------------------|
| **Tag Classification** | `tag.py` system + user prompt | **Moderate** — must match content to taxonomy domains | Assigning wrong domain; suggesting redundant new tags |
| **Atomization** | `atomize.py` system + user prompt | **High** — must identify concept boundaries in unstructured notes | Over-splitting (5 notes from 1 concept); under-splitting (1 note from 5 concepts) |
| **Link Discovery** | `link.py` system + user prompt | **High** — must distinguish genuine vs trivial connections | False positives (trivial links); false negatives (missed deep connections) |
| **Process Pipeline** | `process.py` (chains tag → atomize → link) | Compound | Errors cascade between phases |

### 3.3 Claude Code Agent Tasks (Meta-Level)

| Task | Template | Requires Implicit Reasoning? |
|------|----------|------------------------------|
| DSP Review | `dsp-reviewer.md` | **High** — mathematical correctness |
| Master Review | `master-reviewer.md` | **High** — spec compliance + DSP + ML rigor |
| Implementation | `implement.md` command | **High** — multi-step code generation |
| Knowledge Pipeline | `/reduce`, `/reflect`, `/reweave` skills | **High** — semantic judgment |

---

## 4. Orchestration & Flow

### 4.1 Parts Finder — Request Flow

**Architecture:** Single-endpoint FastAPI service with conditional AI fallback.

```
POST /api/plate-lookup {"plate": "1234567"}
    │
    ▼
┌──────────────┐
│ Plate Format │──invalid──→ 400 Bad Request
│  Validation  │
└──────┬───────┘
       │ valid
       ▼
┌──────────────┐
│ Cache Check  │──hit──→ Return cached response
│  (in-memory) │
└──────┬───────┘
       │ miss
       ▼
┌──────────────┐
│ Gov API Call │──not found──→ 404 Not Found
│ (data.gov.il)│──API error──→ 503 Service Unavailable
└──────┬───────┘
       │ vehicle found
       ▼
┌──────────────┐
│ DB Lookup    │──→ build_response(result)
│ (SQLite)     │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Unmatched categories?│──no──→ Return response (data_source="database")
└──────┬───────────────┘
       │ yes
       ▼
┌──────────────────────┐
│ AI Fallback          │
│ (Claude Haiku)       │──error──→ Return partial response + log miss
│ Retry up to 3x      │
└──────┬───────────────┘
       │ success
       ▼
┌──────────────────────┐
│ Merge AI + DB        │──→ Return response (data_source="hybrid")
│ build_response(      │
│   result, ai_result) │
└──────────────────────┘
```

**Routing logic:** Predetermined — AI fallback only triggered when `unmatched_categories` is non-empty AND `ai_fallback_enabled=True` AND `ANTHROPIC_API_KEY` is set.

**Claude does NOT decide when to call tools.** The route handler (`routes.py:98-107`) makes that decision deterministically.

### 4.2 Notion Notes — Pipeline Flow

**Architecture:** Sequential 3-phase pipeline, single Claude client.

```
Input: list[NotionPage]
    │
    ▼
Phase 1: TAG (Haiku — fast, cheap)
    ├── For each page: fetch content → build prompt → call Claude → parse JSON
    ├── Update Notion properties (Domain, Tags)
    └── Grow taxonomy with new tags
    │
    ▼
Phase 2: ATOMIZE (Sonnet — quality)
    ├── For each page: fetch content → build prompt → call Claude → parse JSON
    ├── Create new concept pages in Notion
    └── Annotate source page + update status
    │
    ▼
Phase 3: LINK (Sonnet — quality)
    ├── Re-fetch all KB pages (to include new atomic notes)
    ├── For each page: pre-filter candidates → build prompt → call Claude
    ├── Create bidirectional relations in Notion
    └── Update pair cache
```

**Orchestration:** Single `Processor.process()` method chains all three phases sequentially. No parallelism. Each phase feeds results to the next.

### 4.3 Claude Code — Agent Orchestration

**Architecture:** Multi-agent system with hooks for behavioral enforcement.

```
Session Start
    │
    ▼
session-orient.ps1 → Load goals, reminders, vault state
    │
    ▼
User Request
    │
    ▼
CLAUDE.md state machine:
    IDLE → ANALYSIS → APPROVAL_PENDING → EXECUTION → VALIDATION → DONE
    │
    ├── Edit/Write attempted? → check_plan_mode.cmd (gate)
    ├── Write to notes/? → validate-note.cmd (schema check)
    ├── Write anywhere? → auto-commit.cmd (async)
    │
    ▼
Specialist Agents (spawned as subagents):
    ├── dsp-reviewer (Opus) — for DSP changes
    ├── master-reviewer (Sonnet) — for module review
    ├── detection-validator — for detection logic
    ├── streamlit-expert — for UI work
    ├── test-writer — for test generation
    └── pr-reviewer — for pre-commit review
    │
    ▼
Session Stop
    ├── check_agents_tag.cmd — verify response includes **Agents:** tag
    └── session-capture.cmd — capture session state
```

### 4.4 Behavioral Contract

**File:** `CLAUDE.md`

This is the **single source of truth** governing Claude's behavior in this project. It defines:
- Priority order: USER LEARNING > QUALITY > INTEGRITY
- Mandatory workflows (plan mode before code)
- Forbidden state transitions (can't skip approval)
- Stop conditions (too many assumptions)
- Domain-specific red flags (DSP parameters)

### 4.5 Dual-AI Strategy

**Confirmed:** The project uses both Claude (Anthropic) and Codex (OpenAI).

| System | Provider | Use Case |
|--------|----------|----------|
| Claude Code | Anthropic (Opus 4.6) | Primary development agent, knowledge management |
| Claude API — Sonnet | Anthropic (claude-sonnet-4-6) | Notion notes atomization, linking |
| Claude API — Haiku | Anthropic (claude-haiku-4-5-20251001) | Notion notes tagging |
| Claude API — Haiku (older) | Anthropic (claude-3-5-haiku-20241022) | Parts Finder AI fallback |
| Codex | OpenAI | Stage-gated implementation tasks |

**Routing logic:** Manual. Claude Code is the primary orchestrator. Codex skills exist in `.codex/skills/` but routing between them is user-initiated, not automated.

---

## 5. Configuration & Parameters

### 5.1 Model Configuration

| Component | Model | File | Line |
|-----------|-------|------|------|
| Claude Code (main agent) | `claude-opus-4-6` | (Claude Code platform) | — |
| DSP Reviewer agent | `opus` | `.claude/agents/dsp-reviewer.md` | 4 |
| Master Reviewer agent | `sonnet` | `.claude/agents/master-reviewer.md` | 4 |
| Notion Notes (default) | `claude-sonnet-4-6` | `notion_notes/config.py` | 20 |
| Notion Notes (tagging) | `claude-haiku-4-5-20251001` | `notion_notes/commands/tag.py` | 31 |
| Notion Notes (linking) | `claude-sonnet-4-6` | `notion_notes/commands/link.py` | 30 |
| Notion Notes (atomize) | `claude-sonnet-4-6` | `notion_notes/commands/atomize.py` | 31 |
| Parts Finder (fallback) | `claude-3-5-haiku-20241022` | `parts-finder/src/parts_finder/config.py` | 33 |

### 5.2 Token & Parameter Settings

| Component | max_tokens | Temperature | top_p | Stop Sequences |
|-----------|-----------|-------------|-------|----------------|
| Notion tag | 256 | SDK default | SDK default | None |
| Notion link | 2048 | SDK default | SDK default | None |
| Notion atomize | 4096 | SDK default | SDK default | None |
| Notion client default | 4096 | SDK default | SDK default | None |
| Parts Finder fallback | 1024 | SDK default | SDK default | None |

**Key finding:** No temperature, top_p, or stop sequence customization anywhere in the codebase. All use Anthropic SDK defaults.

### 5.3 Output Format Constraints

All API prompts constrain output to JSON:
- "Respond with ONLY valid JSON, no markdown fences"
- "Return ONLY valid JSON matching the exact schema"
- All responses parsed through markdown fence stripping + `json.loads()`
- Retry logic on JSON parse failure (tag: 1 retry, atomize: 1 retry, Parts Finder: 3 retries)

### 5.4 Rate Limiting & Cost Controls

| Setting | Value | File |
|---------|-------|------|
| Notion API rate limit | 3.0 req/s | `notion_notes/config.py:22` |
| Parts Finder AI retries | 3 | `parts-finder/src/parts_finder/config.py:34` |
| Tag retries | 1 | `notion_notes/commands/tag.py:33` |
| Atomize retries | 1 | `notion_notes/commands/atomize.py:33` |
| Link candidate limit | 20 | `notion_notes/commands/link.py:31` |
| Content truncation (tag) | 3000 chars | `notion_notes/commands/tag.py:263` |
| Content truncation (link) | 6000 chars | `notion_notes/commands/link.py:41` |
| Content truncation (atomize) | 8000 chars | `notion_notes/commands/atomize.py:34` |
| Cache TTL | 60 min | `parts-finder/src/parts_finder/config.py:30` |

### 5.5 Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Yes (for AI features) | — | Anthropic SDK authentication |
| `NOTION_TOKEN` | Yes (for Notion) | — | Notion API authentication |
| `CLAUDE_MODEL` | No | `claude-sonnet-4-6` | Override default model |
| `NOTION_RATE_LIMIT_RPS` | No | `3.0` | Notion API rate limiting |
| `NOTION_KB_DATABASE_ID` | No | `""` | Knowledge base database |
| `NOTION_NOTES_DATABASE_ID` | No | `""` | Notes database |

### 5.6 Dependencies

**Main project:**
```
anthropic          # Claude API SDK
```

**Parts Finder:**
```
anthropic>=0.40.0  # Claude API SDK
fastapi>=0.109.0   # HTTP framework
uvicorn>=0.27.0    # ASGI server
httpx>=0.27.0      # HTTP client
```

---

## 6. Optimization Opportunities (STAR Framework Assessment)

Based on the arXiv:2602.21814 finding that STAR reasoning frameworks outperform raw context injection for tasks requiring implicit constraint inference:

### 6.1 High-Impact Targets

**A. Parts Finder AI Fallback** — `fallback.py`
- **Current:** Raw context injection (vehicle info → instruction → schema)
- **STAR opportunity:** "Situation: You are looking up specs for a {year} {make} {model} with {engine_code} engine running on {fuel_type}. Task: Determine the correct maintenance specifications for {categories}. Action: Consider the engine displacement, fuel type requirements, and manufacturer specifications to select appropriate parts. Result: Return specifications in the following JSON format."
- **Why:** This task requires implicit constraint inference (0W-20 vs 5W-30 depends on engine type + climate + manufacturer spec — none of which are explicitly stated)
- **Impact:** Reducing hallucinated part numbers would directly improve user trust

**B. Notion Atomization** — `atomize.py`
- **Current:** "Analyze these class notes and identify distinct concepts"
- **STAR opportunity:** "Situation: You have a student's lecture notes on {title} covering {estimated_topic_count} potential concepts. Task: Split these notes into atomic, self-contained concept notes. Action: First identify concept boundaries by looking for: new definitions, new mechanisms, new claims, transitions between topics. Then for each concept, write a self-contained explanation. Result: Return {max_concepts} or fewer concept notes in JSON."
- **Why:** Concept boundary detection is the hardest part — students rarely write with clear boundaries
- **Impact:** Better atomization → better knowledge graph → better linking

**C. Notion Linking** — `link.py`
- **Current:** "Identify which candidates are genuinely related"
- **STAR opportunity:** "Situation: Note '{title}' covers {domain} concepts including {key_topics}. You have {n} candidate notes from related domains. Task: Identify genuine cross-domain connections that would help a student understand deeper relationships. Action: For each candidate, ask: (1) Do these notes share an underlying mechanism? (2) Does one explain why the other works? (3) Do they present contradictory evidence? Result: Return only connections where you can articulate a specific educational benefit."
- **Why:** The current prompt doesn't scaffold the reasoning process — it just says "find connections"
- **Impact:** Fewer false-positive connections, higher-quality knowledge graph

### 6.2 Medium-Impact Targets

**D. Temperature tuning** — Currently ALL prompts use SDK defaults
- Classification tasks (tagging) → lower temperature (0.0-0.3)
- Creative tasks (atomization) → moderate temperature (0.3-0.5)
- Lookup tasks (Parts Finder) → temperature 0.0 (deterministic)

**E. Few-shot examples** — No prompts include examples of correct/incorrect outputs
- Parts Finder: include 1-2 examples of correct vehicle spec lookups
- Tagger: include 1 example of correct domain+tag assignment
- Linker: include 1 example of genuine vs trivial connection

### 6.3 Architecture-Level Observations

**F. The CLAUDE.md behavioral contract IS a STAR-like framework** (approval request = Situation+Task+Action+Result), but this reasoning scaffold only governs the human-in-the-loop workflow. The API-level prompts don't benefit from it.

**G. The knowledge graph pipeline** (`/reduce` → `/reflect` → `/reweave` → `/verify`) is a multi-step reasoning chain that could be enhanced with STAR framing at each stage.

**H. No prompts use explicit confidence calibration.** Adding "If you're less than 80% confident in a specification, return null for that field" to the Parts Finder prompt could reduce hallucinations.

---

## Appendix: File Index

| File | Purpose | Section |
|------|---------|---------|
| `CLAUDE.md` | Behavioral contract | §1.3 |
| `.claude/settings.local.json` | Hooks, permissions, guardrails | §4.3 |
| `.claude/agents/dsp-reviewer.md` | DSP specialist prompt | §1.4 |
| `.claude/agents/master-reviewer.md` | Senior reviewer prompt | §1.4 |
| `.codex/skills/implementor-stage-gate/SKILL.md` | Codex stage gate | §1.5 |
| `notion_notes/claude_client.py` | Anthropic SDK wrapper | §2.2 |
| `notion_notes/config.py` | Notion Notes config | §5.1 |
| `notion_notes/commands/tag.py` | Tagging system + user prompt | §1.2B |
| `notion_notes/commands/link.py` | Linking system + user prompt | §1.2C |
| `notion_notes/commands/atomize.py` | Atomize system + user prompt | §1.2D |
| `notion_notes/commands/process.py` | Pipeline orchestrator | §4.2 |
| `parts-finder/src/parts_finder/config.py` | Parts Finder config | §5.1 |
| `parts-finder/src/parts_finder/api/fallback.py` | AI fallback + system prompt | §1.2A |
| `parts-finder/src/parts_finder/api/routes.py` | Request routing | §4.1 |
| `parts-finder/src/parts_finder/api/app.py` | App factory + AI init | §4.1 |
