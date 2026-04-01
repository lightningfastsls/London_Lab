# Aggregated Insights for /learn — Claude Code & Agent Optimization

**Purpose:** Feed these into arscontexta via `/learn` to create persistent, traversable knowledge about how to build better Claude Code workflows, manage context, and organize agent systems.

**How to use:** Copy each section (or individual claims) into `/learn "..."` commands. Each claim is designed to be atomic and self-contained. The source links let you trace back to the original conversation.

---

## Topic 1: Context Window Degradation

**Source:** Geoffrey Huntley (Sourcegraph) findings, community best practices

### Claims to /learn:

**Claim 1.1 — Context degradation threshold:**
Performance degradation in Claude begins around 40% of the 200k token context window (roughly 80k tokens). The degradation is gradual, not a cliff. Geoffrey Huntley at Sourcegraph found that context window quality degrades specifically around 147k-152k tokens, but meaningful quality loss starts much earlier. The commonly cited sweet spot is to stay within 60-80% of the window and reserve the last 20% for critical multi-file operations. System prompts consume a chunk of the advertised 200k limit, bringing usable space down to roughly 176k.

**Claim 1.2 — Degradation symptoms:**
When context fills up, symptoms include: shorter responses, hallucinated APIs, forgetting earlier instructions, and losing track of file structures. These are gradual — you won't notice a sudden cliff, but output quality erodes incrementally.

**Claim 1.3 — The "context as RAM" principle:**
Treat context like RAM, not like a hard drive. Keep it small and focused rather than accumulating everything. State that needs to persist across tasks belongs in files on disk (CLAUDE.md, specs, DECISIONS.md), not in the conversation context.

**Claim 1.4 — Fresh Context Pattern:**
Each task should get a fresh context window. Between tasks, state persists in external files (markdown specs, task lists, CLAUDE.md). The workflow: implement one module → commit → /clear → start fresh with the next module, referencing the spec file.

**Claim 1.5 — Subagents for context isolation:**
Each subagent operates in its own context window, so the main orchestrator stays clean. The orchestrator holds only the high-level plan while each implementation task gets fresh context automatically. Use the Task tool to delegate discrete units of work.

**Claim 1.6 — Auto-compact configuration:**
Claude Code has built-in auto-compaction. Configure the threshold using `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` environment variable (1-100%, default 95%). To disable: `claude config set -g autoCompactEnabled false`. The smarter approach is to compact strategically at logical breakpoints rather than letting it fire randomly. You can also give preservation instructions: `/compact preserve all file paths, architectural decisions, and error messages from this session`.

**Claim 1.7 — The /clear + /catchup pattern:**
Clear the context state, then run a custom `/catchup` command that makes Claude read all changed files in the git branch. This is a lightweight Fresh Context Pattern automatable with a custom slash command.

**Claim 1.8 — Slash commands for context management:**
Create a `.claude/commands/implement-module.md` slash command that encodes: read spec → implement as subagent task → commit → compact or clear. This way `/implement-module auth-system` handles context management automatically.

---

## Topic 2: Multi-Turn Conversation Degradation

**Source:** Laban et al., "LLMs Get Lost In Multi-Turn Conversation" (https://arxiv.org/html/2505.06120v1)
**Paper:** Laban et al., "LLMs Get Lost In Multi-Turn Conversation" — https://arxiv.org/html/2505.06120v1

### Claims to /learn:

**Claim 2.1 — Multi-turn performance drop:**
All 15 tested LLMs (GPT-4.1, Gemini 2.5 Pro, Claude 3.7 Sonnet, Deepseek-R1) show an average 39% performance drop when tasks are delivered across multiple turns ("sharded") versus a single fully-specified instruction. The degradation is mostly due to increased unreliability (112% increase) rather than loss of aptitude (only 16% drop). Models can still solve problems but do so inconsistently.

**Claim 2.2 — Even 2 turns trigger degradation:**
Even 2-turn conversations trigger the multi-turn degradation effect. Reasoning models (o3, R1) don't help — they actually make it worse by being more verbose.

**Claim 2.3 — Root causes of multi-turn degradation:**
LLMs make premature assumptions in early turns, attempt full solutions too early, produce overly verbose responses, and then get anchored to their own incorrect previous outputs. This is the "answer bloat" problem — long assistant responses from turn 3 become expensive input context for turns 4, 5, 6, compounding both cost and quality degradation.

**Claim 2.4 — The "Concat" strategy:**
Consolidating everything into one message and starting a fresh conversation preserves ~95% of single-turn performance. This validates the Fresh Context Pattern and plan-file-first approach.

**Claim 2.5 — When going off track, restart don't correct:**
When Claude Code goes off track, consolidate and restart rather than adding more correction turns. The paper shows that continuing to correct just makes degradation worse. Writing plan files that consolidate context upfront is essentially the Concat strategy.

**Claim 2.6 — Implications for conversational AI products:**
If building multi-turn chat features (like Cloudy Claude's NL assistant), implement a "snowball" or recap pattern — periodically consolidating the conversation state before sending it to the model. This recovers 15-20% of the loss from naive multi-turn.

---

## Topic 3: Behavioral Contracts (Tangi Vass Approach)

**Source:** Tangi Vass, "Turning AI Coding Agents into Senior Engineering Peers" (https://medium.com/@tangi.vass/turning-ai-coding-agents-into-senior-engineering-peers-c3d178621c9e)
**Article:** Tangi Vass, "Turning AI Coding Agents into Senior Engineering Peers" — https://medium.com/@tangi.vass/turning-ai-coding-agents-into-senior-engineering-peers-c3d178621c9e

### Claims to /learn:

**Claim 3.1 — Agents are mis-incentivized, not incapable:**
AI coding agents aren't broken — they're mis-incentivized. They're trained to appear helpful, agreeable, and competent, which leads to deception, cheerleading, scope creep, and random trial-and-error when they hit difficulty. The fix isn't better prompts — it's structural constraints that force the agent into better decision-making patterns.

**Claim 3.2 — State machine with forbidden transitions:**
Define explicit states (IDLE → ANALYSIS → APPROVAL_PENDING → EXECUTION → VALIDATION → DONE) with forbidden transitions. Critical forbidden transitions: ANALYSIS → EXECUTION (skipping approval), EXECUTION → DONE (skipping validation). These aren't warnings — they're structural impossibilities.

**Claim 3.3 — Approval Request format forces quality:**
Before any state-changing action, the agent must present: Intent, Scope, Commands, Consequences, Risks, Validation plan, and an explicit Ask. Agents resist stating incompetent plans because they're trained to appear competent. "I'll try random things until something works" is hard to write in an Approval Request. Surface the reasoning, and the reasoning improves.

**Claim 3.4 — Struggle Protocol over silent failure:**
When stuck, agents should surface blockers explicitly rather than spiraling through random approaches. Format: 🚨 BLOCKED — What I understand, What I tried, Where I'm stuck, Learning angle. Deception (claiming success) is the failure mode, not struggle.

**Claim 3.5 — DoR/DoD as thinking tools:**
Definition of Ready (gate: ANALYSIS → APPROVAL) and Definition of Done (gate: VALIDATION → DONE) are mental models built by the agent itself, not ceremony. DoR checks: Intent clear? Assumptions within budget? Scope bounded? DoD checks: Code complete? Tests pass? Docs updated? Validation executed?

**Claim 3.6 — Test integrity as non-negotiable:**
Never modify test expected values to make tests pass. This is a Tier 0 integrity rule that applies regardless of context. Fix the code or discuss expectations with the developer. This is especially critical for research projects where test corruption could compromise scientific integrity.

**Claim 3.7 — Tiered contracts by context:**
Different AI contexts need different contract sizes. Claude Code (expensive, capable): ~200 lines, full state machine + approval gates + struggle protocol. Claude.ai conversations: ~50 lines, core rules only. Cheap worker models: ~30 lines, single-task constraints + explicit validation. The full 866-line contract is too systemic to be understood by reading — it must be run.

**Claim 3.8 — Learning-first priority reframes everything:**
When the developer's highest priority is learning (over performance or token optimization), the behavioral contract becomes even more valuable. Approval requests expose the "why" behind decisions. Struggle protocols reveal what's actually hard. The agent's reasoning becomes visible, so the developer learns from how it approaches problems.

**Claim 3.9 — Cost gradient principle:**
Thought → Words → Specs → Code → Tests → Docs → Commits. Cheaper errors on the left, expensive errors on the right. Catch problems as early as possible in this gradient.

**Claim 3.10 — Anti-gaming rule:**
"Technically compliant" ≠ compliant if the user would object with full information. Loophole-finding is itself a violation.

---

## Topic 4: Arscontexta / Skill Graphs

**Source:** arscontexta plugin (https://github.com/agenticnotetaking/arscontexta)
**Plugin:** https://github.com/agenticnotetaking/arscontexta

### Claims to /learn:

**Claim 4.1 — Skill graphs > flat SKILL.md:**
A single SKILL.md or CLAUDE.md is a flat document. Knowledge isn't flat — it's a graph. When working on a project, knowledge about architecture is connected to knowledge about signal processing, which is connected to labeling methodology, which is connected to false positive patterns. A skill graph makes those connections traversable by the agent, pulling in relevant context based on what it's doing right now.

**Claim 4.2 — The context bottleneck shift:**
As AI agents become more capable, the bottleneck shifts from "can the agent do X" to "does the agent have the right context to do X well." Skill graphs address this by making accumulated domain knowledge persistent and selectively accessible.

**Claim 4.3 — Skill graph vs CLAUDE.md complementarity:**
The skill graph is the source of truth for domain knowledge. CLAUDE.md is the source of truth for behavioral contracts and agent coordination. These are complementary, not redundant. Don't let skill graphs replace formal docs (ADRs, DECISIONS.md, module docs) — let them supplement with operational knowledge too granular for formal docs.

**Claim 4.4 — The 6R processing pipeline:**
Record → Reduce → Reflect → Reweave → Verify → Rethink. Each step transforms raw notes into connected, actionable knowledge. /reduce distills session learnings, /reflect finds connections, /reweave updates old notes with new context, /verify quality-checks, /rethink challenges assumptions.

**Claim 4.5 — Each project gets its own graph:**
Different projects need different presets, vocabularies, and traversal patterns. Don't merge project graphs. Use "Experimental" preset for fast-moving development projects, "Research" preset for academic/research work.

**Claim 4.6 — Maintenance overhead must be sustainable:**
Weekly maintenance should take no more than 15 min per project. If you're spending more, scale back. Check: /arscontexta:health, /reflect, /reweave, /stats. After major milestones: /rethink to challenge assumptions.

**Claim 4.7 — The collector's fallacy warning:**
Automated extraction can make you feel knowledgeable because you have a comprehensive graph, when really you've just stored things. The artifact of organization substitutes for actual understanding. With manual Zettelkasten, effort forces selectivity. Automation removes that filter.

**Claim 4.8 — Generation effect tradeoff:**
The generation effect (actively synthesizing information produces stronger memory traces) means automated note-taking may reduce personal learning. Use automation for agent context (where the goal is the agent having context, not you learning). Keep more manual approaches for coursework where the goal is internalizing concepts.

**Claim 4.9 — Seeding knowledge after merging PRs:**
After merging a PR from autonomous work, seed key decisions/findings into arscontexta via /seed, then run /pipeline to process into the vault. This bridges the gap between code changes and knowledge capture.

---

## Topic 5: Memory MCP Evaluation

**Source:** Claude Code plugin ecosystem evaluation

### Claims to /learn:

**Claim 5.1 — Memory MCP solves operational knowledge loss:**
Cross-session memory currently lives in static files (DECISIONS.md, ROADMAP.md, module docs). These capture planned architecture and formal decisions but not informal knowledge accumulated during implementation — things like "the FileConnector chokes on Madaf exports with mixed encoding" or "Prophet's default changepoint_prior_scale was too sensitive for sparse weekly demand data." That operational knowledge dies with the session.

**Claim 5.2 — Memory MCP best use cases:**
ML debugging history (hyperparameter outcomes, degenerate cluster findings), ERP connector edge cases (encoding issues, date formats), and cross-project knowledge transfer (signal processing insights from USV research applicable to time-series forecasting).

**Claim 5.3 — Memory MCP downsides:**
Token overhead from added tools in every session. Unstructured knowledge graph doesn't go through review pipelines. Accumulates junk without curation — need explicit instructions in CLAUDE.md about what to memorize. Recommendation: scope it to storing ML experiment outcomes and data integration edge cases, with explicit instructions like "store data quirks, model training insights, non-obvious connector behavior — do NOT store test results, routine code changes, or information already in DECISIONS.md."

**Claim 5.4 — Memory MCP vs arscontexta:**
Memory MCP is simpler (key-value knowledge graph, auto-recall) but less structured. Arscontexta is more powerful (traversable graph with processing pipeline) but higher overhead. For projects that already use arscontexta, Memory MCP is redundant and may conflict. Choose one per project.

---

## Topic 6: Reviewer Optimization

**Source:** Internal workflow optimization

### Claims to /learn:

**Claim 6.1 — Don't review what you just built in the same session:**
The main Claude Code session already has full context from building the module. Spawning a subagent to review means re-reading everything from scratch (106 tool calls, 150K tokens, 26 minutes for a simple cleanup). Instead: have the main session do the review while context is fresh.

**Claim 6.2 — Tiered review depth:**
Scale review depth to module complexity. Tier 1 (Light — cleanup, config, docs): main session self-review, checklist only, Sonnet. Tier 2 (Standard — new module, routes, services): subagent with focused scope, git diff pre-bundled, Sonnet. Tier 3 (Deep — ML pipeline, security, cross-cutting): dedicated Opus subagent, full architecture review.

**Claim 6.3 — Pre-bundle context for reviewers:**
Instead of the reviewer making 100+ individual file reads, pre-bundle the relevant context: run `git diff`, concatenate changed files, pass the bundle to the reviewer. This cuts tool calls dramatically.

**Claim 6.4 — Batch reviews over logical groups:**
Instead of reviewing after every single module, review after a logical group (all of Phase 0, or all models in Phase 1). This amortizes context-loading cost and lets the reviewer catch cross-module issues that per-module reviews miss.

**Claim 6.5 — Main session writes the output file:**
Have the main session write the review file instead of the subagent. Subagents often fail on file writing due to shell escaping issues with markdown backticks, wasting turns.

---

## Topic 7: Model Selection & Token Economics

**Sources:** Tangi Vass article, WizCloud API evaluation

### Claims to /learn:

**Claim 7.1 — Model selection by task type:**
Planning & Architecture → Opus (complex reasoning, design decisions). Algorithm Implementation → Sonnet (good balance of capability and speed). Code Reviews → Sonnet (thorough but doesn't need Opus-level reasoning for most reviews). Documentation Writing → Haiku (fast, straightforward). Simple Edits/Fixes → Haiku. Codebase Exploration → Haiku.

**Claim 7.2 — Hebrew tokenization 4x cost:**
Hebrew text costs approximately 4x more tokens than equivalent English text. When building Hebrew-language features (like Cloudy Claude's NL assistant), budget accordingly. For 3 managers asking 10-20 questions/day, it's still single-digit dollars at Haiku pricing even with the overhead.

**Claim 7.3 — Output tokens cost more than input:**
In the API, output tokens are roughly 3-5x more expensive than input tokens. On claude.ai, usage limits are primarily driven by output tokens. But in multi-turn conversations, all previous messages (yours and Claude's) get re-sent as input every turn, so verbose Claude responses compound in cost.

---

## Topic 8: Claude Code Plugins & Tools Ecosystem

**Source:** Claude Code plugin ecosystem evaluation

### Claims to /learn:

**Claim 8.1 — Most impactful plugin categories:**
MCP servers are the most impactful extensions because they give Claude Code access to external tools. Key ones: GitHub MCP (repo/PR/issue integration), Playwright MCP (browser automation/testing), Context7 MCP (version-specific API docs replacing stale training data), Memory MCP (persistent cross-session knowledge).

**Claim 8.2 — Orchestration tools landscape:**
Claude Squad — multiple Claude Code agents in separate terminal workspaces. Claude Flow — autonomous orchestration for writing, editing, testing. Feature Dev (official Anthropic) — structured workflow with specialized agents for codebase exploration, architecture, and review.

**Claim 8.3 — Code review plugin complements custom reviews:**
The official Code Review plugin does multi-agent PR review with confidence-based scoring. It's good for lightweight gate on every PR/commit. Your custom tiered review protocol is better for formal phase-boundary reviews. They're complementary — run /code-review for quick sanity checks on every push, keep handoff-based Tier 2/3 for formal reviews.

**Claim 8.4 — Context7 MCP for API accuracy:**
Context7 pulls version-specific, up-to-date API docs directly into the context window, replacing stale training data. Useful when working with rapidly-evolving libraries or APIs where Claude's training data may be outdated.

---

## Topic 9: Autonomous Workflow Patterns

**Source:** Internal workflow design

### Claims to /learn:

**Claim 9.1 — Autonomous task suitability tiers:**
Not all tasks are suitable for autonomous execution. Tier 1 (ideal) — pure math/synthetic data, verifiable with tests. Tier 2 (good with careful scoping) — clear architecture, testable on dummy data. Tier 3 (never automate) — needs human judgment, design decisions, labeling, data interpretation.

**Claim 9.2 — GitHub Actions as autonomous workflow:**
GitHub Actions workflow for autonomous Claude Code work: create a branch, create an issue with a detailed spec, Claude Code picks it up via GitHub Actions, submits a PR, you review from your phone. Advantages: arscontexta completely isolated (can't touch vault), everything is a reviewable PR, you trigger each task explicitly, works from phone, fresh context per issue.

**Claim 9.3 — Vacation workflow structure:**
Before vacation: fix tool compatibility issues (30 min). During: create issues with detailed specs, review PRs from phone when convenient. After: merge PRs, seed learnings into arscontexta, run /reflect. Skip days have zero cost — PRs just wait.

**Claim 9.4 — Things to never run autonomously:**
Never automate: labeling (needs human eyes), architecture decisions, knowledge system maintenance/evolution, anything involving real data interpretation, modifications to stable tested code, modifications to knowledge vault files.

---

## Topic 10: Project Organization Principles

**Sources:** Multiple conversations aggregated

### Claims to /learn:

**Claim 10.1 — Documentation hierarchy:**
CLAUDE.md = behavioral contracts + agent coordination (source of truth for how agents behave). DECISIONS.md / ADRs = architecture decisions (source of truth for why things were built this way). Module docs = implementation details (source of truth for what code does). Skill graph = domain knowledge + operational learnings (source of truth for accumulated insights). Handoffs = session state transfer (ephemeral, per-review).

**Claim 10.2 — Plan files over verbal instructions:**
When planning a new skill or agent, write a plan file (markdown spec) rather than giving manual steps in conversation. This serves as both the instruction and the persistent record. It also enables the Concat strategy from the multi-turn degradation research.

**Claim 10.3 — The dual-AI workflow:**
Claude Code handles reasoning tasks (architecture, complex algorithms, debugging, refactoring, teaching). Codex/mechanical agents handle repetitive tasks (writing unit tests for existing functions, adding docstrings, boilerplate, simple utilities). When a task is clearly mechanical, suggest deferring to Codex to save tokens.

**Claim 10.4 — When to update CLAUDE.md:**
After discovering a new behavioral pattern that should persist across sessions. After a workflow change that affects all future sessions. After identifying a new domain-specific constraint or convention. NOT for individual task details or temporary state.

---

## Quick Reference: Key Thresholds & Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Context degradation onset | ~40% of window (80k tokens) | Context degradation research |
| Hard degradation zone | ~147k-152k tokens | Geoffrey Huntley / Sourcegraph |
| Usable context (after system prompt) | ~176k of 200k | Context degradation research |
| Multi-turn performance drop | 39% average | Laban et al. paper |
| Multi-turn unreliability increase | 112% | Laban et al. paper |
| Multi-turn aptitude drop | Only 16% | Laban et al. paper |
| Concat strategy performance preservation | ~95% of single-turn | Laban et al. paper |
| Hebrew tokenization overhead | ~4x vs English | WizCloud API evaluation |
| Output vs input token cost | ~3-5x more expensive | Laban et al. paper |
| Auto-compact default threshold | 95% | Context degradation research |
| arscontexta weekly maintenance budget | ≤15 min/project | arscontexta docs |
| Tangi Vass full contract size | 866 lines | Tangi Vass article |

---

## External References

### Papers
| Paper | URL |
|-------|-----|
| Laban et al., "LLMs Get Lost In Multi-Turn Conversation" | https://arxiv.org/html/2505.06120v1 |

### Articles & Blog Posts
| Article | URL |
|---------|-----|
| Tangi Vass, "Turning AI Coding Agents into Senior Engineering Peers" | https://medium.com/@tangi.vass/turning-ai-coding-agents-into-senior-engineering-peers-c3d178621c9e |
| Tangi Vass, "I Tried to Kill Vibe Coding" (follow-up) | https://medium.com/@tangi.vass/i-tried-to-kill-vibe-coding-i-built-adversarial-vibe-coding-without-the-vibes-bc4a63872440 |
| Heinrich (@arscontexta), "Skill Graphs > SKILL.md" (X thread) | https://x.com/arscontexta/status/2023957499183829467 |

### GitHub Repositories & Tools
| Repo | URL |
|------|-----|
| arscontexta plugin | https://github.com/agenticnotetaking/arscontexta |
| Memory MCP Server | https://github.com/modelcontextprotocol/servers |
| WizCloud (Hashavshevet) API docs | https://docs.wizcloud.co.il/ |

### Key Findings (no direct link found)
| Finding | Attribution |
|---------|------------|
| Context quality degrades at ~147k-152k tokens | Geoffrey Huntley, Sourcegraph |
