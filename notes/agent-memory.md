---
description: Cross-session memory architecture, forgetting strategies, scoping, evaluation, MCP ecosystem, and multi-agent orchestration infrastructure
type: moc
---

# agent-memory

How agents persist knowledge across sessions, manage memory lifecycle, and coordinate through shared infrastructure. Split from [[context-management]] to separate cross-session persistence and infrastructure concerns from within-session context utilization. The two domains share a key bridge: since [[progressive disclosure in memory retrieval saves 10-20x tokens by loading compact indices before full content]], memory design directly affects within-session context budget.

## Synthesis

Agent memory has emerged as a distinct architectural discipline. Since [[agent memory is a distinct discipline from LLM memory RAG and context engineering with formation evolution and retrieval as core dynamics]], it requires its own design patterns beyond RAG or context engineering. The field is converging on [[hybrid memory architectures combining keyword vector and graph search are converging as the dominant paradigm for agent memory]], with graph-based approaches providing verifiable grounding since [[graph-based memory reduces hallucination by grounding agent outputs in structured verifiable content]]. Active forgetting is not optional — since [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]], the question is which strategy to use, not whether to forget. The MCP ecosystem provides the infrastructure layer, with [[8600 MCP servers and 6500 Claude Code plugin repositories exist as of March 2026 reflecting rapid open ecosystem growth]] and orchestration patterns ranging from subagents to agent teams to community tools.

## Memory Architecture & Paradigms
- [[agent memory is a distinct discipline from LLM memory RAG and context engineering with formation evolution and retrieval as core dynamics]] -- Dec 2025 survey formalizing the field with three-dimensional taxonomy
- [[96 MCP memory servers exist as of March 2026 fragmented across unstructured markdown knowledge graph and hybrid vector-graph paradigms]] -- landscape baseline for the three architectural camps
- [[knowledge graph memory outperforms flat storage for multi-hop reasoning temporal coherence and hallucination reduction but scales poorly with large memories]] -- the key paradigm comparison
- [[hybrid memory architectures combining keyword vector and graph search are converging as the dominant paradigm for agent memory]] -- the convergence trend from Feb 2026 taxonomy survey
- [[graph-based memory reduces hallucination by grounding agent outputs in structured verifiable content]] -- mechanism linking structure to reliability
- [[graph-based memory taxonomy classifies agent memory across temporal scope functional role structure and cognitive type]] -- four-dimensional classification of the design space
- [[Anthropic reference MCP memory server uses entity-relation-observation knowledge graph as JSONL with no built-in decay or scoping]] -- the canonical baseline implementation
- [[complementary memory architecture uses CLAUDE.md for stable rules auto-memory for learned patterns and MCP for structured cross-session knowledge]] -- the three-tier pattern emerging in practice
- [[Claude Code auto-memory captures configuration not learning because it preserves workspace patterns but loses diagnostic reasoning paths]] -- the qualitative limitation of auto-memory
- [[auto-memory 200-line hard-coded limit and lack of automatic consolidation creates growing redundancy without manual intervention]] -- the quantitative constraint

## Forgetting & Lifecycle
- [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]] -- the anti-pattern motivating managed forgetting
- [[three forgetting strategies dominate agent memory temporal decay importance-based pruning and contradiction-based shadowing]] -- the design space for forgetting
- [[HESTIA scoring uses shadow-decay with quadratic penalty for contradicted memories enabling graceful knowledge evolution without deletion]] -- Ember MCP's scoring formula combining all three strategies
- [[Voronoi-based drift detection identifies when memory topic clusters have shifted signaling reorganization needs]] -- structural change detection complementing individual memory scoring
- [[dream-inspired consolidation cycles compress old memories on daily weekly monthly schedules to manage long-term growth]] -- biologically-inspired scheduled maintenance
- [[semantic compression pipeline achieves 30x token reduction through structured compression online synthesis and intent-aware retrieval]] -- SimpleMem's three-stage pipeline for token efficiency

## Scoping & Isolation
- [[memory scoping by project agent and task prevents cross-project contamination in multi-context agent systems]] -- the three-dimensional isolation pattern
- [[single-scope MCP memory causes cross-project contamination when agent contexts are not separated]] -- AIMultiple benchmark failure finding
- [[cross-agent memory bridges enable tool-agnostic knowledge persistence across multiple IDE platforms through shared storage]] -- Memorix 8-IDE pattern

## Token Efficiency
- [[progressive disclosure in memory retrieval saves 10-20x tokens by loading compact indices before full content]] -- memory-specific application of progressive disclosure (bridge to [[context-management]])

## Evaluation & Benchmarks
- [[eight practical criteria for evaluating cross-session memory tools span retrieval precision token efficiency scope isolation forgetting quality setup friction portability auditability and consolidation]] -- synthesized evaluation framework
- [[Graphiti temporal knowledge graph achieved 94.8 percent accuracy on DMR with entropy-gated fuzzy matching for entity deduplication]] -- leading benchmark baseline
- [[MemOS treats memory as managed system resource with three-layer architecture achieving 159 percent improvement in temporal reasoning]] -- memory-as-OS paradigm shift
- [[AgeMem unified memory with tool-based operations outperforms separate LTM and STM components with heuristic controllers]] -- RL-trained memory operations as alternative to rule-based forgetting

## MCP Ecosystem & Orchestration
- [[8600 MCP servers and 6500 Claude Code plugin repositories exist as of March 2026 reflecting rapid open ecosystem growth]] -- landscape baseline: MCP as industry standard
- [[Claude Code plugin architecture grew from zero to 9000 plugins in under a year through directory-based design with no build step or registry approval]] -- minimal friction design driving rapid ecosystem growth
- [[MCP Tool Search reduces context pollution by 85-95 percent through lazy loading of tool descriptions via lightweight search index]] -- the critical enabler for scaling MCP servers (67K to 8.7K tokens, bridge to [[context-management]])
- [[Claude Code as MCP server enables agent-to-agent orchestration where other tools invoke Claude's file editing and command execution remotely]] -- agent-to-agent via MCP

## Multi-Agent Orchestration
- [[Claude Code agent teams enable inter-agent messaging and shared task lists unlike subagents which only report to parent]] -- experimental first-party multi-agent feature
- [[three parallelization strategies in Claude Code serve different isolation needs subagents for focused results teams for collaborative discussion worktrees for filesystem safety]] -- the three-layer parallelization model
- [[agent team token costs scale linearly with teammates making 3-5 the recommended size before coordination overhead dominates]] -- practical scaling constraint
- [[git worktrees have become the standard filesystem isolation primitive for multi-agent coding work]] -- convergent filesystem isolation pattern across five implementations
- [[Claude Squad and Claude Flow provide community multi-agent orchestration with broader tool support than official agent teams]] -- community alternatives with cross-tool support

## Knowledge Activation (RAG Patterns)
- [[activation timing matters as much as retrieval quality in agent knowledge systems]] -- core thesis: four RAG papers converge on the retrieve-or-not decision as the primary bottleneck
- [[Self-RAG reflection tokens translate to procedural gates when model fine-tuning is unavailable]] -- learned retrieve/no-retrieve tokens → explicit skill steps
- [[FLARE uses the agent's intended action as a retrieval query enabling pre-modification knowledge checks]] -- intent-based retrieval from draft output
- [[CRAG's retrieval evaluator prevents noise-induced gate fatigue through relevance thresholds]] -- filter weak results before surfacing to preserve gate credibility
- [[Adaptive RAG routes retrieval depth by query complexity which maps to file modification risk in coding agents]] -- risk-based routing: HIGH/MEDIUM/LOW modification risk → retrieval depth
- [[static inline references and intent-based search activate knowledge at different points in the modification lifecycle]] -- canary comments (reactive, file-level) vs /kcheck (proactive, planning-level)
- [[fewer well-placed activation triggers outperform many ignored ones because noise teaches agents to skip gates]] -- alarm fatigue principle: precision over coverage
- [[cross-agent knowledge transfer requires flattening graph-traversable constraints into self-contained plain text]] -- multi-agent handoff pattern when receiver lacks vault access
- [[whether tracking which surfaced notes agents actually load can identify poorly-described vault entries]] -- extends CRAG evaluator: load-rate as continuous quality signal for descriptions (also in Open Questions)

## Between-Session Processing & Consolidation
- [[Letta sleep-time compute pairs a primary agent with a sleep-time agent that processes memory during idle periods]] -- paired-agent architecture requiring discrete idle gaps for memory consolidation
- [[between-session observation accumulation is directed dreaming that produces patterns no individual session contained]] -- the vault's implementation: observation accumulation + threshold-triggered rethink as directed dreaming
- [[each between-session processing cycle is a training step that does not touch the weights]] -- reframes between-session processing as structural adaptation equivalent to weight updates
- [[session boundaries simultaneously limit agents and enable between-session processing making the limitation the precondition]] -- the paradox: the gap that destroys continuity is the gap that enables consolidation

## Open Questions
- [[whether MCP memory servers would improve this vault's session continuity beyond the current ops handoff approach]] -- practical applicability of MCP memory to this vault
- [[how memory scoping interacts with behavioral contracts when agents share cross-project knowledge]] -- governance/infrastructure boundary question (also in [[agent-governance]])
- [[whether specialization across multiple AI tools via MCP orchestration outperforms monolithic agent approaches for complex coding tasks]] -- tool specialization vs monolithic (also in [[agent-governance]])
- [[whether tracking which surfaced notes agents actually load can identify poorly-described vault entries]] -- CRAG-inspired diagnostic: load-rate as description quality signal

## Agent Notes
- This vault is itself a hybrid memory system: wiki links as typed edges, atomic notes as entities, YAML frontmatter as metadata, qmd for semantic search. The cross-session memory research validates this architecture while revealing gaps (no automatic forgetting, no cross-tool portability).
- The vault's processing pipeline (/reduce -> /reflect -> /reweave -> /verify) serves as a manual forgetting mechanism — it prevents raw accumulation by forcing curation, analogous to the three forgetting strategies.
- The orient->work->persist session rhythm implements temporal scoping: each session starts fresh with just ops/goals.md and MEMORY.md, avoiding the unbounded accumulation that degrades performance.
- Bridge notes to [[context-management]]: MCP Tool Search (context pollution = within-session concern), progressive disclosure (memory retrieval affects context budget), subagent architecture (orchestration pattern in both).
