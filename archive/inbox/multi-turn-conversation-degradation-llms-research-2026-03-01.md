---
description: "Deep survey of multi-turn LLM degradation: 39% avg performance drop, root causes, mitigations, and the broader evaluation landscape"
source_type: article
url: "multiple -- see source log"
author: "multiple sources (primary: Laban et al. 2025)"
date_accessed: "2026-03-01"
status: processed
processed_date: "2026-03-01"
research_tool: "web-search"
research_query: "multi-turn conversation degradation LLMs Laban et al"
research_depth: "deep"
---

# Multi-Turn Conversation Degradation in LLMs

LLMs suffer a systematic, architecture-independent 39% average performance drop when tasks are distributed across conversational turns rather than specified in a single prompt. This degradation is primarily driven by a 112% increase in unreliability rather than capability loss, and persists even in frontier models tested through early 2026. The phenomenon has been accepted as an ICLR 2026 oral presentation, spawned follow-up research reframing the root cause as intent mismatch rather than capability deficit, and motivated a new generation of multi-turn evaluation benchmarks.

---

## The Primary Study: Laban et al. (2025)

### Paper Details

"LLMs Get Lost In Multi-Turn Conversation" by Philippe Laban and Jennifer Neville (Microsoft Research) and Hiroaki Hayashi and Yingbo Zhou (Salesforce Research). Published May 2025 on arXiv (2505.06120), accepted as an ICLR 2026 oral presentation. Code and data released at github.com/microsoft/lost_in_conversation (217 stars, 17 forks as of March 2026) and on HuggingFace (datasets/Microsoft/lost_in_conversation).

### Experimental Methodology: Instruction Sharding

The core innovation is "instruction sharding" -- transforming fully-specified single-turn instructions into sets of smaller informational units that collectively preserve the original meaning. This allows direct comparison between single-turn and multi-turn performance on identical tasks.

The sharding process enforces five properties: (P1) information preservation across shards, (P2) clear initial intent in the first shard, (P3) order-insensitive information revelation, (P4) maximal sharding granularity, and (P5) minimal transformation from the original. The process is semi-automatic: an LLM identifies logical units, rephrases them conversationally, verifies completeness, then authors manually inspect and edit. Approximately 3 hours of manual work per 100 sharded instructions.

A GPT-4o-mini user simulator manages the conversation: it holds the full sharded instruction set, selects which shard to reveal next, rephrases it naturally, and maintains conversational coherence. The assistant's response is classified into 7 categories (clarification, refusal, hedging, interrogation, discussion, missing, or answer attempt). If an answer attempt, it is extracted and scored via task-specific evaluators. The conversation ends when the answer is marked correct or all shards are exhausted.

### Six Benchmark Tasks

| Task | Source Dataset | Description |
|------|---------------|-------------|
| Code | HumanEval, LiveCodeBench | Python function generation |
| Database | Spider | Text-to-SQL queries |
| Actions | BFCL | API function calling |
| Math | GSM8K | Elementary arithmetic word problems |
| Data-to-Text | ToTTo | Tabular data captioning |
| Summary | Summary of a Haystack | Multi-document summarization with citations |

90-120 sharded instructions prepared per task (600 total). N=10 simulations per (model, instruction, conversation type) combination at temperature T=1.0, totaling 200,000+ simulated conversations at an estimated cost of approximately $5,000.

### Five Conversation Regimes Tested

- **Full**: Single-turn, fully-specified baseline (the gold standard)
- **Concat**: All shards concatenated as bullet points in a single turn (controls for rephrasing effects)
- **Sharded**: Multi-turn with one shard per turn (the primary experimental condition)
- **Recap**: Sharded + a final turn recapitulating all information
- **Snowball**: Turn-by-turn accumulation of all previous shards before each new one

### The 15 Models Tested

OpenAI: GPT-4o-mini, GPT-4o, o3, GPT-4.1. Anthropic: Claude 3 Haiku, Claude 3.7 Sonnet. Google: Gemini 2.5 Flash, Gemini 2.5 Pro. Meta: Llama 3.1-8B-Instruct, Llama 3.3-70B-Instruct, Llama 4 Scout. AI2: OLMo-2-13B. Microsoft: Phi-4. Deepseek: R1. Cohere: Command-A.

### Performance Results by Model

Top-tier full (single-turn) performance: GPT-4.1 (92.4%), GPT-4o (92.6%), Gemini 2.5 Pro (92.1%), o3 (92.0%), Claude 3.7 Sonnet (91.8%). Lower-tier: OLMo-2 (52.0%), Llama 3.1-8B (54.8%).

Multi-turn (sharded) performance: GPT-4.1 (55.2%), Gemini 2.5 Pro (53.4%), Claude 3.7 Sonnet (52.8%), GPT-4o (51.3%). Lower-tier: OLMo-2 (23.2%), Llama 3.1-8B (26.1%).

Degradation percentages (Full to Sharded): Llama 3.1-8B (46.0%), OLMo-2 (45.6%), GPT-4o (44.4%), Claude 3 Haiku (40.2%), Gemini 2.5 Pro (39.9%), Claude 3.7 Sonnet (39.9%), Deepseek R1 (39.2%), Command-A (38.9%), GPT-4.1 (38.7%), o3 (38.4%), GPT-4o-mini (37.5%), Phi-4 (36.1%), Llama 4 Scout (34.9%), Llama 3.3-70B (32.4%).

The critical finding: all models degrade by 30-46%, with no systematic advantage for larger, more capable, or reasoning-enhanced models.

### Aptitude vs. Unreliability Decomposition

Three metrics per instruction from N=10 simulations: P-bar (average performance, the mean score), A90 (aptitude, the 90th percentile score representing best-case capability), and U10-90 (unreliability, the gap between 90th and 10th percentiles).

In single-turn settings, higher aptitude correlates with lower unreliability. Top models show the lowest variability.

In multi-turn settings: aptitude drops modestly (16% average decline), but unreliability more than doubles (112% increase). All models converge to similarly high unreliability (approximately 50 percentile-point spread) regardless of their base capability. This is the paper's most striking finding: models do not lose their absolute ability in multi-turn settings -- they become radically unpredictable.

### Gradual Sharding Experiment

31 instructions expanded into 7 variants each, varying shard count from 2 to 8 while fixing task complexity. Tested on GPT-4o and GPT-4o-mini. Finding: both models degrade starting with just two shards. The degradation is not about information volume but about temporal distribution itself.

### Task Susceptibility

Tasks prone to degradation share three properties: (1) generative -- requiring editing/refining content, not just extracting answers, (2) multi-faceted -- multiple explicit specifications creating numerous shards, and (3) non-decomposable -- revealing one shard fundamentally alters the entire solution.

Most vulnerable: Data-to-Text showed consistent significant drops across all models. Summary also showed large degradations.

Most resilient: Actions tasks showed minimal degradation for some models (Command-A, Llama 4 Scout). Code tasks showed better preservation for Claude 3.7 Sonnet and GPT-4.1.

Translation (German-to-English) showed NO degradation at all (GPT-4o-mini: 39% Full vs 42% Sharded; GPT-4o: 36% vs 41%). The explanation: translation is episodic -- each turn translates incremental sentences independently, and solutions are decomposable.

---

## Four Root Causes Identified

### 1. Premature Answer Attempts

LLMs generate full solution proposals early in the conversation, making assumptions about details not yet specified. When contradicted by subsequent user turns, models struggle to revise their initial commitments. This behavior is rational given RLHF training objectives that reward helpfulness and penalize evasive responses -- the training signal encourages confident early answers.

### 2. Answer Bloat

Responses grow increasingly verbose across turns. Models rely on prior (often incorrect) answer attempts, adding assumptions rather than replacing incorrect content. New shards rarely invalidate prior guesses; instead, each response layers on more content without pruning errors.

### 3. Loss of Middle Turns

LLMs disproportionately attend to the first and last turns of a conversation, neglecting information revealed in middle turns. This is a recency/primacy bias operating at the conversation level, analogous to the well-known "lost in the middle" phenomenon for long-context retrieval.

### 4. Excessive Verbosity

Reasoning models generate 33% longer responses on average, introducing more assumptions that confuse context. Additional test-time compute (o3, Deepseek-R1) does not solve multi-turn unreliability despite producing longer reasoning chains. More thinking does not compensate for the structural problem of premature commitment.

---

## Mitigation Strategies Tested by Laban et al.

### Concat (Information Consolidation)

Concatenating all shards as bullet points in a single turn preserves approximately 95.1% of full single-turn performance on average. This confirms the degradation is not caused by information loss from rephrasing but by the multi-turn temporal distribution itself. Smaller models (under 13B) show more pronounced Concat drops (86-92%), indicating greater sensitivity to paraphrasing.

### Recap (End-of-Conversation Recapitulation)

A final turn recapitulates all previously revealed information. Results: GPT-4o-mini improved from 50.4% (Sharded) to 66.5% (Recap), GPT-4o from 59.1% to 76.6%. This is a 16-17 percentage point improvement. Limitation: unrealistic in practice because it requires knowing the conversation will end.

### Snowball (Turn-by-Turn Recapitulation)

Each turn includes all previously revealed shards before the new one. Results: GPT-4o-mini improved from 50.4% to 61.8%, GPT-4o from 59.1% to 65.3%. Improvement of 6-7 percentage points. Advantage: realistic and does not require knowing when the conversation ends. Limitation: only partially mitigates degradation.

### Temperature Reduction

At T=0.0, single-turn unreliability decreases by 50-80%. But multi-turn unreliability remains approximately 30 even at T=0.0 (vs approximately 40 at T=1.0). A 15-20% improvement at best. The explanation: one early token difference compounds across turns, and even deterministic decoding cannot prevent the cascade because the conversation structure itself introduces variation.

---

## 2026 Re-Run With Newer Models

Philippe Laban reported (February 2026) re-running experiments with newer frontier models including GPT-5.2 and Claude 4.6. Key findings: performance degradation shrank from 39% to 33% on average. The biggest gains appeared in Python coding tasks, where some models lost only 10-20%. However, as Laban noted, "the issue is far from solved" and real-world performance may be worse than test results, particularly when users change their minds mid-conversation.

---

## Follow-Up Research: Intent Mismatch (Liu et al. 2026)

"Intent Mismatch Causes LLMs to Get Lost in Multi-Turn Conversation" by Geng Liu, Fei Zhu, Rong Feng, Changyi Ma, Shiqi Wang, and Gaofeng Meng (City University of Hong Kong, HKISI CAS, Jilin University). arXiv 2602.07338, February 2026.

This paper reframes the "lost in conversation" phenomenon as pragmatic misalignment rather than capability deficit. The core argument: premature assumptions are rational behavior given training objectives, and the real bottleneck is "a pragmatic mismatch between user expression and model interpretation." Users exhibit systematic individual variation where the same utterance may map to disparate underlying intentions, and general models default to population-level priors rather than individual pragmatics.

### The Mediator-Assistant Framework

A two-stage pipeline: (1) a Mediator analyzes accumulated context and distilled user-specific guidelines to reconstruct ambiguous inputs into explicit, fully-specified instructions, (2) an Assistant executes tasks based on clarified instructions. The Mediator approximates P(It|Ct, Eu) where Eu represents distilled pragmatic experiences.

### Experience Refiner

An LLM-based module that identifies failed multi-turn trajectories paired with successful single-turn inputs from historical logs, analyzes the pairs to extract structured textual guidelines capturing pragmatic patterns (e.g., "If the user hasn't explicitly approved a solution, they remain unsatisfied"), and provides these as system instructions for the Mediator.

### Results

Average recovery of approximately 20 percentage points in accuracy. For GPT-4o-mini: Full 86.9%, Sharded 53.6%, With Mediator 73.9% (20.3pp recovery). Even reasoning-enhanced DeepSeek-V3.2-Thinking gained 21.1%, suggesting intent clarity complements reasoning depth. Tested on Code, Database, Actions, and Math tasks.

Notably, the paper found that RAG-based memory (Mem0) provided only 3% improvement (53.6% to 56.5%), demonstrating that "retrieving context is not equivalent to resolving intent." Simple summarization baselines also showed marginal gains.

An important finding: approximately 60% of relative performance degradation is constant across model sizes, contradicting the hypothesis that scaling solves the problem.

---

## Critical Response: Arani (2025)

Reza Arani published a critical analysis arguing the Microsoft-Salesforce study does not prove LLMs are intrinsically incapable of robust multi-turn reasoning. His main points:

1. A significant portion of the performance drop vanishes when conversation history is managed more intelligently
2. The study's prompts contain design flaws including conflicting instructions, unrealistic segment limits, ambiguous labels, and inconsistent JSON formatting
3. These prompt-level issues systematically bias model outputs, making it essential to distinguish between genuine model limitations and artifacts of the experimental setup
4. The fully automated simulation framework is simplistic and idealized -- conversations are guaranteed to end with sufficient information, and the simulator limits unexpected behavior

This critique suggests the 39% figure may overstate the problem for well-designed production systems while possibly understating it for messy real-world interactions.

---

## Related Multi-Turn Benchmarks

### MultiChallenge (Sirdeshmukh et al. 2025)

Published at ACL 2025 Findings. Identifies four categories of multi-turn challenges: (1) instruction retention -- following first-turn instructions throughout the conversation, (2) inference memory -- recalling and connecting scattered details from previous turns, (3) reliable versioned editing -- properly revising materials through back-and-forth iterations, (4) self-coherence -- maintaining consistency with prior responses and avoiding sycophancy.

Despite near-perfect scores on existing multi-turn benchmarks, all frontier models scored below 50% on MultiChallenge. Claude 3.5 Sonnet (October 2024) achieved only 41.4% average accuracy as the top performer.

### MT-Eval (Kwan et al. 2024)

Published at EMNLP 2024. Categorizes multi-turn interaction patterns into four types: recollection (recall from previous turns), expansion (addressing queries on the same topic), refinement (adhering to progressively complex instructions), and follow-up (responding to queries building on prior responses). 1170 multi-turn queries with single-turn versions for comparison. Key finding: distance to relevant content and susceptibility to error propagation are the primary factors influencing multi-turn performance.

### MT-Bench (Zheng et al. 2023)

The foundational multi-turn evaluation benchmark, published at NeurIPS 2023. Introduced LLM-as-a-judge methodology alongside Chatbot Arena. Strong LLM judges like GPT-4 match human agreement at 80%+. Laban et al. distinguish their work from MT-Bench by focusing on underspecified, non-episodic conversations rather than episodic subtasks.

---

## The Compounding Error Problem

The multi-turn degradation connects to a broader phenomenon in autoregressive generation. Each step in a chain of reasoning carries a probability of being wrong, and compound failures accumulate: even a 1% error rate per token can escalate to an 87% chance of error by the 200th token. In multi-turn conversations, this manifests as cascading deviations -- one early token difference compounds across turns, and the model's inability to "start fresh" within a conversation means errors become self-reinforcing.

This is distinct from context length degradation (the "lost in the middle" problem), though both contribute to multi-turn failure. The compounding error effect is about the generative process itself, not about retrieval from context.

---

## Practical Mitigation Strategies (Beyond the Paper)

### Architecture-Level Solutions

**Concat-and-Retry (most effective known strategy)**: Gather all requirements across turns, then submit as a single consolidated prompt to a fresh model instance. Restores approximately 95% of single-turn performance. This is essentially what the paper's "Concat" condition demonstrates.

**Mediator-Assistant Pattern**: Separate intent understanding from task execution. The Mediator rewrites ambiguous multi-turn input into explicit single-turn instructions before the Assistant processes them. Recovers approximately 20 percentage points (Liu et al. 2026).

**Observation Masking (for agents)**: Preserve reasoning and action history while replacing older environment observations with placeholders. JetBrains research on SWE-bench found this reduces costs by 50%+ while outperforming LLM summarization in 4/5 configurations. LLM-generated summaries actually caused trajectory elongation (13-15% longer agent runs).

**Masking-First Hybrid**: Use observation masking as the primary strategy, applying LLM summarization only when masking is insufficient. Combines efficiency with reasoning preservation.

### State Management Approaches

**Sliding Window with Summarization**: Compress older conversation segments while preserving recent exchanges verbatim. Used in Google's Agent Development Kit (ADK).

**Dual-Stream Architecture (CaveAgent)**: Decouple state management into a lightweight semantic stream for reasoning and a persistent, deterministic Python Runtime stream for execution.

**Evolving Context Playbooks (ACE Framework)**: Treat contexts as evolving playbooks that accumulate, refine, and organize strategies through generation, reflection, and curation. Prevents context collapse with structured, incremental updates.

### User-Level Strategies

1. Start fresh conversations rather than continuing when context derails
2. Ask the model to summarize everything discussed, then bring that summary to a new conversation
3. Keep instructions concise and focused to reduce assumption-building opportunities
4. Consolidate requirements before asking for generation

---

## Paper Limitations (Acknowledged by Laban et al.)

1. Automated simulation with GPT-4o-mini user, not natural human conversation. Conversations follow a narrow structure lacking real-world nuance, misunderstandings, and user abandonment.
2. Benign testing ground: perfect information revelation, no derailing, guaranteed solvability. Real-world degradation is likely worse.
3. Restricted to analytical tasks with clear solutions; excludes creative writing and open-ended generation.
4. English and text-only; no multilingual or multimodal evaluation.
5. Six tasks, while representative, do not cover the full spectrum of LLM applications.

---

## Implications by Stakeholder

### For System/Agent Builders
Agent frameworks can partially offset multi-turn limitations through turn repetition. Recap is impractical; Snowball offers realistic 15-20% recovery. But native LLM multi-turn support remains valuable because agent scaffolding adds complexity and cost.

### For LLM Builders
The primary call-to-action is joint optimization of aptitude AND reliability. The challenge criteria proposed: similar aptitude in single/multi-turn settings, unreliability (U10-90) below 15 in multi-turn, and performance maintained at unmodified T=1.0. Making early assumptions is rational under current RLHF training -- changing this behavior requires rethinking the reward signal.

### For NLP Practitioners
Adopt instruction sharding methodology for evaluation benchmarks. Release sharded versions alongside fully-specified versions. Use the three-property framework (generative, multi-faceted, non-decomposable) to guide task selection for multi-turn evaluation.

### For Users
If time allows, start new conversations rather than continuing lost ones (randomness may help). Consolidate all requirements before retrying. Ask for summaries mid-conversation.

---

## Open Questions and Future Directions

1. How does multi-turn degradation interact with multimodal inputs (images, code outputs)?
2. Can RLHF training be modified to reward "asking for clarification" over "premature helpfulness"?
3. Does instruction-tuning on multi-turn conversation data (rather than single-turn) reduce the gap?
4. How does degradation manifest in creative/open-ended tasks vs. analytical ones?
5. Can chain-of-thought prompting or explicit "uncertainty markers" help models avoid premature commitment?
6. What is the interaction between context window size and multi-turn degradation (independent phenomena or correlated)?
7. How do multimodal models handle multi-turn degradation when visual context provides grounding?

---

## Source Log

| # | URL | Status | Relevance | Key Finding |
|---|-----|--------|-----------|-------------|
| 1 | https://arxiv.org/html/2505.06120v1 | fetched | high | Primary paper: full methodology, results for 15 models, 6 tasks, 5 conversation types |
| 2 | https://arxiv.org/html/2602.07338v1 | fetched | high | Follow-up: intent mismatch reframing, Mediator-Assistant framework, 20pp recovery |
| 3 | https://github.com/microsoft/lost_in_conversation | fetched | high | Code/data release, 217 stars, 7 tasks in code, Streamlit viewer |
| 4 | https://openreview.net/forum?id=VKGTGGcwl6 | fetched | medium | ICLR 2026 oral acceptance confirmed |
| 5 | https://the-decoder.com/even-frontier-llms-from-gpt-5-onward-lose-up-to-33-accuracy-when-you-chat-too-long/ | fetched | high | 2026 re-run: degradation 39% -> 33%, GPT-5.2 and Claude 4.6 tested |
| 6 | https://www.prompthub.us/blog/why-llms-fail-in-multi-turn-conversations-and-how-to-fix-it | fetched | medium | Practical mitigations: concat-and-retry as primary strategy |
| 7 | https://www.getmaxim.ai/blog/from-turn-1-to-turn-10-how-llms-get-lost-in-multi-turn-conversations/ | fetched | medium | Maxim analysis: unreliability as first-class design objective |
| 8 | https://nlp.elvissaravia.com/p/llms-get-lost-in-multi-turn-conversation | fetched | medium | NLP newsletter: practitioner recommendations |
| 9 | https://wand.ai/blog/compounding-error-effect-in-large-language-models-a-growing-challenge | fetched | medium | Compounding errors: 1% per-token -> 87% by token 200 |
| 10 | https://arxiv.org/abs/2501.17399 | fetched | medium | MultiChallenge: 4 challenge categories, all frontier models below 50% |
| 11 | https://aclanthology.org/2025.findings-acl.958/ | search result | medium | MultiChallenge ACL 2025 Findings publication |
| 12 | https://github.com/KwanWaiChung/MT-Eval | search result | medium | MT-Eval benchmark: 4 interaction pattern types |
| 13 | https://arxiv.org/abs/2306.05685 | search result | medium | MT-Bench: foundational multi-turn benchmark with LLM-as-judge |
| 14 | https://blog.jetbrains.com/research/2025/12/efficient-context-management/ | fetched | high | Agent context management: observation masking > summarization, SWE-bench results |
| 15 | https://www.semanticscholar.org/paper/bb5d81576c113f0b16234fd0db4238a4281c8388 | fetched (empty) | low | Citation data unavailable from fetch |
| 16 | https://x.com/PhilippeLaban/status/2026329136864645390 | search result | medium | Laban 2026 update announcement: "LLMs *Still* Get Lost" |
| 17 | https://x.com/PhilippeLaban/status/1921936307816694206 | search result | low | Original paper announcement thread |
| 18 | https://medium.com/@reza.arani/are-large-language-models-really-lost-in-multi-turn-conversations-0f2980ab25af | search result (403) | medium | Critical analysis: prompt design flaws, methodology concerns |
| 19 | https://www.microsoft.com/en-us/research/publication/llms-get-lost-in-multi-turn-conversation/ | search result | low | Microsoft Research publication page |
| 20 | https://huggingface.co/datasets/microsoft/lost_in_conversation | search result | medium | HuggingFace dataset release |
| 21 | https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/ | search result | medium | Google ADK context engineering |
| 22 | https://arxiv.org/abs/2510.04618 | search result | medium | ACE framework: evolving context playbooks |
| 23 | https://arxiv.org/html/2601.01569v1 | search result | medium | CaveAgent: dual-stream context architecture |
| 24 | https://www.aimodels.fyi/papers/arxiv/llms-get-lost-multi-turn-conversation | search result | low | Paper summary aggregator |
| 25 | https://scale.com/leaderboard/multichallenge | search result | medium | MultiChallenge leaderboard |

## Research Context

- **Query**: Multi-turn conversation degradation in LLMs, focusing on Laban et al. (2505.06120) plus deeper methodology, model comparisons, mitigations, and follow-up work
- **Depth**: deep (auto-detected from multi-faceted scope + user request to "dig deeper")
- **Existing vault knowledge**: Zero notes on this topic; entirely new ground
- **Knowledge gap addressed**: Complete coverage of the multi-turn degradation phenomenon, from primary evidence through root cause analysis, evaluation landscape, practical mitigations, and emerging architectural solutions
