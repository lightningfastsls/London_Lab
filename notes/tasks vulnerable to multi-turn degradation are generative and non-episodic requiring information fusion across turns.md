---
description: "Three properties predict susceptibility: generative (editing/creating content), multi-faceted (many specifications), and non-decomposable (shards alter the entire solution)"
type: pattern
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# Tasks vulnerable to multi-turn degradation are generative and non-episodic requiring information fusion across turns

Laban et al. identify three properties that predict a task's vulnerability to multi-turn degradation: (1) generative — requiring editing or creating content rather than extracting answers, (2) multi-faceted — having multiple explicit specifications that create numerous shards, and (3) non-decomposable — where revealing one shard fundamentally alters the entire solution space.

Data-to-Text showed consistent significant drops across all models, because it requires generating structured text that must satisfy multiple constraints revealed incrementally. Summary tasks also degraded heavily. In contrast, Actions tasks (API function calling) showed minimal degradation for some models, and Code tasks preserved better for Claude 3.7 Sonnet and GPT-4.1.

The paper frames this as a non-episodic vs. episodic distinction: episodic tasks can be evaluated turn-by-turn independently (each turn is self-contained), while non-episodic tasks require fusing information across the full conversation history. Since [[episodic multi-turn tasks that decompose into independent subtasks overestimate LLM multi-turn performance]], the vulnerability framework helps predict which real-world applications will suffer most.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[episodic multi-turn tasks that decompose into independent subtasks overestimate LLM multi-turn performance]] -- the resilient counterpart
- [[most existing multi-turn benchmarks are episodic and overestimate real multi-turn performance]] -- why this matters for evaluation
- [[LLMs prematurely commit to incorrect solutions in early turns and fail to revise them producing cascading errors]] -- the mechanism that makes non-decomposable tasks fail

Topics:
- [[agent-cognition]]
