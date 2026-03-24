---
description: "MT-Bench and chatbot arenas test turn-independent subtasks — Laban's non-episodic sharding reveals the actual 39% degradation hidden by episodic evaluation"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[multi-turn-degradation]]"
---

# Most existing multi-turn benchmarks are episodic and overestimate real multi-turn performance

Laban et al. argue that the foundational multi-turn benchmarks — MT-Bench (Zheng et al. 2023, NeurIPS), chatbot arenas, and similar evaluations — are predominantly episodic: each turn asks a largely self-contained question or subtask that can be evaluated independently. Models score well on these benchmarks because the multi-turn structure does not require fusing information across turns.

However, real-world multi-turn interactions — debugging sessions, iterative document editing, requirements gathering, research discussions — are fundamentally non-episodic. Information revealed in turn 3 fundamentally changes the correct response to turn 1's question. The instruction sharding methodology addresses this gap by creating non-episodic, underspecified conversations where performance depends on integrating all prior context.

MultiChallenge (Sirdeshmukh et al. 2025, ACL Findings) corroborates this: despite near-perfect scores on existing benchmarks, all frontier models scored below 50% on their non-episodic challenges. Claude 3.5 Sonnet achieved only 41.4% as top performer. MT-Eval (Kwan et al. 2024, EMNLP) found that distance to relevant content and susceptibility to error propagation are the primary factors — both of which are masked by episodic evaluation designs.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[episodic multi-turn tasks that decompose into independent subtasks overestimate LLM multi-turn performance]] -- the episodic resilience finding
- [[tasks vulnerable to multi-turn degradation are generative and non-episodic requiring information fusion across turns]] -- what makes tasks vulnerable
- [[instruction sharding methodology enables controlled comparison between single-turn and multi-turn LLM performance]] -- the methodology that exposes the gap

Topics:
- [[agent-cognition]]
