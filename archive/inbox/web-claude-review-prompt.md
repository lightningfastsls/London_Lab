# Prompt for Web Claude — Review Extraction Plan

Copy everything below this line into web Claude:

---

I'm building a knowledge graph for my USV (ultrasonic vocalization) research project. The knowledge graph uses atomic notes where each note title IS a claim (a complete proposition you could agree or disagree with). I use a processing pipeline: /learn (research) -> /reduce (extract atomic claims) -> /reflect (connect them).

I just ran /learn on multi-turn LLM conversation degradation and produced a rich source file. Now I need to /reduce it into atomic notes. My AI agent proposed 23 extraction candidates, but I don't know the paper well enough to judge whether these are the RIGHT 23 claims to extract.

**Please read the primary paper** and give me your opinion:
- Paper: https://arxiv.org/html/2505.06120v1 (Laban et al., "LLMs Get Lost In Multi-Turn Conversation")
- Follow-up: https://arxiv.org/html/2602.07338v1 (Liu et al., "Intent Mismatch Causes LLMs to Get Lost")

**Here are the 23 proposed extractions:**

**Core findings (8):**
1. LLMs lose 39 percent average performance when tasks are distributed across conversational turns
2. multi-turn degradation is primarily a 112 percent increase in unreliability rather than capability loss
3. even two conversational turns trigger multi-turn degradation regardless of task complexity
4. tasks vulnerable to multi-turn degradation are generative multi-faceted and non-decomposable
5. episodic decomposable tasks like translation show no multi-turn degradation
6. compounding errors in autoregressive generation escalate per-token errors across turns distinct from context length degradation
7. 2026 re-run with GPT-5.2 and Claude 4.6 reduced multi-turn degradation from 39 to 33 percent but the structural problem persists
8. approximately 60 percent of multi-turn degradation is constant across model sizes contradicting scaling as solution

**Root causes (4):**
9. RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses
10. answer bloat compounds multi-turn errors as responses grow verbose without pruning incorrect assumptions
11. LLMs exhibit conversation-level primacy-recency bias neglecting information revealed in middle turns
12. reasoning models generate 33 percent longer responses and additional test-time compute does not solve multi-turn unreliability

**Mitigations (6):**
13. concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance
14. snowball turn-by-turn accumulation recovers 6-7 percentage points as a realistic production mitigation
15. Mediator-Assistant framework separates intent understanding from task execution recovering approximately 20 percentage points
16. observation masking outperforms LLM summarization for agent context management while reducing costs by 50 percent
17. temperature reduction to zero only improves multi-turn reliability by 15-20 percent because conversation structure itself introduces variation
18. RAG-based memory provides only 3 percent improvement versus intent resolution demonstrating that retrieving context is not equivalent to resolving intent

**Method (1):**
19. instruction sharding methodology enables controlled comparison between single-turn and multi-turn LLM performance

**Tension (1):**
20. the 39 percent degradation figure may overstate the problem for well-designed systems while understating it for messy real-world interactions

**Open questions (2):**
21. whether RLHF can be modified to reward clarification-seeking over premature helpfulness in multi-turn settings
22. whether context window size and multi-turn degradation are independent or correlated phenomena

**Implementation idea (1):**
23. dual-stream architecture decoupling semantic reasoning from deterministic execution prevents context collapse in multi-turn agents

**What I need from you:**
1. After reading the paper, are these 23 the RIGHT claims to extract? Did I miss anything important?
2. Are any of these wrong or misleading based on what the paper actually says?
3. Should any be merged (too granular) or split (too broad)?
4. Are the claim titles accurate and precise, or do any need rewording?
5. For my context: I use Claude Code as my AI agent, so claims about agent workflow design are directly actionable for me. The vault also tracks USV bioacoustic research — would any of these claims connect to how I design my analysis pipeline?

Please be specific and cite sections of the paper where relevant.
