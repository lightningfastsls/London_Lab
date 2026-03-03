---
description: Empirical foundations, multi-agent architectures, cost optimization, effectiveness research, and tooling for AI-assisted code review as a governance mechanism
type: moc
parent_map: "[[agent-governance]]"
---

# code-review-governance

How code review functions as a governance mechanism for AI coding agents -- the empirical case for independent review, multi-agent review architectures, cost optimization strategies, effectiveness research, and available tooling. Split from [[agent-governance]] because code review is the dominant application domain where agent governance theory meets deployment practice.

## Synthesis

The empirical foundation is stark: since [[same-model generation and review creates confirmation bias producing 8x duplicated code and 72 percent Java security failures]], independent review is not optional for AI-generated code. The convergent architectural pattern is structural role separation -- whether through ASDLC context swaps, blackboard architectures, or pipeline invariants -- enforcing that generators cannot approve their own work. This is economically viable because [[model cascading routes 70-90 percent of review to cheap models achieving 60-87 percent cost reduction]] and [[layered review depth tiering mirrors human review practice by matching investment to complexity]]. However, effectiveness research reveals a paradox: since [[automated code review increases PR closure time by 42 percent despite 74 percent comment acceptance rate]], more review does not automatically mean better outcomes, and since [[AI review false positive rates of 60-80 percent erode developer trust with concise comments 3x more likely to be acted upon]], signal quality matters more than coverage.

## Empirical Foundation

- [[same-model generation and review creates confirmation bias producing 8x duplicated code and 72 percent Java security failures]] -- the core empirical case for independent review
- [[AI code generation caused 4x increase in code cloning and first-ever dominance of copy-paste over moved code]] -- GitClear 2025 code quality trend showing downstream effects

## Multi-Agent Review Architectures

- [[fresh context swap between generation and review eliminates conversation drift and confirmation bias]] -- ASDLC context swap pattern
- [[adversarial builder-critic separation catches silent performance risks that pass all tests]] -- what deterministic gates miss
- [[multi-agent debate with circuit breaker prevents infinite review loops while 3-7 agents achieves optimal accuracy-to-cost ratio]] -- Nielsen debate architecture
- [[3-5 actor-critic review rounds eliminate over 90 percent of issues at under 2 dollars per feature]] -- cost-effectiveness data
- [[memory wipe per review turn prevents attention degradation treating each attempt as fresh start guided by coach feedback]] -- Block g3 dialectical pattern
- [[no instruction path from failure to commit is the critical safety invariant in automated code pipelines]] -- metaswarm pipeline safety
- [[supervisory QA-Checker agent monitoring conversation prevents prompt drifting improving vulnerability confirmation from 73 to 93 percent]] -- novel oversight agent pattern

## Cost & Efficiency

- [[model cascading routes 70-90 percent of review to cheap models achieving 60-87 percent cost reduction]] -- cost optimization via tiered models
- [[pre-bundling diffs into single context reduces review tool calls from 100-plus to a few]] -- token optimization technique
- [[layered review depth tiering mirrors human review practice by matching investment to complexity]] -- 3-tier depth matching

## Effectiveness Research

- [[automated code review increases PR closure time by 42 percent despite 74 percent comment acceptance rate]] -- counter-intuitive ICSE 2025 finding
- [[AI review false positive rates of 60-80 percent erode developer trust with concise comments 3x more likely to be acted upon]] -- the noise-trust problem
- [[vendor self-evaluation bias means every AI code review benchmark vendor wins their own evaluation]] -- evaluation methodology warning
- [[code review follows orientation then analytical phases where skipping orientation degrades analytical quality]] -- CRDM cognitive model
- [[code review provides more value through knowledge transfer and team awareness than through defect detection]] -- reframes review purpose
- [[LLM-assisted review works best as complement in AI-led co-reviewer or interactive on-demand mode not as replacement]] -- two effective integration modes
- [[using AI agents effectively is fundamentally a code review skill requiring hourly pattern recognition for suspicious behavior]] -- agent supervision as review skill
- [[whether accepted or rejected AI review comments should feed back into agent learning through persistent memory]] -- review-to-learning feedback loop

## Tools & Benchmarks

- [[Greptile full codebase indexing with code graph achieves 82 percent bug catch rate through multi-hop investigation]] -- code graph architecture for review
- [[WarpGrep RL-trained search subagent lifts SWE-Bench Pro by 11.6 percentage points while reducing cost 15.6 percent and improving speed 28 percent]] -- RL-trained search as orthogonal capability multiplier
- [[Claude Code GitHub Actions provides official automated PR review with automatic mode detection responding to mentions assignments and triggers]] -- official Anthropic review automation
- [[whether specialization across multiple AI tools via MCP orchestration outperforms monolithic agent approaches for complex coding tasks]] -- open question on specialization vs monolithic

## Related Areas

- [[agent-governance]] -- parent: the formal frameworks, practitioner patterns, and convergence theory that code review implements
- [[agent-cognition]] -- root causes (multi-turn degradation, confirmation bias) that motivate review architectures
- [[context-management]] -- fresh context patterns function as both context management AND implicit governance

---

Topics:
- [[index]]
- [[agent-governance]]
