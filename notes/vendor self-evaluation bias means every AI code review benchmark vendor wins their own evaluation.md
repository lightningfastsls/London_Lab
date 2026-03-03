---
description: "Greptile scored 82 percent in own benchmark but 45 percent from Augment Code on the same repos — benchmark survey of 99 papers shows critical gaps in runtime metrics and security coverage"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[code-review-governance]]"
---

# Vendor self-evaluation bias means every AI code review benchmark vendor wins their own evaluation

The vendor self-evaluation problem in AI code review is stark and quantifiable. Greptile reported an 82% catch rate in its own benchmark — but when Augment Code ran an evaluation on the same 5 repositories, Greptile scored only 45%. Same tool, same repos, wildly different results depending on who designs and runs the benchmark.

This is not unique to Greptile. A survey of code review benchmarks spanning 99 papers (2015-2025) reveals systematic limitations: no runtime metrics (build success, test verification), inconsistent granularity (chunk-level to PR-level), and underrepresentation of security detection tasks. The field has shifted from classification tasks to generative peer review, with ~60% of LLM-era datasets focused on peer review, but evaluation methodology has not kept pace.

DeepSource attempted to address the bias by using 165 real CVEs from the OpenSSF dataset — a third-party vulnerability database that no vendor can optimize for. This "external ground truth" approach is more credible but still incomplete: CVEs represent security findings, not the broader spectrum of code quality issues.

Additional data reinforces this: AIMultiple's independent 2026 evaluation of 309 PRs scored CodeRabbit at 4/5 correctness and 4/5 actionability but only 1/5 completeness and 2/5 depth — a profile very different from CodeRabbit's own marketing. In Greptile's benchmark, CodeRabbit achieved only 44% catch rate versus Greptile's 82%. Cursor's BugBot saw resolution rates improve from 52% to 70%+ — but some developers migrated to Claude Code GitHub Actions, citing better detection at 5.5x fewer tokens.

The implication for practitioners is that vendor-published benchmarks should be treated as marketing material, not evidence. Any credible evaluation must either use independent datasets (OpenSSF, real-world PRs from non-affiliated projects) or independent evaluators (third parties running the same methodology). The parallel to academic research is instructive: peer review exists precisely because self-evaluation is unreliable.

This finding has relevance beyond code review tools. Any domain where vendors self-benchmark — including the bioacoustic tools evaluated in this vault's USV detection work — faces the same structural incentive to optimize benchmarks rather than real-world performance. The evaluation gap is compounded by the fact that since [[code review provides more value through knowledge transfer and team awareness than through defect detection]], vendor benchmarks measure only the defect detection dimension while ignoring the knowledge transfer dimension entirely.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[AI review false positive rates of 60-80 percent erode developer trust with concise comments 3x more likely to be acted upon]] -- the real-world metric vendors have incentive to obscure
- [[automated code review increases PR closure time by 42 percent despite 74 percent comment acceptance rate]] -- independent field study as counter-example to vendor claims
- [[code review provides more value through knowledge transfer and team awareness than through defect detection]] -- vendor benchmarks only measure defect detection, missing knowledge transfer entirely

Topics:
- [[agent-governance]]
