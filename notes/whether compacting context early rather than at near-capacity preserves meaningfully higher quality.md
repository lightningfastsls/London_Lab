---
description: "Open question — if degradation is already underway at 60-70 percent capacity then auto-compact triggers at 75-95 percent may fire after quality has already declined"
type: open-question
confidence: speculative
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Whether compacting context early rather than at near-capacity preserves meaningfully higher quality

Current auto-compact implementations trigger at 75-95% of context capacity. But since [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] and since [[Claude Sonnet exhibits a qualitative performance cliff at 147K-152K tokens which is 73-76 percent of its 200K window]], quality may already be declining by the time compaction fires. The question is whether compacting earlier — at 50-60% capacity, before quality degradation begins — would preserve higher quality in the compacted summary and in subsequent generation.

The argument for early compaction: if the model performs better at lower context utilization, the summary it produces at 50% capacity will be higher quality than the summary it produces at 90% capacity. Since [[context compaction quality degrades cumulatively with multiple compressions regardless of implementation]], a single high-quality early compaction might outperform multiple low-quality late compactions.

The argument against: early compaction means more frequent compaction, which means more cumulative compression loss. And it discards recent context that might still be valuable. The optimal trigger point is likely task-dependent — long exploratory sessions might benefit from early compaction while focused implementation sessions might be better served by completing before any compaction is needed.

No empirical study has directly compared compaction timing's effect on downstream task quality. This is testable: run identical tasks with compaction triggered at different capacity thresholds and measure completion quality.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[context compaction quality degrades cumulatively with multiple compressions regardless of implementation]] -- the cumulative degradation concern
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- the degradation onset that motivates earlier compaction
- [[Claude Sonnet exhibits a qualitative performance cliff at 147K-152K tokens which is 73-76 percent of its 200K window]] -- the cliff that late compaction might not prevent
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- the avoidance alternative: subagents sidestep the compaction question entirely by using fresh context per task

Topics:
- [[agent-cognition]]
