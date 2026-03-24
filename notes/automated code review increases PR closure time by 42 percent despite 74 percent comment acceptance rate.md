---
description: "ICSE 2025 field study at Beko across 4335 PRs — PR closure went from 5h52m to 8h20m, human review volume unchanged at 3.65 comments per PR, 68.8 percent saw minor quality improvement"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[code-review-governance]]"
---

# Automated code review increases PR closure time by 42 percent despite 74 percent comment acceptance rate

The most rigorous field study of automated code review deployment to date — Beko's use of Qodo PR-Agent (GPT-4-32K) across 4,335 PRs, published at ICSE 2025 — reveals a counter-intuitive result: automated review INCREASED pull request closure time from 5 hours 52 minutes to 8 hours 20 minutes, a 42% increase, despite 73.8% of automated comments being labeled as resolved (developers implemented suggestions).

Three findings make this study important. First, the high acceptance rate (73.8%) demonstrates that automated suggestions have genuine value — developers are not dismissing them. Second, the time increase suggests that processing automated feedback adds overhead that exceeds the time savings from catching issues earlier. Developers now have two review streams to respond to (automated and human), and the automated stream adds work without reducing the human stream — no statistically significant change in human review volume (3.65 comments/PR before and after).

Third, the perceived quality improvement was modest: 68.8% reported minor improvement, not transformative. Developer concerns centered on unnecessary suggestions, out-of-scope recommendations, and potential over-reliance.

This finding challenges the implicit assumption that "more review = better outcomes." The mechanism is likely that automated review generates additional work for developers (reading, evaluating, implementing suggestions) without displacing existing human review effort. Since [[AI review false positive rates of 60-80 percent erode developer trust with concise comments 3x more likely to be acted upon]], the noise from automated review consumes developer attention even when the signal is present.

The practical implication: optimizing for useful-comment rate matters more than maximizing coverage. Tools that generate fewer, higher-quality comments may improve velocity more than tools that catch everything. The time increase from automated review parallels the cost gradient insight: since [[the cost gradient from thought to commit means errors caught earlier cost exponentially less to fix]], adding a review layer should catch errors at a cheaper point on the gradient -- but if the review layer itself adds more work than it saves, the net effect shifts the cost rightward rather than leftward.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[AI review false positive rates of 60-80 percent erode developer trust with concise comments 3x more likely to be acted upon]] -- the noise problem that drives time increase
- [[code review provides more value through knowledge transfer and team awareness than through defect detection]] -- the human review value that automated tools cannot displace
- [[LLM-assisted review works best as complement in AI-led co-reviewer or interactive on-demand mode not as replacement]] -- augmentation framing
- [[the cost gradient from thought to commit means errors caught earlier cost exponentially less to fix]] -- review should shift error detection leftward, but noise can shift cost rightward

Topics:
- [[agent-governance]]
