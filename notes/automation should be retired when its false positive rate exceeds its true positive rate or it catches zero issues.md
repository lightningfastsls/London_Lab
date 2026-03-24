---
description: "automation hygiene principle — hooks and checks that never fire or mostly false-alarm should be retired before they teach agents to ignore all gates"
type: claim
confidence: likely
created: 2026-03-08
topics:
  - "[[agent-external-cognition]]"
---

# automation should be retired when its false positive rate exceeds its true positive rate or it catches zero issues

Automation that produces more false positives than true positives trains agents to ignore its outputs — the automation equivalent of alarm fatigue. Since [[fewer well-placed activation triggers outperform many ignored ones because noise teaches agents to skip gates]], keeping noisy automation active is worse than having no automation at all. Similarly, automation that catches zero issues over a sustained period may indicate the problem it guards against has been structurally eliminated.

The retirement principle is: monitor automation hit rates, and retire checks whose signal-to-noise ratio has degraded below usefulness. Since [[nudge theory explains graduated hook enforcement as choice architecture for agents]], this applies to graduated enforcement too — if the warning tier never escalates, the problem may not exist. Retirement is not failure; it is recognition that the automation served its purpose and the environment has changed.

---

Relevant Notes:
- [[fewer well-placed activation triggers outperform many ignored ones because noise teaches agents to skip gates]] — noise-driven gate bypass
- [[nudge theory explains graduated hook enforcement as choice architecture for agents]] — graduated enforcement design
- [[hook-driven learning loops create self-improving methodology through observation accumulation]] — automation as learning infrastructure
